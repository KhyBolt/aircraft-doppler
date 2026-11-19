"""Stream historical readsb archives and save only the regional time slice."""

import argparse
import csv
from datetime import datetime
import gzip
import json
import math
from pathlib import Path
import tarfile
from time import monotonic
from urllib.request import Request, urlopen


class ArchiveStream:
    def __init__(self, urls, max_bytes, max_seconds):
        self.urls = iter(urls)
        self.response = None
        self.bytes_read = 0
        self.bytes_announced = 0
        self.report_at = 256 * 2**20
        self.max_bytes = max_bytes
        self.deadline = monotonic() + max_seconds

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.response is not None:
            self.response.close()

    def read(self, size):
        while monotonic() < self.deadline:
            if self.response is None:
                url = next(self.urls, None)
                if url is None:
                    return b""
                self.response = urlopen(Request(url, headers={
                    "User-Agent": "Aircraft-Doppler-Benchmark/1.0",
                    "Accept-Encoding": "identity",
                }), timeout=30)
                size_header = self.response.headers.get("Content-Length")
                if size_header is None:
                    raise ValueError("ADS-B asset did not declare its transfer size")
                self.bytes_announced += int(size_header)
                if self.bytes_announced > self.max_bytes:
                    raise ValueError("ADS-B assets exceed the stated transfer cap")
            data = self.response.read(size)
            if data:
                self.bytes_read += len(data)
                if self.bytes_read > self.max_bytes:
                    raise ValueError("ADS-B assets exceeded the stated transfer cap")
                if self.bytes_read >= self.report_at:
                    print(f"Streamed {self.bytes_read // 2**20} MiB", flush=True)
                    self.report_at += 256 * 2**20
                return data
            self.response.close()
            self.response = None
        raise TimeoutError("ADS-B acquisition exceeded its time cap")


def utc(value):
    date = datetime.fromisoformat(value)
    if date.utcoffset() is None:
        raise ValueError("Specify an explicit UTC offset")
    return date.timestamp()


def crop(archive, start, stop, latitude, longitude, half_width_m):
    lat_pad = math.degrees(half_width_m / 6371008.8)
    lon_pad = lat_pad / math.cos(math.radians(latitude))
    rows = []
    counts = {"traces_seen": 0, "traces_retained": 0, "points_without_geometric_height": 0}
    for member in archive:
        if not member.isfile() or not Path(member.name).name.startswith("trace_full_"):
            continue
        counts["traces_seen"] += 1
        data = archive.extractfile(member).read()
        if data.startswith(b"\x1f\x8b"):
            data = gzip.decompress(data)
        trace = json.loads(data)
        epoch = trace["timestamp"]
        if epoch > stop:
            continue
        before = len(rows)
        for point in trace["trace"]:
            t = epoch + point[0]
            if t < start or t > stop:
                continue
            lat, lon, altitude, flags = point[1], point[2], point[3], point[6] or 0
            if (altitude == "ground" or flags & 1 or lat is None or lon is None
                    or abs(lat - latitude) > lat_pad or abs(lon - longitude) > lon_pad):
                continue
            geom = point[10] if len(point) > 10 else None
            if geom is None and flags & 8:
                geom = altitude
            if geom is None:
                counts["points_without_geometric_height"] += 1
                continue
            rows.append((trace["icao"], trace.get("t", ""), t, lat, lon, geom * 0.3048,
                         bool(flags & 2)))
        counts["traces_retained"] += int(len(rows) > before)
    if not counts["traces_seen"] or not rows:
        raise ValueError("No usable readsb track points in the prescribed time and area")
    return sorted(rows), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="+")
    parser.add_argument("--start", type=utc, required=True)
    parser.add_argument("--stop", type=utc, required=True)
    parser.add_argument("--center", type=float, nargs=2, required=True, metavar=("LAT", "LON"))
    parser.add_argument("--half-width-km", type=float, default=25)
    parser.add_argument("--max-mib", type=int, required=True)
    parser.add_argument("--max-seconds", type=int, default=360)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stop <= args.start or args.max_mib < 1 or args.max_seconds < 1 or args.half_width_km <= 0:
        parser.error("Use a positive time interval, width and resource cap")
    with ArchiveStream(args.url, args.max_mib * 2**20, args.max_seconds) as stream:
        with tarfile.open(fileobj=stream, mode="r|", bufsize=2**20) as archive:
            rows, counts = crop(archive, args.start, args.stop, *args.center, args.half_width_km * 1000)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(("aircraft_id", "aircraft_type", "utc_s", "latitude", "longitude",
                         "altitude_ellipsoid_m", "new_leg"))
        writer.writerows(rows)
    metadata = {**counts, "retained_points": len(rows), "archive_bytes_streamed": stream.bytes_read,
                "urls": args.url, "center_lat_lon": args.center, "half_width_km": args.half_width_km,
                "start_utc_s": args.start, "stop_utc_s": args.stop,
                "license": "ODbL-1.0", "source": "adsb.lol contributors"}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
