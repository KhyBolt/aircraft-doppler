"""Recreate the regional adsb.lol catalogue alteration in memory.

The CLI prints only the public SQL or a checksum receipt, never track rows.
"""
import argparse
import csv
import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import io
import json
import math
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

XYZ = ["x_m", "y_m", "z_m"]
TRACK_GAP_S = 15
origin = (37.69338888888888, -121.82036111111111, 90.)
sound = 343.
branch_tail = 1027.242 + 137.488
base_endpoint = "https://kvzqttvc2n.eu-west-1.aws.clickhouse-staging.com/?"
read_limit = 8 * 1024**2
digest = lambda b: hashlib.sha256(b).hexdigest()



def geodetic_to_ecef(lat, lon, ellipsoid_height_m):
    lat, lon, h = np.broadcast_arrays(np.deg2rad(lat), np.deg2rad(lon), ellipsoid_height_m)
    if np.any(~np.isfinite(lat)) or np.any(abs(lat) > np.pi / 2) or np.any(~np.isfinite(lon)) or np.any(abs(lon) > np.pi) or np.any(~np.isfinite(h)):
        raise ValueError("Invalid geodetic coordinate")
    a, e2 = 6378137.0, 6.6943799901413165e-3
    n = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    return np.stack(((n+h)*np.cos(lat)*np.cos(lon),
                     (n+h)*np.cos(lat)*np.sin(lon),
                     (n*(1-e2)+h)*np.sin(lat)), axis=-1)


@dataclass
class Track:
    flight_id: str
    t: np.ndarray
    xyz: np.ndarray
    engine: str = "unknown"
    max_gap_s: float = 15.0
    aircraft_id: str | None = None

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float)
        self.xyz = np.asarray(self.xyz, dtype=float)
        if self.t.ndim != 1 or len(self.t) < 4 or self.xyz.shape != (len(self.t), 3):
            raise ValueError("Track needs >=4 observations and an N x 3 position matrix")
        if not np.all(np.isfinite(self.t)) or not np.all(np.isfinite(self.xyz)):
            raise ValueError("Nonfinite flight data")
        if not str(self.flight_id).strip() or "|" in self.flight_id:
            raise ValueError("Track ID is empty or contains reserved separator |")
        if not np.isfinite(self.max_gap_s) or self.max_gap_s <= 0:
            raise ValueError("Maximum track gap must be positive and finite")
        if np.any(np.diff(self.t) <= 0) or np.any(np.diff(self.t) > self.max_gap_s):
            raise ValueError("Track timestamps must increase, with no excessive data gaps; split gaps first")
        self.epoch = float(self.t[0])
        local_t = self.t-self.epoch
        self.spline = CubicSpline(local_t, self.xyz, axis=0, bc_type="natural", extrapolate=False)
        delta = np.diff(local_t)[:, None]
        slopes = self.spline(local_t, 1)
        # Cubic Bézier control hull encloses each spline segment. Its AABB
        # gives a lower bound for the interval minimum without solving roots.
        controls = np.stack((self.xyz[:-1], self.xyz[:-1]+slopes[:-1]*delta/3,
                             self.xyz[1:]-slopes[1:]*delta/3, self.xyz[1:]))
        self.interval_lo = controls.min(axis=0)
        self.interval_hi = controls.max(axis=0)

    def position(self, t):
        return self.spline(np.asarray(t)-self.epoch)

    def velocity(self, t):
        return self.spline(np.asarray(t)-self.epoch, 1)


def airport_frame(origin):
    """Local tangent frame with a common EGM96 vertical datum over the airport area."""
    lat, lon = np.deg2rad(origin[:2])
    rotation = np.array([[-np.sin(lon), np.cos(lon), 0],
                         [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
                         [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]])
    return geodetic_to_ecef(*origin), rotation


ecef,rot=airport_frame(origin)


def catalogue_tracks(path, start, duration, frame, sound_mps):
    """Read a regional civilian CSV with absolute times and a common height datum."""
    data = pd.read_csv(path, dtype={"aircraft_id": str})
    height_names = {"altitude_ellipsoid_m", "altitude_msl_m"} & set(data.columns)
    fields = {"aircraft_id", "utc_s", "latitude", "longitude"}
    if len(height_names) != 1 or not fields <= set(data.columns):
        raise ValueError("Catalogue requires aircraft IDs, UTC seconds, coordinates and one height datum")
    height, = height_names
    if data.aircraft_id.isna().any() or data.aircraft_id.str.contains(r"\|").any():
        raise ValueError("Catalogue has missing or invalid aircraft IDs")
    numbers = ["utc_s", "latitude", "longitude", height]
    data[numbers] = data[numbers].apply(pd.to_numeric)
    if not np.isfinite(data[numbers].to_numpy()).all():
        raise ValueError("Catalogue track samples must be finite")
    data["time_s"] = data.utc_s - start
    data = data[data.time_s.between(-110, duration + 25)].copy()
    if "new_leg" not in data:
        data["new_leg"] = False
    elif data.new_leg.isna().any() or not data.new_leg.isin([True, False, 0, 1]).all():
        raise ValueError("Catalogue leg boundaries must be boolean")
    data["new_leg"] = data.new_leg.astype(bool)
    audit = {"aircraft_in_temporal_crop": int(data.aircraft_id.nunique()),
             "samples_in_temporal_crop": len(data), "height_column": height,
             "short_segments": 0, "non_subsonic_segments": 0, "conflicting_aircraft": 0}
    ecef, rotation = frame
    data[XYZ] = (geodetic_to_ecef(data.latitude, data.longitude, data[height]) - ecef) @ rotation.T
    tracks = []
    for aircraft_id, aircraft in data.groupby("aircraft_id", sort=True):
        aircraft = aircraft.sort_values("time_s")
        aircraft = aircraft.drop_duplicates(subset=["time_s", *XYZ, "new_leg"])
        if aircraft.time_s.duplicated().any():
            audit["conflicting_aircraft"] += 1
            continue
        segment_breaks = (aircraft.time_s.diff().gt(TRACK_GAP_S) | aircraft.new_leg).cumsum()
        for segment, rows in aircraft.groupby(segment_breaks):
            if len(rows) < 4:
                audit["short_segments"] += 1
                continue
            times, xyz = rows.time_s.to_numpy(), rows[XYZ].to_numpy()
            track = Track(f"{aircraft_id}-{segment}", times, xyz, aircraft_id=aircraft_id,
                          max_gap_s=TRACK_GAP_S)
            t = np.linspace(times[0], times[-1], int(np.ceil(np.ptp(times)/.25))+1)
            if np.any(np.linalg.norm(track.velocity(t), axis=1) >= sound_mps):
                audit["non_subsonic_segments"] += 1
                continue
            tracks.append(track)
    audit["retained_segments"] = len(tracks)
    audit["retained_aircraft"] = len({track.aircraft_id for track in tracks})
    return tracks, audit


def fmt(t):return dt.datetime.fromtimestamp(t,dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


def cx(x):return int(4294967295*(x+180)/360)


def cy(y):return int(4294967295*(.5-math.log(math.tan((y+90)*math.pi/360))/(2*math.pi)))


def morton(x,y):return sum((((x>>i)&1)<<(2*i))|(((y>>i)&1)<<(2*i+1)) for i in range(32))


x1,x2,y1,y2=cx(-122.10449073373086),cx(-121.53623148849137),cy(37.91821897982002),cy(37.46855879795775)
shift=23;tiles=[]
for tx in range(x1>>shift,(x2>>shift)+1):
    for ty in range(y1>>shift,(y2>>shift)+1):
        l=morton(tx<<shift,ty<<shift)
        tiles.append(f'(mortonEncode(mercator_x,mercator_y) BETWEEN {l} AND {l+(1<<(2*shift))-1})')


def query_for_shard(day, hour):
    shift=day+hour*3600.
    left=shift-180
    right=min(day+86400.+120,shift+3600.+branch_tail+120)
    dates=sorted({fmt(left).split()[0],fmt(right).split()[0]})
    datelist=', '.join(repr(d) for d in dates)
    query=f"""SELECT icao,toUnixTimestamp64Milli(time),lat,lon,geometric_altitude,altitude,flags,aircraft_type,ground_speed,track_degrees,source,dbFlags,data_source
FROM planes_mercator WHERE date IN ({datelist})
AND time >= toDateTime64('{fmt(left)}',3,'UTC')
AND time < toDateTime64('{fmt(math.ceil(right))}',3,'UTC')
AND data_source='adsb.lol' AND source='adsb_icao' AND bitAnd(flags,1)=0 AND bitAnd(dbFlags,1)=0
AND mercator_x BETWEEN {x1} AND {x2} AND mercator_y BETWEEN {y1} AND {y2}
AND ({' OR '.join(tiles)})
ORDER BY icao,time,lat,lon,geometric_altitude,source
LIMIT 100000 FORMAT JSONCompact"""
    return query


def fetch(date_hour):
    url=base_endpoint+urlencode({'user':'website','query':query_for_shard(*date_hour)})
    for attempt in range(2):
        try:
            with urlopen(url,timeout=24) as f:
                blob=f.read(read_limit+1)
            assert len(blob)<=read_limit
            rows=json.loads(blob)['data']
            assert len(rows)<100000
            if any(len(row)!=13 or row[-1]!='adsb.lol' for row in rows):
                raise ValueError("The archive returned a row outside the adsb.lol-only contract")
            rows=[row[:-1] for row in rows]
            return rows,len(blob),digest(json.dumps(rows,separators=(',',':')).encode())
        except Exception:
            if attempt:raise


def alias(icao):
    return 'F'+digest(('livermore-day-screen-v1:'+icao).encode())[:14]


def catalogue(rows,day,hour):
    start=day+hour*3600
    stop=min(start+3600.,day+86400.-branch_tail)
    grouped=defaultdict(dict)
    lookup={}
    for r in rows:
        if 0<r[4]<40000:grouped[r[0]].setdefault(r[1],r)
    stream=io.StringIO()
    w=csv.writer(stream)
    w.writerow(('aircraft_id','utc_s','latitude','longitude','altitude_ellipsoid_m','new_leg'))
    for icao,items in grouped.items():
        aid=alias(icao)
        lookup[aid]=icao
        prev_in=prev=None;new=True
        for r in items.values():
            t=r[1]/1000
            if prev_in is None or t-prev_in>15. or int(r[6])&2:
                new=True;prev=None
            prev_in=t
            if prev is not None and t-prev<2.:continue
            w.writerow((aid,t,r[2],r[3],r[4]*.3048,new))
            new=False;prev=t
    stream.seek(0)
    epoch=start-180.
    tracks,audit=catalogue_tracks(stream,epoch,stop+branch_tail+120.-epoch,(ecef,rot),sound)
    width=(stop-start)/2.
    tracks=tuple(replace(t,t=t.t-180.-width) for t in tracks)
    return (tracks,lookup,width,start+width,audit)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["2026-05-17","2026-05-18"], required=True)
    parser.add_argument("--hour", type=int, choices=range(24), required=True)
    parser.add_argument("--check", action="store_true", help="Fetch the shard and return only counts and digests")
    args = parser.parse_args()
    day = dt.datetime.fromisoformat(args.date).replace(tzinfo=dt.timezone.utc).timestamp()
    if not args.check:
        print(query_for_shard(day, args.hour))
        return
    rows, bytes_read, row_sha = fetch((day, args.hour))
    records, _, _, _, audit = catalogue(rows, day, args.hour)
    print(json.dumps({
        "date_utc": args.date, "hour_utc": args.hour, "provider": "adsb.lol",
        "raw_rows": len(rows), "response_bytes": bytes_read, "canonical_scoring_rows_sha256": row_sha,
        "admitted_track_segments": len(records), "admitted_rows_before_segment_checks": audit["samples_in_temporal_crop"],
    }, indent=2))

if __name__ == "__main__":
    main()
