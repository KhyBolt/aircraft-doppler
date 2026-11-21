"""Fetch or verify the two original WAV members used in the Livermore replay."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import wave
from zipfile import ZipFile

from benchmarks.range_wav import RangeFile


ARCHIVE = "https://zenodo.org/api/records/24075029/files/raw_flight_data.zip/content"
ARCHIVE_BYTES = 132_352_360
TRANSFER_LIMIT = 32 * 2**20
CLIPS = (
    (
        "N9393T_R02_2026-05-17T22-03-02.336Z.wav",
        "7834bad6d620df2143b396325fd861d6581d5ad25dd9bcfa997eb953d13a65f5",
        6_599_424,
    ),
    (
        "N86LP_R01_2026-05-17T22-20-09.578Z.wav",
        "a19c78d9d147a6e8b4b2c495bd344787363875ea411e3af21b59490517832495",
        5_493_505,
    ),
)
CATALOGUES = (
    (
        date(2026, 5, 17),
        177_352,
        "d29fb193d0ae54f4f74da4c8186bb11e861d32e4d62332b9abc20a7495d0af47",
    ),
    (
        date(2026, 5, 18),
        124_713,
        "bbdc5046a0456e62f95c3f1bbd3bff52ab31651d7f08ca4facc4050f660c1c5e",
    ),
)


def verify(path, expected_sha256, expected_frames):
    with path.open("rb") as source:
        actual_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{path.name}: unexpected WAV SHA-256 {actual_sha256}")
    with wave.open(str(path), "rb") as source:
        sample_count = source.getnframes()
        sample_rate_hz = source.getframerate()
        channels = source.getnchannels()
    if (sample_count, sample_rate_hz, channels) != (expected_frames, 48_000, 1):
        raise ValueError(f"{path.name}: unexpected WAV format")
    return {
        "filename": path.name,
        "sha256": actual_sha256,
        "sample_rate_hz": sample_rate_hz,
        "sample_count": sample_count,
        "channels": channels,
    }


def stage(output):
    output.mkdir(parents=True, exist_ok=True)
    missing = [clip for clip in CLIPS if not (output / clip[0]).is_file()]
    if not missing:
        return 0

    archive_file = RangeFile(ARCHIVE, TRANSFER_LIMIT)
    if archive_file.size != ARCHIVE_BYTES:
        raise ValueError("Unexpected Zenodo archive size")
    with ZipFile(archive_file) as archive:
        for name, sha256, frames in missing:
            members = [
                member for member in archive.infolist()
                if not member.is_dir() and PurePosixPath(member.filename).name == name
            ]
            if len(members) != 1:
                raise ValueError(f"Expected one ZIP member named {name}, found {len(members)}")
            member, = members
            if member.file_size > TRANSFER_LIMIT:
                raise ValueError(f"{name}: ZIP member exceeds the extraction limit")
            with tempfile.TemporaryDirectory(dir=output) as temporary:
                path = Path(temporary) / name
                with archive.open(member) as source, path.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=2**20)
                verify(path, sha256, frames)
                path.replace(output / name)
    return archive_file.fetched


def verify_catalogues():
    from benchmarks.livermore_archive import fetch

    def receipt(clock):
        rows, response_bytes, digest = fetch(clock)
        return clock[1], len(rows), digest, response_bytes

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = []
        for day, expected_count, expected_digest in CATALOGUES:
            midnight = datetime.combine(day, time.min, timezone.utc).timestamp()
            shards = list(pool.map(receipt, ((midnight, hour) for hour in range(24))))
            canonical = [shard[:3] for shard in shards]
            digest = hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()
            count = sum(shard[1] for shard in shards)
            if (count, digest) != (expected_count, expected_digest):
                raise ValueError(
                    f"{day}: indexed track inputs differ from the frozen receipt "
                    f"({count} rows, SHA-256 {digest})"
                )
            results.append({
                "date_utc": str(day),
                "shards": len(shards),
                "rows_including_hourly_overlap": count,
                "response_bytes": sum(shard[3] for shard in shards),
                "canonical_row_manifest_sha256": digest,
            })
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/livermore-input"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true", help="Verify local WAVs without network access")
    mode.add_argument("--check-archive", action="store_true", help="Also verify both UTC days of regional ADS-B receipts")
    args = parser.parse_args()

    fetched = 0 if args.verify_only else stage(args.output)
    clips = [verify(args.output / name, sha256, frames) for name, sha256, frames in CLIPS]
    report = {"clips": clips, "archive_bytes_fetched": fetched}
    if args.check_archive:
        report["archive_receipts"] = verify_catalogues()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
