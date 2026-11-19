"""Replay the one bundled synthetic example without using its answer."""
import argparse
from pathlib import Path
import json
from aircraft_doppler.wide import infer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Use a new/empty directory")
    args = parser.parse_args()
    fixture = Path(__file__).resolve().parent / "example"
    result = infer(fixture/"synthetic.wav", fixture/"tracks.csv", fixture/"candidates.csv",
                   args.output, hardened=True)
    print(json.dumps({field: result.get(field) for field in (
        "status", "evidence_grade", "estimated_xyz_m", "estimated_pair",
        "estimated_clock_offset_s", "audio_event_separation_s", "abstention_reasons")}, indent=2))
    return 0 if "estimated_xyz_m" in result else 2


if __name__ == "__main__":
    raise SystemExit(main())
