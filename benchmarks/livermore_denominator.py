"""Audit the original Livermore replay's ordered-pair set without scoring pitch."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from benchmarks.livermore_replay import (
    CLOCK_STEP_S, EDGE_M, GAP_S, MANIFESTS, curve, load_shards, prepare_shard, write_json,
)


def eligible_pairs(shards, spans):
    eligible = set()
    for north in range(-EDGE_M, EDGE_M + 1, 1000):
        for east in range(-EDGE_M, EDGE_M + 1, 1000):
            for clocks, flights in shards:
                if not len(clocks) or not flights:
                    continue
                arrivals = np.array([curve(flight, (east, north, 0))[0][[0, -1]] for flight in flights])
                first_start, first_end = ((arrivals - spans[0] - clocks[0]) / CLOCK_STEP_S).T
                second_start, second_end = ((arrivals - spans[1] - clocks[0]) / CLOCK_STEP_S).T
                earliest = np.maximum(0, np.ceil(np.maximum(first_start[:, None], second_start)))
                latest = np.minimum(len(clocks) - 1, np.floor(np.minimum(first_end[:, None], second_end)))
                for first, second in zip(*np.nonzero(earliest <= latest)):
                    first_id, second_id = flights[first][0], flights[second][0]
                    if first_id != second_id:
                        eligible.add((first_id, second_id))
    return eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--archive", type=Path, default=Path("runs/livermore-auto/archive"))
    args = parser.parse_args()
    lock = json.loads((args.directory / "fit-lock.json").read_text(encoding="utf-8"))
    if lock["mode"] != "full_day_all_pairs" or lock["input_date_utc"] != "2026-05-17":
        raise ValueError("Expected the original full-day replay output")
    ridges = json.loads((args.directory / "automatic-ridges.json").read_text(encoding="utf-8"))
    ranks = json.loads((args.directory / "coarse-rankings.json").read_text(encoding="utf-8"))
    spans = [np.asarray(row["time_frequency_rows"])[[0, -1], 0] + slot * GAP_S for slot, row in enumerate(ridges)]
    rows, _, manifest = load_shards(args.archive)
    if manifest != MANIFESTS[lock["input_date_utc"]] or manifest != lock["archive_manifest_sha256"]:
        raise ValueError("Archive does not match the replay manifest")
    eligible = eligible_pairs([prepare_shard(shard, hour, None) for hour, shard in enumerate(rows)], spans)
    ranked = [tuple(row["pair_icao"]) for row in ranks]
    if eligible != set(ranked) or len(eligible) != len(ranked) or len(ranked) != lock["unconditioned_distinct_ordered_icao_pairs"]:
        raise ValueError("Rankings omit or duplicate one or more eligible ordered pairs")
    if sorted(ranks, key=lambda row: (row["mse_hz2"], row["pair_icao"])) != ranks:
        raise ValueError("Rankings are not in deterministic score order")
    receipt = {"method": "Intersect the two track arrival-coverage intervals at each node on the discrete clock grid, without pitch scoring.",
               "manifest_sha256": manifest, "distinct_ordered_pairs": len(eligible), "matches_coarse_rankings_exactly": True,
               "sorted_pair_list_sha256": sha256(json.dumps(sorted(eligible), separators=(",", ":")).encode()).hexdigest()}
    write_json(args.directory / "denominator-audit.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
