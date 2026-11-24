# Livermore automatic audio replay

This is a fresh diagnostic for the two study recordings, not a rerun of the
archived solver that reported 1 of 17,101 pairs and 320.619 m. It does not use
the hand-guided pitch traces, reported absolute clip starts, study aircraft
tracks, or phone position as fitting inputs. The date (May 17, 2026 UTC), the
published 1,027.242 s gap between clip starts, and the airport-centered
10-by-10 km search area are supplied priors. The fit fixes the receiver's
up coordinate at 0 m in the local ENU frame centered at latitude
37.693388889 N, longitude 121.820361111 W,
90 m ellipsoid height. This is a tangent plane, not a terrain model.

| Run | Rank of documented ordered pair | Leading training MSE (Hz^2) | Distance to study phone (m) | Held-out blocks passed |
| --- | ---: | ---: | ---: | --- |
| Original, all pairs | 1 of 12,215 | 0.625033 | 404.897 | 2/5 and 2/4 |
| Original, given the matching pair | Not ranked | 0.625033 | 404.897 | 2/5 and 2/4 |
| May 17, swap audio slots | 386 of 12,164 | 2.670267 | 3,462.935 | 0/4 and 2/5 |
| May 18, original audio order | Absent from 5,817 | 2.845233 | 8,179.197 | 2/5 and 2/4 |

The original winning order was N9393T (`ad09eb`) followed by N86LP (`abcd86`).
Its locked point is latitude 37.666097327 N, longitude 121.779078415 W.
It remained first among the 24 locally refined coarse leaders. No run verifies a location. The
reported distances are separations from the study's phone position, not
accuracy measurements. The swapped run keeps the positive gap and tests
the same documented ordered pair, not its reverse; the reversed pair was
not eligible on the coarse lattice. The wrong-day run retains the audio and
all search settings, replacing only the UTC day and its archive.

## Reproduce from the repository root

Use Python 3.12 and `requirements-verified.txt`. These PowerShell commands
assume `python` resolves to that environment. On POSIX, use
`export OPENBLAS_NUM_THREADS=1` and equivalent shell variable syntax. Every
`--output` must name a new directory. The optional `$reference` below is the
portfolio [study reference GeoJSON](https://khybolt.github.io/doppler/assets/livermore/study-reference.geojson),
whose observer is derived from the matching published CSV rows. If that
file is unavailable, omit `--reference $reference`; the fit, rank and
locked predictions do not depend on it, and the distance will be null.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
python -m pip install -r requirements-verified.txt
python -B -m benchmarks.livermore_inputs --output runs/livermore-input
$reference = '../KhyBolt.github.io/doppler/assets/livermore/study-reference.geojson'
python -B -m benchmarks.livermore_replay --audio runs/livermore-input --output runs/livermore-auto/full-day --reference $reference
python -B -m benchmarks.livermore_denominator runs/livermore-auto/full-day
python -B -m benchmarks.livermore_replay --audio runs/livermore-input --output runs/livermore-auto/matched --matched-pair --reference $reference
python -B -m benchmarks.livermore_replay --audio runs/livermore-input --output runs/livermore-auto/control-swapped --swap-clips --reference $reference
python -B -m benchmarks.livermore_replay --audio runs/livermore-input --archive runs/livermore-auto/archive-may18 --output runs/livermore-auto/control-wrong-day --date 2026-05-18 --reference $reference
$env:LIVERMORE_AUDIO_DIR = 'runs/livermore-input'
python -B -m unittest discover -s tests -p test_livermore_replay.py -v
```

The cached original replay took about 84 s on the run machine; the matched
run about 2 s. Acquisition time depends on network access. The replay verifies
the exact WAV SHA-256, 48 kHz mono format, and frame counts in
`livermore_inputs.py`. The WAVs come from Praveen Pathak, *Aeroplane data
full analysis and raw files*, v2, [Zenodo 24075029](https://doi.org/10.5281/zenodo.24075029),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

`livermore_archive.py` regenerates 24 hourly adsb.lol shards, with enough
overlap to cover both clips from every admitted first-start clock. The
manifest is SHA-256 of a compact JSON array of `[hour, row count, shard
SHA-256]`, in hour order. Each row digest uses compact JSON in query order
after removing the verified provider tag. Counts include shard overlap.
An archive whose count or data changes will fail the manifest check.

| UTC date | Shards | Rows | Manifest SHA-256 |
| --- | ---: | ---: | --- |
| 2026-05-17 | 24 | 177,352 | `d29fb193d0ae54f4f74da4c8186bb11e861d32e4d62332b9abc20a7495d0af47` |
| 2026-05-18 | 24 | 124,713 | `bbdc5046a0456e62f95c3f1bbd3bff52ab31651d7f08ca4facc4050f660c1c5e` |

Contains information from [adsb.lol historical data](https://www.adsb.lol/docs/open-data/historical),
available under the [Open Database License (ODbL 1.0)](https://opendatacommons.org/licenses/odbl/1-0/).
The source query and catalogue transformation are in `livermore_archive.py`.
Keep the downloaded inputs and per-airframe diagnostic artifacts in the
ignored `runs/` directory; the map export has no flight catalogue or waveform.

## What the replay measures

1. **Detect from sound alone.** Resample each WAV to 2 kHz and compute a
   2 s Hann STFT at 0.25 s hops, using alternate columns. Search fundamentals
   from 70 through 190.5 Hz in steps of 0.5 Hz. For each fundamental and its
   first three overtones, score local spectral prominence from the lesser of
   temporal and frequency contrast. Use a Viterbi path to favor smooth,
   descending ridges across the full clip. Keep frames with two agreeing
   harmonic peaks (agreement within 0.65 Hz after harmonic division) and
   exclude >3 Hz outliers relative to a robust audio-only ideal flyover fit.
   The two original clips yielded 151 and 164 ridge frames. This detector
   was tuned on these clips; it is neither a blind extraction benchmark nor
   an independently validated detector.
2. **Search on guarded training pitch.** Take 72 / 92 training frames and
   hold out 5 / 4 eligible blocks using the 16 s periodic split and 2 s
   guard in `corroboration.py`. Use only training frequencies in scoring.
   Path selection, contrast, outlier rejection and scored-span endpoints
   use the full extracted clip before that split. The held-out check tests
   the locked aircraft/location/clock/frequency fit on disjoint STFT frames,
   not independence from detector tuning or signal preprocessing.
3. **Fit the retarded-time Doppler model.** The archive transformation
   supplies natural-cubic position/velocity splines on admissible
   subsonic track segments. For emission time `t`, distance `r(t)` to the
   candidate receiver and radial velocity `dr/dt`, compute reception time
   `t + r(t)/343` and ratio `D = 1/(1 + (dr/dt)/343)`. Sample the emission
   curve at <=0.25 s steps and interpolate `D` linearly in arrival time.
   Interpolate at one common candidate clock plus each clip's relative
   sample times and the fixed gap for clip two. For each clip profile a
   constant emission frequency `sum(D * f_observed) / sum(D * D)`. Score
   the equal mean of the two clips' training MSEs; neither clip is weighted
   by its sample count.
4. **Rank coverage-eligible ordered addresses.** Evaluate 121 nodes on a
   1 km ENU lattice and first-start clocks spaced 2 s apart. Spline segments
   must cover both detected ridge spans at the same candidate clock. Skip
   same-address pairs, preserve slot order, and take the best training score
   across all eligible nodes, clocks, shard overlaps and segment pairs for
   each ordered ICAO pair. This counts ADS-B address pairs as airframe
   proxies, not independent flights, track segments, or continuous-time
   possibilities. Audit the denominator separately by intersecting the
   arrival-coverage time intervals on the discrete clock grid; no pitch
   scoring is used in that audit.
5. **Refine and lock.** Run a bounded local least-squares fit of ENU east,
   north, and a relative clock within 90 s of the seed. Refine 24 pairs
   using only each pair's best coarse seed and its two winning track
   segments. Select by training score. This is not exhaustive global
   continuous optimization. Write `fit-lock.json` before consulting the
   phone coordinate. On locked frequencies, point and clock, require at
   least 75% of blocks per clip to have median absolute prediction error
   within three times the local ridge jitter proxy with a 0.0625 Hz floor.
   The leading original hypothesis failed in both clips (2/5 and 2/4).

## Map and other outputs

- `automatic-ridges.json`: detector diagnostics and clip-relative measured
  `[time_s, frequency_hz]` samples.
- `coarse-rankings.json`: ordered pair minima and corresponding coarse
  location, clock and segment indices. `denominator-audit.json` separately
  records the exact pair-set comparison.
- `fit-lock.json` and `locked-predictions.json`: locked fit, input receipts,
  and holdout observations/predictions for the winner and up to three
  next-best refined candidates. The SHA-256 in predictions identifies the lock.
- `map.json`: WGS84 `[longitude, latitude]` boundary and an explicitly
  unverified marker. The 1 km layer profiles all eligible pairs over the
  discrete clock grid (121 finite cells). The 100 m layer (441 cells) and
  25 m layer (841 cells) use only the two locked winner segments; at each
  fine node, search covered clocks within 90 s of the fitted clock and
  locally refine around the best sampled clock. Both layers independently
  reprofile the two source frequencies. These local fine grids are not
  all-pair rankings. Each polygon is a grid node's midpoint footprint, clipped
  to the search bounds. Color shows training loss, not confidence.
