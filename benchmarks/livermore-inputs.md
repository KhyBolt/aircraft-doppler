# Livermore diagnostic inputs

This guide verifies the WAV and archive inputs for the earlier full-day
adsb.lol-only Livermore search. Its top candidate was 320.619 m (about 321 m)
from the study-reported phone position, and the matching ordered pair ranked
first of 17,101. The verifier abstained with zero acoustic survivors.

The [Livermore case study](https://khybolt.github.io/doppler/#livermore)
shows the newer automatic replay (1 of 12,215 covered ordered pairs, about
405 m from the study's phone position, no verified fix). To rerun that
search, use the [automatic replay instructions](livermore-replay.md).
A separate assisted diagnostic used the
study's supplied hand-guided pitch traces and matching flight tracks, with a fixed
nominal UTC start for each clip. Its top candidate was about 144 m from the
same reference. This replay also abstained. The
[assisted map data](https://khybolt.github.io/doppler/assets/livermore/livermore-assisted-real-audio-map-v1.json)
pins solver revision `2c4b48ab7fbc7a1cd94a3f2c3ea7ef8bee848633`. These
distances are not verified accuracy measurements; the phone position has no
independent survey bound.

## Verify the inputs

From the repository root, install `requirements-verified.txt` with Python
3.12. Run:

~~~sh
OPENBLAS_NUM_THREADS=1 timeout 180s python3.12 -B -m benchmarks.livermore_inputs --check-archive
~~~

The command acquires exact WAV members using bounded byte ranges and verifies
both days of the indexed flight input. It reports source checksums and row
counts. It performs no location search. Use `--verify-only` to check WAVs
already in `runs/livermore-input` offline.

Praveen Pathak, *Aeroplane data full analysis and raw files*, v2,
[Zenodo 24075029](https://doi.org/10.5281/zenodo.24075029), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The members are lossless copies from `raw_flight_data.zip`, unchanged.

| WAV member | SHA-256 |
| --- | --- |
| `N9393T_R02_2026-05-17T22-03-02.336Z.wav` | `7834bad6d620df2143b396325fd861d6581d5ad25dd9bcfa997eb953d13a65f5` |
| `N86LP_R01_2026-05-17T22-20-09.578Z.wav` | `a19c78d9d147a6e8b4b2c495bd344787363875ea411e3af21b59490517832495` |

Both WAVs are 48 kHz mono. The adsb.lol search used a date and the exact
1,027.242 s gap between first samples from the published timestamps, along
with the 10 × 10 km airport-centered area. Absolute clip timestamps,
publisher-traced ridges, and known aircraft identities were withheld from the
ranker. The study reference was used for diagnostic distance after locking.

Contains information from [adsb.lol historical data](https://www.adsb.lol/docs/open-data/historical),
available under the [Open Database License (ODbL 1.0)](https://opendatacommons.org/licenses/odbl/1-0/).
The [public index](https://github.com/ClickHouse/adsb.exposed#data-source)
serves several providers. The query requires `data_source='adsb.lol'` and
checks the returned provider tag before accepting any rows. Only counts and
canonical hashes are emitted. Hourly overlap is included in the counts.

| UTC day | Shards | Source rows | Canonical manifest SHA-256 |
| --- | ---: | ---: | --- |
| 2026-05-17 | 24 | 177,352 | `d29fb193d0ae54f4f74da4c8186bb11e861d32e4d62332b9abc20a7495d0af47` |
| 2026-05-18 | 24 | 124,713 | `bbdc5046a0456e62f95c3f1bbd3bff52ab31651d7f08ca4facc4050f660c1c5e` |

The staged archive helper uses the same query and transformations as the
[portfolio recipe](https://khybolt.github.io/doppler/assets/livermore/archive-query.py),
whose SHA-256 is `82ecd7388ae1bef4286c6fbb2459da67d6e495d74c2e333e279f9a926af73b93`.
The two changed call sites invoke the existing direct imports `urlencode`
and `urlopen`. The original `--check` otherwise fails before its first request.

## Earlier adsb.lol search and controls

The adsb.lol search receipt documents a 1 km ENU search lattice, 2 s coarse
clock stride, 30 s clock buckets, 2 km retention tiles, 24 refinement seeds,
and 343 m/s sound speed. It specifies automatic harmonic extraction and a
locked held-out split. See the [adsb.lol replay method](https://khybolt.github.io/doppler/assets/livermore/replay-method.md)
and its [machine-readable receipt](https://khybolt.github.io/doppler/assets/livermore/overlay-metadata.json).

| Arm | Published pair rank | Diagnostic distance | Held-out pass blocks | Acoustic survivors |
| --- | ---: | ---: | --- | ---: |
| May 17 original | 1 of 17,101 | 320.619 m | 3/3 and 1/4 | 0 |
| May 17 swapped | 1,230 of 17,170 | 5,701.072 m | 1/4 and 2/3 | 0 |
| May 18 wrong day | Absent from 9,252 | 3,992.016 m | 2/3 and 1/4 | 0 |

The separate 268 m DopplerTrack comparator used the supplied flight tracks
and timing on these same WAVs. Attribute it to
[Michael Higgins's DopplerTrack](https://github.com/mjshiggins/dopplertrack).
The comparator is outside this input verifier.

## Reproduction limits and archived map artifacts

Reproducing the 321 m search requires solver
`e963ab83dccd5a205f3b16558bd90d4307bc38b6`,
clock screen `5c361a53653d47f203ec69521542f3cf6955e63f`,
and the exact adsb.lol-only invocation and report-only score hook. Both
commits are missing from the public checkout. A saved private harness exists
for the earlier UTC-day screen and imports both revisions. The exact
invocation used for the single-provider rerun and its report hook remain
unverified. The input tools only verify the WAVs and archive manifests.

The earlier search's [coarse score raster](https://khybolt.github.io/doppler/assets/livermore/livermore-adsblol-coarse-score.png)
is a 768 × 768 EPSG:3857 image. Its SHA-256 is
`32598daaaf98fa787898821b2c70348c52c4228176a32bac449b329151ffa9ab`.
Use the receipt's `image_corners_wgs84_lon_lat` in northwest, northeast,
southeast, southwest order. The [WGS84 search boundary](https://khybolt.github.io/doppler/assets/livermore/search-boundary.json)
is a closed polygon with 41 listed positions. The 121 observed nodes carry
the minimum equal-weight training frequency MSE in Hz². The PNG maps the
nearest node to one of six documented color bins and has transparent pixels
outside the boundary. The archived adsb.lol assets do not include exact
per-node values or the refined candidate's coordinates.
