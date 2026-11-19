# Eclipse Soundscapes 505 replay protocol

This protocol fixes selection before opening any audio samples or solving. It is a known-answer replay. The public station coordinates and the original study's results were available when the protocol was written.

- Start with registered five-minute window `Fe45fe4783028d3a6` on 2024-04-07 at 16:34:18 UTC. The source member is `ESID_505_2024_04_07/20240407_163418.WAV`, at offset zero. A 30-minute window begins at the same point. A second disjoint comparison covers the following 15 minutes regardless of the first results. The second registered packet `Fe6af787cb07bac45` on April 8 is a separate comparison if the compressed-prefix transfer is practical. [Eclipse Soundscapes audio archive](https://doi.org/10.5281/zenodo.21808323)
- Use a 10 × 10 km search square centered at 39.73° N, 84.20° W, the rounded vicinity of Dayton's Carillon Historical Park. This prior differs from the registered 20 km squares and was chosen with access to the station metadata. The solver receives the box and start-time prior alongside its audio and tracks.
- Use adsb.lol prod-0 release data for the relevant UTC day. Crop by time within 25 km east and north of the origin before preparing any clips. Keep eligible unknown aircraft independent of the study's pass labels and scores. Use reported GNSS heights, split gaps, and audit unusable traces.
- Shift the April 7 five-minute audio's claimed time by +2 hours. For the wrong-audio control, swap in the first five minutes of the fixed adjacent comparison, keeping the primary track and clock inputs. Apply the same scoring and gates to both controls and preserve inconclusive and no-track outcomes.
- Report runtime, the candidate count, actual flyover count used, score or abstention, distance of any diagnostic and accepted point to the independent published station, and the known-answer status. A diagnostic point is never a supported fix. Evaluate independently of inference. Record whether the search reached its time bound.

## Recorded detector change

The first five-minute replay of the interior-only detector found zero usable ridges. Its spectrogram shows a descending tone already present at the recording boundary, while the only peaks proposed by the original detector were later. The follow-up adds fixed start/end windows of 24, 40 and 64 seconds in each predeclared band. The same support, downward-change, and interior-CPA requirements still apply to each measured window. This change uses the registered audio as development data. Run the same follow-up profile on the preselected adjacent window and both controls before assessing the distance of any follow-up candidate to the station.

## Code review

The v0.4 wide runner was built around the bundled synthetic lattice, two 65–195 Hz tone ridges, and a ±80 s search for the shared clock. Its emission-time propagation, analytic source-frequency fit, independent aircraft requirement, complete track coverage, and guarded verification are appropriate for a bounded replay. The field runner reuses these parts. It allows a single ridge to generate a diagnostic fit under the study's ±300 s clock allowance and preserves the two-pass requirement for a supported result.

[AircraftWitness](https://arxiv.org/html/2611.08791v1#A1) uses timed single-pass matching alongside a date-only two-pass path. [DopplerTrack](https://github.com/mjshiggins/dopplertrack/blob/main/docs/experiments.md) documents broader tonal evidence, additional flyovers, and failure modes from interrupted tracks. Both informed the field-profile changes. This replay retains a fixed ridge model, parabolic sub-bin measurements, and a single still-air sound speed. It has no RPM-drift, weather, engine-type, or terrain-grid model. Candidate ground is the local tangent plane at 190 m ellipsoid height, based on roughly 220 m topographic elevation and a NOAA GEOID18 undulation of −33.217 m at the prior center. Absolute track heights use readsb geometric fields. Data gaps over 15 s, detected new flight legs, and segments with non-subsonic spline velocities are discarded or split before searching. The published position's decimal precision gives no independent estimate of deployment uncertainty.

## Exact commands

Run from the repository root with Python 3.12. Use empty output directories for each inference. All inference and scoring commands work offline once the inputs are staged. Raw audio and trajectories stay in the ignored `runs/` directory.

```sh
python -m pip install -r requirements-verified.txt
python -m benchmarks.range_wav https://zenodo.org/api/records/21808323/files/ESID_505_2024_04_07.zip/content ESID_505_2024_04_07/20240407_163418.WAV --duration 2700 --max-mib 256 --output runs/field-input/apr07-first45m.wav
python -m benchmarks.crop_adsb https://github.com/adsblol/globe_history_2024/releases/download/v2024.04.07-planes-readsb-prod-0/v2024.04.07-planes-readsb-prod-0.tar --start 2024-04-07T16:27:00Z --stop 2024-04-07T18:46:00Z --center 39.73 -84.20 --max-mib 1850 --max-seconds 360 --output runs/field-input/apr07-adsb.csv
python -m benchmarks.eclipse_replay infer --audio runs/field-input/apr07-first45m.wav --adsb runs/field-input/apr07-adsb.csv --arm registered --detector interior-only --budget-s 120 --output runs/eclipse-505-interior/registered
```

For the field runs, replace `<arm>` with `registered`, `extended`, `adjacent`, `wrong-time`, and `wrong-audio`. Run inference for the arms first, then evaluate with the coordinates from the independent Zenodo record.

```sh
python -m benchmarks.eclipse_replay infer --audio runs/field-input/apr07-first45m.wav --adsb runs/field-input/apr07-adsb.csv --arm <arm> --detector edge-aware --budget-s 120 --output runs/eclipse-505-edge/<arm>
python -m benchmarks.eclipse_replay evaluate --reference 39.7125 -84.152222 --output runs/eclipse-505-edge/<arm>
```

For the two additional 30-minute runs, use `--budget-s 180` with a new output ending in `extended-180s`, and `--final-spacing-m 100 --budget-s 120` with a new output ending in `extended-100m`. The 100 m run is diagnostic because the acceptance gate requires a 25 m search.

Download the map asset and render the registered diagnostic. On Windows PowerShell use `curl.exe`.

```sh
curl -fLsS 'https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/export?bbox=-84.28,39.67,-84.12,39.79&bboxSR=4326&imageSR=4326&size=1200,900&format=png32&f=image' -o runs/field-input/usgs-topo.png
python -m benchmarks.plot_eclipse --run runs/eclipse-505-edge/registered --basemap runs/field-input/usgs-topo.png --output runs/eclipse-505-edge/registered-map.png
python -m unittest discover -s tests -v
python demo_replay.py --output runs/synthetic-after-field
```

The selected WAV member is DEFLATE-compressed. The range extractor fetched 205,521,451 bytes from the 6,600,078,968-byte ZIP. It records the publisher's ZIP-directory CRC and the SHA-256 of the saved excerpt. Full-member CRC verification would require reading the full 12.4-hour WAV. The ADS-B reader streamed 1,900,820,480 bytes and retained 12,034 geometric-height points from 163 aircraft traces, skipping 76 in-area samples lacking geometric height. The April 8 registered WAV begins 19,204 seconds into a separately compressed member. That offset requires inflating a large prefix to extract the five-minute window. The 15-minute disjoint comparison uses the staged April 7 audio instead.

## Observed results

The registered clip is a known-answer development replay. Several arms reuse the same station and some audio, so these outcomes do not estimate a success rate. Errors are great-circle distances to the coordinates published by Eclipse Soundscapes. Diagnostic error is the distance of a lowest-residual candidate retained for inspection. Only an accepted point within 1 km can count as a hit.

| Replay | Selected ridges | Outcome | Diagnostic error | Solver wall time |
| --- | ---: | --- | ---: | ---: |
| Registered five minutes, interior-only | 0 | Abstained | — | 0.67 s |
| Registered five minutes, edge-aware | 1 | Abstained | 395 m | 52.56 s |
| Extended 30 minutes, 25 m | 2 | Timed out | — | 120.05 s |
| Extended 30 minutes, 25 m retry | 2 | Timed out | — | 180.03 s |
| Extended 30 minutes, 100 m | 2 | Abstained | 1,036 m | 77.61 s |
| Adjacent 15 minutes | 0 | Abstained | — | 1.92 s |
| Wrong time, +2 hours | 1 | Abstained | 5,253 m | 28.17 s |
| Wrong audio, matched-duration swap | 0 | Abstained | — | 0.76 s |

The five-minute fit ranked 22 eligible track hypotheses at 4,962 sampled positions. Its winner had 0.237 Hz training RMSE. Only half of the held-out blocks passed, and 157 near-best sampled cells spanned 700 m east–west and 2,375 m north–south. The 30-minute, 100 m run passed half of the first ridge's held-out blocks and none of the second ridge's. No run produced an accepted position, including either control. The station coordinates supply position ground truth. The identity of the audible source is not independently labelled here.

AircraftWitness reported a 181 m dopplerprint hit on the registered five-minute case under its own 20 km prior and runner with a 600 s cap. The replay above uses our own detector, scorer, narrower prior, and different acceptance gates. [AircraftWitness published rows](https://github.com/nyu-kr/aircraftwitness/blob/main/results/fresh_rows.csv), [registered runner](https://github.com/nyu-kr/aircraftwitness/blob/main/studies/runner/task_fresh.py)

The [machine-readable run evidence](../evidence/eclipse-505/summary.json) includes input fingerprints and reasons. The [Coarse grid CSV](../evidence/eclipse-505/registered-grid-500m.csv) uses local east, north, and up coordinates in metres and mean squared ridge error in Hz². The [map and spectrogram](https://khybolt.github.io/doppler/field/) label these as fit residuals.
