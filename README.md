# Aircraft Audio Geolocation

Aircraft Doppler signatures can constrain where a recording was made.
This project detects descending tone ridges, compares them with flight
tracks, and can abstain when the available evidence is too weak.

## Install

Use Python 3.12 with the versions pinned for validation. Run the commands
from the repository root.

~~~sh
git clone https://github.com/KhyBolt/aircraft-doppler.git
cd aircraft-doppler
~~~

~~~sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-verified.txt
~~~

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.
PCM WAV works directly. Install FFmpeg on PATH for MP3, compressed WAV,
and video such as MP4.

## Analyze a local recording

The experimental `locate` command looks for two flyovers in a continuous
recording or in two separate clips you made or have permission to analyze.
The recorder must stay in the same place. For two clips, use one clear
flyover per clip and put the earlier clip first. The clips must not overlap.

~~~text
python -m aircraft_doppler locate --audio first.wav --second-audio second.wav --clip-gap-s 180.25 --tracks regional.csv --start-utc 2026-05-17T22:03:00Z --time-radius-s 120 --area-center 37.69 -121.82 90 --area-width-km 8 --output runs/local
~~~

Replace the example inputs with your own:

- `--clip-gap-s` is the elapsed time from the first audio sample of the first clip to the first audio sample of the second clip. It is not the silence between files. The search shifts both starts together; it does not fit the gap.
- `--start-utc` is the approximate start of the first clip, with a time zone. `--time-radius-s 120` searches two minutes on either side (default ±80 seconds; 1–3,600 seconds allowed). For video, use the start of the audio, not the picture.
- `--area-center` takes latitude, longitude, and the assumed recorder height above the WGS84 ellipsoid, in metres. `--area-width-km` is the full side length of the square (default 10 km; 1–30 whole km allowed). The search uses a flat local plane at that height, not terrain elevation.
- `--tracks` is a local ADS-B track CSV. `locate` does not download tracks. Supply regional tracks for both flyovers, covering both audio windows across your time uncertainty and allowing for sound travel time.

For a continuous recording, omit `--second-audio` and `--clip-gap-s`:

~~~text
python -m aircraft_doppler locate --audio recording.mp4 --tracks regional.csv --start-utc 2026-05-17T22:03:00Z --area-center 37.69 -121.82 90 --output runs/continuous
~~~

The decoder uses the first audio stream and channel 0 by default. Set
`--channel` to use a different zero-based channel in every input file.
Start with a small area and time window; wider searches may need a larger
`--max-seconds` budget. The [full-day Livermore replay](benchmarks/livermore-replay.md)
uses a separate script and detector tuned on those clips.

Prepare a local CSV with
`flight_id,time_s,latitude,longitude,altitude_ellipsoid_m`. Use Unix UTC
seconds and WGS84 ellipsoid metres. Split tracks at gaps over 15 seconds,
give each segment a unique `flight_id`, and use `aircraft_id` to link
segments of the same airframe. The optional `engine` column accepts
`propeller`, `jet`, or `unknown`. The command runs offline.

The output directory must be new or empty. Exit 0 reports an estimate that
passed the model's checks. Exit 2 means no supported estimate was returned
(or the input was invalid). Read `outcome` in the JSON to distinguish a
candidate found, no supported ridge, no candidate, and time budget reached.
`locate` looks for ridges across multiple frequency bands and near clip edges.
A single ridge can produce a diagnostic candidate but cannot verify a location.
The `audio_audit` records which proposals were selected or rejected and why.
The held-out check tests a locked fit on separate time blocks of the selected
ridges. It checks the fit conditional on those ridges, not independent ridge discovery.
A completed run writes `runs/local/inference.json` and
`runs/local/result-map.html`. Open the HTML in a browser to pan, zoom, and
inspect the search boundary, sampled fit scores, and selected flight paths.
An accepted run marks the receiver as supported under the model. On abstention
or a timeout, a finite lowest-residual point is shown separately as an
**estimated candidate, not accepted**, never as a receiver fix. The JSON
stores that point and its WGS84 coordinates in `diagnostic_candidate`,
or `null` if nothing was scored; `estimated_*` fields appear only after
all acceptance gates pass. On a timeout the candidate only describes
completed scoring work. A leading track hypothesis is unverified.
Add `--published-reference LAT LON` only for an intentionally published
benchmark location you want marked on the map. It is not used to fit audio
or accept a point. The map reports separation only when both points exist.

The survey and refinement layers show minimum training ridge MSE (Hz²) at
the positions the solver actually scored. Toggle each stage separately or
use **Fit finest grid** and zoom further to inspect individual samples.
With one refinement layer visible, colors adapt to the visible scores
and the cell readout keeps the original scale in Hz².
When finite scores exist, `evaluated-grid.json` saves the exact local
east, north and up coordinates, loss, stage spacing, first evaluation
spacing, and clipped scored-cell footprint for each stage with finite scores.
A reused point keeps its original score. No scores are interpolated, and
the footprint describes coverage, not location probability or uncertainty.
The HTML works offline. You can enable USGS topographic tiles in the browser
when they are available. The offline grid and result layers remain usable
without them. The HTML contains location and flight geometry, so share it only
with people who can receive those details.

### Related implementations

- [DopplerTrack](https://github.com/mjshiggins/dopplertrack) fits straight-pass templates and can score whether a ridge stands out from nearby alternatives. Its experiments also show how a bad search prior or an overly strict early filter can discard the right track.
- [digitalis-p/doppler-locate](https://github.com/digitalis-p/doppler-locate) is a Python alternative that proposes curves from log-frequency templates and groups harmonics before refitting.
- [dopplerprint](https://github.com/Hexay/dopplerprint) detects harmonic families with a tracker that can restart rather than forcing one continuous path through a crop. This is the strongest candidate for a side-by-side detector test.
- [AircraftWitness](https://github.com/nyu-kr/aircraftwitness) builds on dopplerprint and uses shifted-time controls. It is an evaluation reference, not an independent detector baseline.
- [Cobez02/doppler-locator](https://github.com/Cobez02/doppler-locator/tree/438c1b77699a6f4f8c49d0d1227dc6224c8fea61) searches for harmonic families and isolated tones with a tracker that can pause through fades. It distinguishes tentative points and one-pass lines from accepted locations.
- [Cinna Davis's DOPPLERGEIST](https://github.com/1337isnot1337/DOPPLERGEIST/tree/a033f83574a450cb1837bf7c5acfa8b6550634cc) tests multiresolution curve proposals, counts related harmonics as one group, and reports ranked points even when the result is inconclusive. The reviewed revision uses the PolyForm Noncommercial license.

Compare detectors on the same frozen WAVs, time and area priors, and track
catalogue, with whole sites or dates held out and audibility checked by a
reviewer. Measure ridge coverage, false alerts, track rank, false accepts and
runtime separately. Supplied-track, hand-traced and archive-search results are
not directly comparable. The solver does not include code from the compared implementations.

`--max-seconds` sets the cooperative search budget. Decoding and input
preparation can also take time. After a timeout only positions that finished scoring are exported, including
positions within an interrupted refinement stage.

Intake was verified with synthetic PCM WAV, MP3, and MP4 with AAC. The
same synthetic flight pair and 14.2 m error were measured for each format.
The video-without-audio, single-flyover, missing-coverage, shifted-clock,
and wrong-geometry controls produced no estimate.

## Replay the synthetic benchmark

The bundled run searches eight simulated tracks with two flyovers about
15 minutes apart in a 27-minute WAV. It uses the same 10 × 10 km search
area and ±80 s recording clock prior.

~~~sh
python demo_replay.py --output runs/exact-demo
python plot_demo.py --result runs/exact-demo --output runs/exact-demo.png
~~~

![Synthetic search grid and two Doppler signatures](evidence/demo.png)

| Quantity | Guarded v0.4 replay |
| --- | --- |
| Estimated position | (4,500, −4,400, 0) m in a local synthetic frame |
| Virtual track pair | sim-17 and sim-41 |
| Inferred recording offset | −66.875 s |
| Audio-derived event separation | 903.874 s |
| Distance to synthetic reference | 14.2 m |

Inference reads the [WAV](example/synthetic.wav),
[tracks](example/tracks.csv), and [candidate grid](example/candidates.csv).
The [synthetic reference](example/truth.json) is used afterward to measure
error. See the [saved result](evidence/summary.json) and
[measured traces](evidence/detected_events.json).

The detector uses the full WAV to pick ridges. The guarded fit shares one
clock offset across both passes and checks its locked predictions on
withheld ridge observations. The plot shows sampled search scores.

~~~sh
python -m aircraft_doppler.hardened --case fresh-southeast --output runs/paired-example
python -m aircraft_doppler.hardened --output runs/paired-suite
python -m unittest discover -s tests -v
~~~

Both profiles accepted 3 of 4 unseen synthetic positives with 8.1–14.2 m
error. False accepts fell from 4 of 8 to 1 of 8 across the seven primary
controls and the exact-equivalence case. In that last case, a translated
catalogue created the same acoustic pattern and still produced a false
accept. The [scorecard](evidence/scorecard.csv) records all outcomes.

## Field recordings

The [Eclipse Soundscapes 505 benchmark](benchmarks/eclipse-505-protocol.md)
documents an executable replay on licensed audio and historical tracks.
All completed arms abstained.

The earlier [Livermore adsb.lol search](https://khybolt.github.io/doppler/assets/livermore/replay-method.md)
used two public research WAVs, their exact relative start gap, a date, and
a 10 × 10 km prior. The matching pair ranked first of 17,101. The leading
candidate was 321 m from the study-reported phone position, but the verifier
abstained after the second pass failed its held-out checks.

The [Livermore case study](https://khybolt.github.io/doppler/#livermore)
shows a newer replay that extracts pitch from the original recordings and
searches the full day without supplied flight matches. The matching pair
ranked first of 12,215 covered ordered pairs. The best fit was about 405 m
from the study's phone position and did not pass verification. The separate
assisted replay in the input guide is not part of either full-day ranking.
See the [automatic replay instructions](benchmarks/livermore-replay.md)
to reproduce the newer result and its controls.

The [Livermore input guide](benchmarks/livermore-inputs.md) verifies the
original WAVs and archive receipts, not the 321 m result. Recomputing that
search requires the unavailable solver and clock-screen revisions, the exact
adsb.lol-only invocation, and the report-only score hook.

[Source and data credits](RIGHTS.md)
