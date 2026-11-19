# Aircraft Doppler: a reproducible synthetic demo

Can two Doppler signatures in one long recording constrain the position of a
stationary virtual microphone? This example starts from a **27-minute
synthetic WAV**, a **10 × 10 km** search area, and an unknown recording clock
offset of **±80 seconds**. It identifies two tone ridges about **15 minutes**
apart and ranks matches against eight simulated tracks, including decoys.

![Wide synthetic run: sampled grid and two Doppler signatures](evidence/demo.png)

## Replay the exact files

Use Python 3.12 with the pinned dependencies used for validation.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-verified.txt
python demo_replay.py --output runs/exact-demo
```

On Windows, activate the environment with .venv\Scripts\Activate.ps1. The
replay runs offline after dependencies are installed. It reads only
[synthetic.wav](example/synthetic.wav), [tracks.csv](example/tracks.csv), and
[candidates.csv](example/candidates.csv); it does not read the answer.

Expected result:

| Quantity | Guarded v0.4 replay |
| --- | --- |
| Estimated position | (4,500, −4,400, 0) m in a local synthetic frame |
| Virtual track pair | sim-17 and sim-41 |
| Inferred recording offset | −66.875 s |
| Audio-derived event separation | 903.874 s |
| Distance to known synthetic truth | 14.2 m |

To regenerate the image after replay, run: 

`python plot_demo.py --result runs/exact-demo --output runs/exact-demo.png`.

Inspect the [recorded summary](evidence/summary.json),
[measured traces](evidence/detected_events.json), and
[expected ground truth](example/truth.json). The runner estimates position
and timing; the error in the table is calculated afterward. The WAV is
downsampled mono PCM for the low-frequency benchmark. Use the Raw/Download
button if your browser does not preview it.

Regenerate the same fixture and compare both profiles on it, or run the full
paired suite:

```sh
python -m aircraft_doppler.hardened --case fresh-southeast --output runs/paired-example
python -m aircraft_doppler.hardened --output runs/paired-suite
python -m unittest discover -s tests -v
```

## What the comparison showed

Both profiles accepted the same **3 of 4** previously unseen synthetic
positive cases (8.1–14.2 m error). Across the primary seven controls and the
separately constructed exact-equivalence challenge, false accepts fell from
**4 of 8** to **1 of 8**. The v0.4 checks stopped a 4.38 km wrong-track match
and a 1.09 km near-decoy match. The exact-equivalence challenge still failed:
if the true tracks are missing and a translated catalogue generates the very
same audio pattern, the sound cannot tell the two locations apart.

The profile requires the tracks to span each measured event, fits one
shared recording offset across the two passages, and checks the locked fit
on temporally withheld ridge observations. The detector still looks at
the full WAV to select the ridges. The spatial map displays tested cells,
not probabilities. This is a **synthetic demonstration**, not a field
accuracy claim or direct benchmark against other projects.

See the [complete scorecard](evidence/scorecard.csv) and
[provenance](RIGHTS.md).

## Real-recording replay

The [Eclipse Soundscapes 505 benchmark](benchmarks/eclipse-505-protocol.md)
replays the solver on licensed field audio and historical ADS-B tracks. It
includes the registered clip, a longer window, a disjoint same-station
comparison, two negative controls, and a separately scored station reference.
