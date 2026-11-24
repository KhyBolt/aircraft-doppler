# Aircraft Audio Geolocation

Aircraft Doppler estimates where you recorded two planes flying overhead. It compares the drop in pitch as each plane passes with what you’d expect from nearby flight tracks.

## Install

From the repository root, run `python -m pip install -r requirements-verified.txt` with Python 3.12. PCM WAV files work directly; MP3 and MP4 need [FFmpeg](https://ffmpeg.org/download.html).

## Analyze a local recording

Use two non-overlapping clips you have permission to analyze, recorded from the same spot. Put the earlier flyover first. You’ll also need a CSV of nearby flight tracks; the tool doesn’t download them for you.

```sh
python -m aircraft_doppler locate --audio first.wav --second-audio second.wav --clip-gap-s 180 --tracks regional.csv --start-utc 2026-06-01T14:00:00Z --time-radius-s 120 --area-center 40.00 -105.25 1600 --area-width-km 8 --output runs/local
```

- **Timing:** Here the clips start 180 seconds apart. The search covers two minutes either side of the first clip’s approximate UTC start.
- **Area:** Here the square is 8 km wide. Give its center as latitude, longitude and recorder height in metres above the WGS84 ellipsoid.
- **Result:** Open `runs/local/result-map.html` in a browser. If there’s a match, the map marks the best fit; `inference.json` says whether the checks support it. Use a new or empty output folder each time.

<details>
<summary>Track CSV format</summary>

Use `flight_id,time_s,latitude,longitude,altitude_ellipsoid_m`, with time in Unix UTC seconds and height in WGS84 ellipsoid metres. Include tracks covering both clips across the time window, allowing for sound travel time. Split tracks at gaps over 15 seconds; give each segment a new `flight_id` and keep the same `aircraft_id` for segments of the same plane.

</details>

## Field recordings

In [Livermore](https://khybolt.github.io/doppler/#livermore), the tool searched a day of flight tracks using pitch from two phone recordings. The right flights ranked first out of 12,215 pairs with track coverage. The best fit was about 405 m from the study’s phone. The [full-day replay](benchmarks/livermore-replay.md) gives the inputs, checks and results.

[Source credits and data rights](RIGHTS.md)

<a name="replay-the-synthetic-benchmark"></a>

<details>
<summary>Developer references</summary>

[Bundled test replay](demo_replay.py) · [Generated inputs](example/) · [Saved result](evidence/summary.json) · [Control scorecard](evidence/scorecard.csv)

[Eclipse Soundscapes field protocol](benchmarks/eclipse-505-protocol.md) · [Archived Livermore input receipts](benchmarks/livermore-inputs.md)

### Related implementations

[DopplerTrack](https://github.com/mjshiggins/dopplertrack) · [doppler-locate](https://github.com/digitalis-p/doppler-locate) · [dopplerprint](https://github.com/Hexay/dopplerprint) · [AircraftWitness](https://github.com/nyu-kr/aircraftwitness) · [doppler-locator](https://github.com/Cobez02/doppler-locator/tree/438c1b77699a6f4f8c49d0d1227dc6224c8fea61) · [DOPPLERGEIST](https://github.com/1337isnot1337/DOPPLERGEIST/tree/a033f83574a450cb1837bf7c5acfa8b6550634cc)

</details>
