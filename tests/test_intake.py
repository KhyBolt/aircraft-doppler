import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from aircraft_doppler.audio import read_audio
from aircraft_doppler.benchmark import _write_json
from aircraft_doppler.cli import main
from aircraft_doppler.intake import _utc
from aircraft_doppler.physics import Track, forward_curve, geodetic_to_ecef
from aircraft_doppler.search import Event
from aircraft_doppler.wide import BudgetExceeded
from test_result_map import map_data


def tone(frequency):
    return np.rint(12000 * np.sin(2 * np.pi * frequency * np.arange(24000) / 8000)).astype(np.int16)


def peak(signal, rate):
    frequencies = np.fft.rfftfreq(len(signal), 1 / rate)
    return frequencies[np.argmax(abs(np.fft.rfft(signal)))]


def invoke(*args):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with patch("sys.argv", ["aircraft-doppler", *map(str, args)]):
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                status = main()
            except SystemExit as error:
                status = error.code
    return status, stdout.getvalue(), stderr.getvalue()


class AudioIntakeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.mono = self.root / "tone with space.wav"
        self.stereo = self.root / "stereo.wav"
        wavfile.write(self.mono, 8000, tone(120))
        wavfile.write(self.stereo, 8000, np.column_stack((tone(200), tone(260))))

    def ffmpeg(self, *args):
        if not shutil.which("ffmpeg"):
            self.skipTest("FFmpeg is not installed")
        result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *map(str, args)],
                                capture_output=True, text=True)
        if result.returncode and "Unknown encoder" in result.stderr:
            self.skipTest(result.stderr.strip())
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pcm_channel_selection_without_ffmpeg(self):
        with patch("aircraft_doppler.audio.subprocess.run") as decoder:
            rate, signal = read_audio(self.stereo, channel=1)
            decoder.assert_not_called()
        self.assertEqual(rate, 2000)
        self.assertAlmostEqual(peak(signal, rate), 260, delta=1)

    def test_mp3_channel_and_compressed_wav(self):
        mp3 = self.root / "stereo.mp3"
        compressed = self.root / "compressed.wav"
        self.ffmpeg("-i", self.stereo, "-c:a", "libmp3lame", "-b:a", "48k", mp3)
        self.ffmpeg("-i", self.mono, "-c:a", "adpcm_ima_wav", compressed)
        for path, channel, expected in ((mp3, 1, 260), (compressed, 0, 120)):
            with self.subTest(path=path.name):
                rate, signal = read_audio(path, channel=channel)
                self.assertAlmostEqual(peak(signal, rate), expected, delta=1)
                self.assertAlmostEqual(len(signal) / rate, 3, delta=.15)

    def test_video_maps_first_audio_stream_even_if_later_stream_has_more_channels(self):
        video = self.root / "two-tracks.mp4"
        self.ffmpeg("-f", "lavfi", "-i", "color=s=16x16:r=2:d=3", "-i", self.mono,
                    "-i", self.stereo, "-map", "0:v", "-map", "1:a", "-map", "2:a",
                    "-c:v", "mpeg4", "-c:a", "aac", video)
        rate, signal = read_audio(video)
        self.assertAlmostEqual(peak(signal, rate), 120, delta=1)
        with self.assertRaisesRegex(ValueError, "Only channel 0"):
            read_audio(video, channel=1)

    def test_no_audio_video_has_clear_cli_error(self):
        video = self.root / "silent video.mp4"
        self.ffmpeg("-f", "lavfi", "-i", "color=s=16x16:r=2:d=1", "-c:v", "mpeg4", video)
        status, _, stderr = invoke("spectrogram", video, self.root / "unused.npz")
        self.assertEqual(status, 2)
        self.assertIn("No audio stream", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertFalse((self.root / "unused.npz").exists())

    def test_empty_wav_and_missing_ffmpeg_have_clear_errors(self):
        mp3 = self.root / "media.mp3"
        mp3.touch()
        with patch("aircraft_doppler.audio.subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(ValueError, "Install ffmpeg"):
                read_audio(mp3)
        empty = self.root / "empty.wav"
        wavfile.write(empty, 8000, np.array([], dtype=np.int16))
        with self.assertRaisesRegex(ValueError, "no samples"):
            read_audio(empty)


class RecordingIntakeTests(unittest.TestCase):
    def test_utc_requires_timezone_and_normalizes_offset(self):
        with self.assertRaisesRegex(ValueError, "include a timezone"):
            _utc("2024-04-07T16:34:18")
        self.assertEqual(_utc("2024-04-07T18:34:18+02:00"),
                         _utc("2024-04-07T16:34:18Z"))

    def test_silence_with_compatible_tracks_abstains_without_point(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            wavfile.write(path / "silent.wav", 2000, np.zeros(16000, dtype=np.int16))
            tracks = path / "tracks.csv"
            tracks.write_text("flight_id,time_s,latitude,longitude,altitude_ellipsoid_m\n" +
                              "".join(f"sim-a,{1712507658+i},5,{5+.0003*i},1050\n" for i in range(7)))
            out = path / "out"
            status, stdout, stderr = invoke("locate", "--audio", path / "silent.wav",
                                           "--tracks", tracks, "--start-utc", "2024-04-07T16:34:18Z",
                                           "--area-center", "5", "5", "100", "--output", out)
            self.assertEqual((status, stderr), (2, ""))
            result = json.loads(stdout)
            self.assertEqual(result, json.loads((out / "inference.json").read_text()))
            self.assertEqual(result["status"], "abstained")
            self.assertEqual(result["outcome"], "no_supported_ridge")
            self.assertEqual((result["area_width_m"], result["clock_half_width_s"]), (10000, 80))
            self.assertEqual(result["ridges_detected"], 0)
            self.assertIn("0 passed", result["abstention_reasons"][0])
            self.assertFalse(any(key.startswith("estimated_") for key in result))
            self.assertEqual(sorted(item.name for item in out.iterdir()),
                             ["inference.json", "result-map.html"])
            data = map_data(out / "result-map.html")
            self.assertFalse(data["accepted"])
            self.assertIsNone(data["receiver"])
            self.assertIsNone(result["diagnostic_candidate"])
            self.assertIsNone(data["diagnostic_candidate"])
            self.assertIsNone(data["published_reference"])
            self.assertIsNone(data["reference_separation_m"])
            self.assertEqual((data["grid"], data["tracks"]), ([], []))
            self.assertEqual(data["reasons"], result["abstention_reasons"])
            self.assertEqual(len(data["boundary"]), 4)

    def test_diagnostic_candidate_is_retained_without_promoting_a_fix(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            tracks = path / "tracks.csv"
            tracks.write_text("flight_id,time_s,latitude,longitude,altitude_ellipsoid_m\n" +
                              "".join(f"sim-a,{1712507658+i},5,{5+.0003*i},1050\n" for i in range(7)))
            for mode, public_status, candidate_status, outcome in (
                ("abstained", "abstained", "not_accepted", "candidate_found"),
                ("timed_out", "timed_out", "partial_fit_not_accepted", "time_budget_reached"),
                ("supported_in_this_fixture", "supported_under_model", "also_supported", "estimate_supported"),
            ):
                with self.subTest(mode=mode):
                    out = path / mode
                    def recorded_fit(_audio, _tracks, _points, guarded, **kwargs):
                        guarded.mkdir()
                        (guarded / "grid_25m.csv").write_text(
                            "x_m,y_m,z_m,loss_hz2\n25,0,0,0.125\n")
                        fit = {"status": mode, "abstention_reasons": ["gate"] if mode != "supported_in_this_fixture" else [],
                               "best_fit_diagnostic":{"xyz_m": [25,0,0], "loss_hz2": .125}}
                        if mode == "timed_out":
                            (guarded / "inference.json").write_text(json.dumps(fit))
                            raise BudgetExceeded("budget")
                        if mode == "supported_in_this_fixture":
                            fit.update(estimated_xyz_m=[25,0,0], estimated_pair="sim-a|sim-b",
                                       estimated_clock_offset_s=0, evidence_grade="conditional_on_supplied_track_catalogue")
                        return fit
                    with patch("aircraft_doppler.intake.infer", side_effect=recorded_fit):
                        exit_code, stdout, stderr = invoke(
                            "locate", "--audio", path/"unused.wav", "--tracks", tracks,
                            "--start-utc", "2024-04-07T16:34:18Z",
                            "--area-center", "5", "5", "100", "--output", out,
                            "--published-reference", "5", "5")
                    self.assertEqual(stderr, "")
                    result = json.loads(stdout)
                    data = map_data(out/"result-map.html")
                    self.assertEqual(result["status"], public_status)
                    self.assertEqual(result["outcome"], outcome)
                    self.assertEqual(exit_code, 0 if public_status == "supported_under_model" else 2)
                    candidate = result["diagnostic_candidate"]
                    self.assertEqual(candidate["status"], candidate_status)
                    self.assertEqual(candidate["enu_m"], [25,0,0])
                    self.assertAlmostEqual(candidate["lat_lon"][0], 5, places=7)
                    self.assertGreater(candidate["lat_lon"][1], 5)
                    self.assertEqual(data["diagnostic_candidate"]["lat_lon"], candidate["lat_lon"])
                    self.assertEqual(data["grid"][0]["spacing_m"], 25)
                    self.assertEqual(data["grid"][0]["centers_en_m"], [[25,0]])
                    self.assertEqual(data["published_reference"]["lat_lon"], [5,5])
                    self.assertAlmostEqual(data["reference_separation_m"], 25, delta=.2)
                    self.assertNotIn("published_reference", result)
                    if public_status == "supported_under_model":
                        self.assertEqual(result["estimated_lat_lon"], candidate["lat_lon"])
                        self.assertIsNotNone(data["receiver"])
                    else:
                        self.assertNotIn("estimated_lat_lon", result)
                        self.assertIsNone(data["receiver"])

    def test_locate_scores_one_ridge_without_releasing_a_fix(self):
        epoch = 1712507658
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            audio = path / "audio.wav"
            wavfile.write(audio, 8000, np.tile(tone(120), 20))
            tracks = path / "tracks.csv"
            times = np.arange(44.)
            tracks.write_text("flight_id,time_s,latitude,longitude,altitude_ellipsoid_m\n" +
                              "".join(f"pass,{epoch+index},0.008,{(index-21)*.00065},800\n" for index in range(44)))
            track = Track("pass", times, np.column_stack((72.36*(times-21),
                          np.full_like(times, 884.6), np.full_like(times, 800))))
            curve = forward_curve(track, (0, 0, 0))
            audio_times = np.arange(10., 36., .5)
            frequencies = 120*np.interp(audio_times, curve.arrival_s, curve.ratio)
            event = Event((10., 35.5), 24., (115, 125, .2),
                          trace_points=np.column_stack((audio_times, frequencies)).tolist())
            proposal = {"band_hz": [60, 200]}
            with patch("aircraft_doppler.wide.detect_field_events", return_value=(
                    [event], {"selected": [proposal], "supported_proposals": [proposal], "rejected": []})) as detector:
                code, stdout, stderr = invoke(
                    "locate", "--audio", audio, "--tracks", tracks, "--output", path/"out",
                    "--start-utc", "2024-04-07T16:34:18Z", "--area-center", "0", "0", "0")
            detector.assert_called_once()
            self.assertEqual((code, stderr), (2, ""))
            report = json.loads(stdout)
            self.assertEqual(report["outcome"], "candidate_found")
            self.assertEqual(report["ridges_detected"], 1)
            self.assertEqual(report["audio_audit"]["selected"], [proposal])
            self.assertIsNotNone(report["diagnostic_candidate"])
            self.assertFalse(any(key.startswith("estimated_") for key in report))


class TwoClipIntakeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.first = self.root / "first.wav"
        self.second = self.root / "second.wav"
        self.tracks = self.root / "tracks.csv"
        self.epoch = 1712507658

    def locate(self, output, *options):
        return invoke("locate", "--audio", self.first, "--tracks", self.tracks,
                      "--start-utc", "2024-04-07T16:34:18Z", "--area-center", "0", "0", "0",
                      "--output", self.root / output, *options)

    def test_locate_rejects_invalid_clip_options_and_bounds(self):
        cases = [(("--clip-gap-s", 3), "--clip-gap-s requires --second-audio"),
                 (("--second-audio", self.second), "positive, finite --clip-gap-s")]
        cases.extend((("--second-audio", self.second, "--clip-gap-s", gap), "positive, finite --clip-gap-s")
                     for gap in (0, -1, "nan", "inf"))
        for option, above_maximum in (("--area-width-km", 31), ("--time-radius-s", 3601)):
            cases.extend(((option, value), f"{option} must be a whole number") for value in (0, above_maximum))
            cases.append(((option, "1.5"), f"{option}: invalid int value"))
        for options, message in cases:
            with self.subTest(options=options):
                code, stdout, stderr = self.locate("invalid", *options)
                self.assertEqual((code, stdout), (2, ""))
                self.assertIn(message, stderr)
                self.assertNotIn("Traceback", stderr)

    def test_locate_rejects_overlapping_clips_but_allows_adjacent_clips(self):
        sample_rate = 8000
        first_samples = np.append(tone(120), np.int16(0))
        first_duration = len(first_samples) / sample_rate
        wavfile.write(self.first, sample_rate, first_samples)
        wavfile.write(self.second, sample_rate, tone(220))
        self.tracks.write_text("flight_id,time_s,latitude,longitude,altitude_ellipsoid_m\n" +
                               "".join(f"first-pass,{self.epoch + index},0.006,{index*.00065},800\n"
                                       for index in range(4)))
        options = ("--second-audio", self.second, "--area-width-km", 1)
        code, stdout, stderr = self.locate("overlap", *options, "--clip-gap-s", 3)
        self.assertEqual((code, stdout), (2, ""))
        self.assertIn("clips must not overlap", stderr)

        code, stdout, stderr = self.locate("adjacent", *options, "--clip-gap-s", first_duration)
        self.assertEqual((code, stderr), (2, ""))
        report = json.loads(stdout)
        self.assertEqual(report["clip_start_gap_s"], first_duration)
        self.assertEqual(report["clip_durations_s"], [first_duration, 3])
        self.assertEqual(report["recording_duration_s"], first_duration + 3)
        self.assertEqual(report["outcome"], "no_supported_ridge")

    def test_locate_shifts_independent_ridges_and_fits_one_clock_within_custom_bounds(self):
        gap = 91.25
        clock = 3
        trace_times = np.arange(3., 43., .25)
        emission_times = np.arange(43.)
        displacement = (emission_times - 23) * .00065
        rows = ["flight_id,time_s,latitude,longitude,altitude_ellipsoid_m\n"]
        detections = []
        for identifier, audio, duration, offset, lat, lon, frequency in (
                ("first-pass", self.first, 45, 0, np.full_like(displacement, .006), displacement, 120),
                ("second-pass", self.second, 48, gap, displacement, np.full_like(displacement, -.006), 220)):
            wavfile.write(audio, 8000, np.column_stack((np.tile(tone(400), duration // 3),
                                                       np.tile(tone(frequency), duration // 3))))
            local = (geodetic_to_ecef(lat, lon, 800) - geodetic_to_ecef(0, 0, 0))[:, [1, 2, 0]]
            times = emission_times + offset
            rows.extend(f"{identifier},{self.epoch + time},{latitude},{longitude},800\n"
                        for time, latitude, longitude in zip(times, lat, lon))
            curve = forward_curve(Track(identifier, times, local), (0, 0, 0))
            frequencies = frequency * np.interp(trace_times + offset + clock, curve.arrival_s, curve.ratio)
            event = Event((trace_times[0], trace_times[-1]), 25., (frequency - 2, frequency + 2, .25),
                          trace_points=np.column_stack((trace_times, frequencies)).tolist())
            proposal = {"band_hz": [frequency - 2, frequency + 2], "center_audio_s": 25.}
            detections.append(([event], {"selected": [proposal], "supported_proposals": [proposal], "rejected": []}))
        self.tracks.write_text("".join(rows))

        def detect(spec, sound_speed, *, max_events):
            self.assertEqual(max_events, 1)
            self.assertEqual(spec.t[0], 1.)
            dominant = spec.f[np.argmax(spec.power.mean(axis=1))]
            if dominant == 120 and spec.t[-1] == 44.:
                return detections[0]
            self.assertEqual((dominant, spec.t[-1]), (220, 47.))
            return detections[1]

        options = ("--second-audio", self.second, "--channel", 1, "--area-width-km", 1)
        with (patch("aircraft_doppler.wide.detect_field_events", side_effect=detect) as detector,
              patch("aircraft_doppler.wide._write_json", wraps=_write_json) as write_json):
            _, stdout, stderr = self.locate("shared", *options, "--clip-gap-s", gap, "--time-radius-s", 4)
        self.assertEqual(stderr, "")
        self.assertEqual(detector.call_count, 2)
        written = {call.args[0].name: call.args[1] for call in write_json.call_args_list}
        report = json.loads(stdout)
        self.assertEqual(report["ridges_detected"], 2)
        self.assertEqual(report["clip_durations_s"], [45, 48])
        self.assertEqual(report["recording_duration_s"], 93)
        self.assertEqual(report["clip_start_gap_s"], gap)
        self.assertEqual([clip["clip_start_offset_s"] for clip in report["audio_audit"]["clips"]], [0, gap])
        self.assertEqual([proposal["clip"] for proposal in report["audio_audit"]["selected"]], [1, 2])
        self.assertIn("second_audio", report["input_sha256"])
        for serialized, (events, _), offset in zip(written["detected_events.json"], detections, (0, gap)):
            event = events[0]
            np.testing.assert_allclose(serialized["window_s"], np.asarray(event.window_s) + offset)
            self.assertEqual(serialized["cpa_audio_s"], event.cpa_audio_s + offset)
            np.testing.assert_allclose(serialized["trace_points"], np.asarray(event.trace_points) + [offset, 0])
        best = written["inference.json"]["best_fit_diagnostic"]
        self.assertEqual(best["pair"], "first-pass|second-pass")
        self.assertAlmostEqual(best["clock_offset_s"], clock, delta=.125)
        self.assertLess(best["loss_hz2"], 1e-6)
        self.assertEqual((report["area_width_m"], report["clock_half_width_s"]), (1000, 4))
        grid = json.loads((self.root / "shared" / "evaluated-grid.json").read_text())
        self.assertEqual(grid["search_half_width_m"], 500)
        self.assertEqual(grid["stages"][0]["attempted_positions"], 9)
        for stage in grid["stages"]:
            self.assertTrue(all(abs(east) <= 500 and abs(north) <= 500 for east, north, *_ in stage["points"]))

        for output, clip_gap, radius in (("tight-clock", gap, 1), ("wrong-gap", gap + 8, 12)):
            with self.subTest(output=output), patch("aircraft_doppler.wide.detect_field_events", side_effect=detect):
                code, stdout, stderr = self.locate(output, *options, "--clip-gap-s", clip_gap, "--time-radius-s", radius)
                self.assertEqual((code, stderr), (2, ""))
                report = json.loads(stdout)
                self.assertEqual(report["ridges_detected"], 2)
                self.assertEqual(report["clock_half_width_s"], radius)
                self.assertIsNone(report["diagnostic_candidate"])
                self.assertIn("common clock offset", report["abstention_reasons"][0])


if __name__ == "__main__":
    unittest.main()
