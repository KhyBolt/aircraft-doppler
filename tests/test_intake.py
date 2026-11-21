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
from aircraft_doppler.cli import main
from aircraft_doppler.intake import _utc
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
            self.assertEqual(result["ridges_detected"], 0)
            self.assertIn("0 passed", result["abstention_reasons"][0])
            self.assertFalse(any(key.startswith("estimated_") for key in result))
            self.assertEqual(sorted(item.name for item in out.iterdir()),
                             ["inference.json", "result-map.html"])
            data = map_data(out / "result-map.html")
            self.assertFalse(data["accepted"])
            self.assertIsNone(data["receiver"])
            self.assertEqual((data["grid"], data["tracks"]), ([], []))
            self.assertEqual(data["reasons"], result["abstention_reasons"])
            self.assertEqual(len(data["boundary"]), 4)


if __name__ == "__main__":
    unittest.main()
