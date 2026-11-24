import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io.wavfile import write

from aircraft_doppler.corroboration import split_trace
from aircraft_doppler.measurement import ideal_received_frequency
from benchmarks.livermore_inputs import CLIPS, verify
from benchmarks.livermore_replay import harmonic_ridge, profile, score_shard


class LivermoreReplayTests(unittest.TestCase):
    def test_harmonic_consensus_tracks_synthetic_flyover(self):
        rate = 2000
        times = np.arange(120 * rate) / rate
        target = ideal_received_frequency(times, 72, 2200, 123, 60)
        phase = 2 * np.pi * np.cumsum(target) / rate
        envelope = np.sin(np.pi * times / 120) ** 2
        noise = np.random.default_rng(2026).normal(0, 0.03, len(times))
        pcm = noise + envelope * (0.15 * np.sin(phase) + 0.20 * np.sin(2 * phase)
                                  + 0.10 * np.sin(3 * phase) + 0.05 * np.sin(4 * phase))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flyover.wav"
            write(path, rate, (pcm * 32767).astype(np.int16))
            trace, _ = harmonic_ridge(path)
        self.assertGreater(len(trace), 120)
        self.assertLess(np.median(abs(trace[:, 1] - np.interp(trace[:, 0], times, target))), 0.1)

    def test_profile_fits_scale_but_requires_whole_trace_coverage(self):
        arrival = np.arange(8, dtype=float)
        ratio = np.linspace(1.1, 0.9, len(arrival))
        observations = np.column_stack((arrival[1:5], 125 * ratio[1:5]))
        mse, sources = profile([(arrival, ratio)], [observations], [(0, 5)], np.array([-1, 0, 2, 3]))
        np.testing.assert_array_equal(np.isfinite(mse), [False, True, True, False])
        self.assertAlmostEqual(mse[1], 0.0)
        self.assertAlmostEqual(sources[0][1], 125.0)

    def test_pairs_are_distinct_ordered_addresses_and_include_coverage_endpoint(self):
        emission = np.array([0.0, 2.0])
        position = np.array([[343.0, 0, 0], [343.0, 0, 0]])
        velocity = np.zeros_like(position)
        first = ("a11111", emission, position, velocity)
        first_duplicate_segment = ("a11111", emission, position, velocity)
        second = ("a22222", emission, position, velocity)
        observations = [np.array([[1.0, 120.0], [3.0, 120.0]])] * 2
        ranked = score_shard((np.array([0, 2]), [first, first_duplicate_segment, second]),
                             (0, 0, 0), observations, [(1, 3), (1, 3)], None)
        self.assertEqual(set(ranked), {("a11111", "a22222"), ("a22222", "a11111")})
        self.assertTrue(all(loss == 0 and clock == 0 for loss, clock, *_ in ranked.values()))

    @unittest.skipUnless(os.environ.get("LIVERMORE_AUDIO_DIR"), "requires the exact staged study WAVs")
    def test_archive_audio_ridge_frame_receipts(self):
        directory = Path(os.environ["LIVERMORE_AUDIO_DIR"])
        for (name, digest, sample_count), expected in zip(CLIPS, [(151, 72, 5), (164, 92, 4)]):
            with self.subTest(clip=name):
                path = directory / name
                verify(path, digest, sample_count)
                trace, _ = harmonic_ridge(path)
                training, heldout = split_trace(trace)
                self.assertEqual((len(trace), len(training), len(heldout)), expected)


if __name__ == "__main__":
    unittest.main()
