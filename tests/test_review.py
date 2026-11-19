import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.io import wavfile

from aircraft_doppler.audio import Spectrogram, best_frequency_score, load_spectrogram, read_audio
from aircraft_doppler.benchmark import infer, benchmark
from aircraft_doppler.io import load_tracks, load_inputs
from aircraft_doppler.physics import Curve, Track, closest_approach, geodetic_to_ecef
from aircraft_doppler.search import Event, SearchSettings, search, _time_offsets

FIXTURE = Path(__file__).resolve().parents[1]/"fixtures"


class ReviewTests(unittest.TestCase):
    def test_cached_stft_rejects_inverted_axes_bad_power_and_missing_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"spec.npz"
            np.savez(out, t=[0,1])
            with self.assertRaisesRegex(ValueError, "Missing"):
                load_spectrogram(out)
            np.savez(out, t=[1,0], f=[40,41], power=np.ones((2,2)), whitened=np.zeros((2,2)))
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                load_spectrogram(out)
            np.savez(out, t=[0,1], f=[40,41], power=-np.ones((2,2)), whitened=np.zeros((2,2)))
            with self.assertRaisesRegex(ValueError, "nonnegative power"):
                load_spectrogram(out)

    def test_invalid_channel_does_not_silently_use_mono(self):
        with self.assertRaisesRegex(ValueError, "Only channel 0"):
            read_audio(FIXTURE/"synthetic.wav", channel=1)

    def test_mixed_cartesian_and_geographic_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp)/"candidate.csv"
            f.write_text("latitude,longitude,altitude_ellipsoid_m\n13,24,100\n")
            with self.assertRaisesRegex(ValueError, "frames disagree"):
                load_inputs(FIXTURE/"tracks.csv", f)
            f.write_text("latitude,longitude,altitude_ellipsoid_m,x_m,y_m,z_m\n13,24,100,0,0,0\n")
            with self.assertRaisesRegex(ValueError, "disagree with WGS84"):
                load_inputs(FIXTURE/"tracks.csv", f)
        with self.assertRaisesRegex(ValueError, "Invalid"):
            geodetic_to_ecef(23, 240, 0)

    def test_missing_flight_id_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracks = pd.read_csv(FIXTURE/"tracks.csv").head()
            tracks.loc[1,"flight_id"] = ""
            f = Path(tmp)/"tracks.csv"
            tracks.to_csv(f,index=False)
            with self.assertRaisesRegex(ValueError, "nonempty flight IDs"):
                load_tracks(f)

    def test_time_offset_lattice_is_centered_and_inclusive(self):
        assert_allclose(_time_offsets(.1,.25),[-.1,0,.1])
        assert_allclose(_time_offsets(.55,.25),[-.55,-.5,-.25,0,.25,.5,.55])
        assert_allclose(_time_offsets(0,.25),[0])

    def test_sparse_trace_scoring_does_not_invent_observations_between_picks(self):
        times = np.linspace(0,10,61)
        spec = Spectrogram(times, np.array([50,200]), np.ones((2,61)), np.ones((2,61)))
        observed = np.array([0,1,8,10])
        curve = Curve(times, 1+0.02*np.sin(np.pi*times), times)
        event = Event((0,10),5,(100,100,1),trace_points=np.column_stack((observed,[100.]*4)).tolist())
        score, _ = best_frequency_score(spec, curve, event, 0.0)
        self.assertAlmostEqual(score, 0.)

    def test_aircraft_segment_aliases_do_not_count_as_distinct_flyovers(self):
        t = np.arange(0,81,2.)
        position = np.column_stack([60*(t-40), 400+0*t, 1500+0*t])
        tracks = [Track("a",t,position,aircraft_id="same"), Track("b",t+150,position,aircraft_id="same")]
        times = np.arange(0,235.)
        spec = Spectrogram(times, np.array([50,200]), np.ones((2,len(times))), np.ones((2,len(times))))
        events = [Event((20,60),44,(100,100,1)),Event((170,210),194,(100,100,1))]
        candidates = pd.DataFrame({"x_m":[0],"y_m":[0],"z_m":[0]},index=["nonnumeric"])
        progress = []
        result, audit = search(tracks,candidates,events,spec,progress=progress.append)
        self.assertTrue(result.empty)
        self.assertEqual(audit["cpa_eligible_pairs"], 0)
        self.assertEqual(progress, ["Scored 1/1 candidate points"])
        self.assertIn("flight_pair", result.columns)

    def test_jet_filter_is_explicit_per_event(self):
        t = np.arange(0,81,2.)
        track = Track("jet",t,np.column_stack([60*(t-40),400+0*t,1500+0*t]),"jet")
        times = np.arange(0,81.)
        spec = Spectrogram(times,np.array([50,200]),np.ones((2,len(times))),np.ones((2,len(times))))
        candidates = pd.DataFrame({"x_m":[0],"y_m":[0],"z_m":[0]})
        settings = SearchSettings(known_audio_start_s=0,known_start_tolerance_s=.1,start_step_s=.25)
        default,_ = search([track],candidates,[Event((20,65),44.5,(100,100,1))],spec,settings)
        explicit,_ = search([track],candidates,[Event((20,65),44.5,(100,100,1),allowed_engines=("jet",))],spec,settings)
        self.assertTrue(default.empty)
        self.assertFalse(explicit.empty)

    def test_cpa_interval_bounds_preserve_global_minimum(self):
        rng = np.random.default_rng(930)
        for epoch in (0, 1700000000):
            t = epoch+np.arange(0,101,5.)
            q = t-epoch
            track = Track("turn",t,np.stack((350*np.cos(.17*q), 260*np.sin(.17*q), 450+.5*q+40*np.cos(.32*q)),axis=1))
            for observer in rng.uniform(-500,500,(5,3)):
                bounded = closest_approach(track,observer)
                lo,hi = track.interval_lo.copy(),track.interval_hi.copy()
                track.interval_lo[:],track.interval_hi[:] = -np.inf,np.inf
                exhaustive = closest_approach(track,observer)
                track.interval_lo[:],track.interval_hi[:] = lo,hi
                self.assertAlmostEqual(bounded.emission_s,exhaustive.emission_s,places=6)
                self.assertAlmostEqual(bounded.range_m,exhaustive.range_m,places=6)


class OfflineBenchmarkTests(unittest.TestCase):
    def test_truth_is_not_an_inference_dependency_and_heatmap_is_real(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            original = Path.read_text
            def deny_truth_read(obj, *args, **kwargs):
                if obj.name == "truth.json":
                    raise AssertionError("truth read during inference")
                return original(obj,*args,**kwargs)
            with patch.object(Path, "read_text", deny_truth_read):
                report = infer(FIXTURE/"synthetic.wav",FIXTURE/"tracks.csv",FIXTURE/"candidates.csv",out/"blind")
            self.assertEqual(report["status"], "supported_in_this_fixture", report)
            self.assertEqual(report["estimated_pair"], "flyover_a|flyover_b")
            self.assertLess(np.linalg.norm(np.asarray(report["estimated_xyz_m"])-[117,-183,0]),75)
            self.assertEqual(len(report["audio_only_detection"]), 2)
            with self.assertRaisesRegex(ValueError,"new or empty output"):
                infer(FIXTURE/"synthetic.wav",FIXTURE/"tracks.csv",FIXTURE/"candidates.csv",out/"blind")
            # Invoke evaluation for real, using the verified inference once.
            with patch("aircraft_doppler.benchmark.infer", return_value=report):
                summary = benchmark(FIXTURE,out/"blind")
            self.assertLess(summary["position_error_m"],75)
            png = (out/"blind"/"score_heatmap.png").read_bytes()
            self.assertTrue(png.startswith(b"\x89PNG"))
            self.assertGreater(len(png), 30000)
            for grid in ("coarse_grid.csv","fine_grid.csv"):
                self.assertIn("profile_score", (out/"blind"/grid).read_text().splitlines()[0])

    def test_wrong_relative_track_time_abstains_after_detecting_both_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tracks = pd.read_csv(FIXTURE/"tracks.csv")
            tracks.loc[tracks.flight_id.isin(["flyover_b","decoy_time"]),"time_s"] += 32
            path = tmp/"wrong_time.csv"
            tracks.to_csv(path,index=False)
            report = infer(FIXTURE/"synthetic.wav",path,FIXTURE/"candidates.csv",tmp/"results")
            self.assertEqual(report["status"],"abstained")
            self.assertEqual(len(report["audio_only_detection"]),2)
            self.assertNotIn("estimated_xyz_m", report)

    def test_silence_stationary_tones_and_noise_abstain_without_writing_a_pin(self):
        t = np.arange(620000)/2000.
        stationary = np.sin(2*np.pi*108*t)
        noise = np.random.default_rng(8025).normal(0,.09,len(t))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            signals = (np.zeros_like(t), noise+.15*stationary, noise+.10*stationary*
                       (np.exp(-((t-87)/12)**2)+np.exp(-((t-237)/12)**2)))
            for i, audio in enumerate(signals):
                path = tmp/f"control_{i}.wav"
                wavfile.write(path,2000,audio.astype(np.float32))
                output = tmp/f"results_{i}"
                report = infer(path,FIXTURE/"tracks.csv",FIXTURE/"candidates.csv",output)
                self.assertEqual(report["status"],"abstained",report)
                self.assertNotIn("estimated_xyz_m",report)
                self.assertFalse((output/"fine_grid.csv").exists())


if __name__ == "__main__":
    unittest.main()
