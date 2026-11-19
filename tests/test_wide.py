import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from aircraft_doppler.physics import Track, forward_curve
from aircraft_doppler.search import Event
from aircraft_doppler.wide import (CLOCKS, Profile, _trace_loss, _minima, _modes, BudgetExceeded, _deadline_check,
                                   evaluate, infer, run)


class WideSyntheticTests(unittest.TestCase):
    def test_global_clock_bounds_and_exact_frequency_profile(self):
        self.assertEqual(len(CLOCKS),161)
        self.assertEqual(tuple(CLOCKS[[0,-1]]),(-80.,80.))
        arrival = np.arange(20.,50.)
        ratio = 1-.009*(arrival-33)
        times = np.arange(25.,33.)
        target_clock,f = 5,120.
        obs = np.column_stack((times,f*np.interp(times+target_clock,arrival,ratio)))
        grid = np.array([-8.,-1.,5.,8.])
        loss,f0 = _trace_loss((arrival,ratio),obs,(100.,125.),grid)
        self.assertTrue(np.isinf(loss[0]))
        self.assertEqual(int(np.argmin(loss)),2)
        self.assertAlmostEqual(loss[2],0)
        self.assertAlmostEqual(f0[2],f)
        _, bounded = _trace_loss((arrival,ratio),obs,(100.,110.),grid)
        self.assertEqual(bounded[2],110.)

    def test_clock_basins_allow_an_inclusive_global_end(self):
        assert_allclose(_minima(np.array([5.,9.,4.,6.,1.])),[4,2,0])
        self.assertEqual(len(_minima(np.array([np.inf,np.inf]))),0)

    def test_one_event_honors_the_requested_clock_prior(self):
        time_s = np.arange(100., 260.)
        track = Track("pass", time_s, np.column_stack((75 * (time_s - 178),
                      np.full_like(time_s, 1000), np.full_like(time_s, 800))))
        observer = (0, 0, 0)
        curve = forward_curve(track, observer)
        audio_t = np.arange(10., 40., .5)
        observed = 120 * np.interp(audio_t + 160, curve.arrival_s, curve.ratio)
        event = Event((audio_t[0], audio_t[-1]), 22, (110, 130, .2),
                      trace_points=np.column_stack((audio_t, observed)).tolist())
        self.assertTrue(Profile([track], [event]).table([observer]).empty)
        recovered = Profile([track], [event], clock_center_s=155, clock_half_s=5).table([observer]).iloc[0]
        self.assertEqual(recovered.flight_pair, "pass")
        self.assertEqual(recovered.clock_offset_s, 160)
        self.assertAlmostEqual(recovered.loss_hz2, 0)
        self.assertAlmostEqual(recovered.event_1_source_hz, 120)

    def test_no_top_pair_quota_drops_viable_distant_alternatives(self):
        points = [(0,0,0), (2000,0,0), (0,2000,0), (2000,2000,0), (-2000,2000,0)]
        table = pd.DataFrame([(*q,.01+i*.01,f"pair_{i}") for i,q in enumerate(points)],
                             columns=["x_m","y_m","z_m","loss_hz2","flight_pair"])
        retained = _modes(table,points,500,0,600,.5)
        self.assertEqual(set(retained),set(points))

    def test_control_evaluation_preserves_abstention(self):
        report = {"status": "abstained","abstention_reasons":["multiple modes"]}
        truth = {"stationary_virtual_receiver_xyz_m":[0,0,0],"clock_offset_s":3,
                 "expected_outcome":"abstain","correct_flight_pair":"independent"}
        score = evaluate(report,truth)
        self.assertIsNone(score["position_error_m"])
        self.assertTrue(score["expected_behavior_observed"])

    def test_deadline_checkpoint_marks_partial_as_unverified(self):
        with self.assertRaises(BudgetExceeded):
            _deadline_check(0)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/"out"
            with patch("aircraft_doppler.wide._deadline_check",side_effect=BudgetExceeded()):
                report = run(dest,case_slugs=["holdout-nw"],max_seconds=1,progress=lambda _:None)
            self.assertEqual(report["status"],"timed_out_partial")
            self.assertEqual(report["cases_completed"],{})
            self.assertEqual(json.loads((dest/"suite_summary.json").read_text())["first_unfinished_case"],"holdout-nw")

    def test_inference_is_independent_of_truth_file(self):
        from aircraft_doppler.wide_fixture import SCENES, make_scene
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"generated"
            make_scene(path,next(s for s in SCENES if s.slug == "no-flyovers"))
            def deny(*args, **kwargs):
                raise AssertionError("inference cannot read JSON ground truth")
            files = [path/name for name in ("synthetic.wav","tracks.csv","candidates.csv")]
            with patch.object(Path,"read_text",deny):
                result = infer(*files,Path(tmp)/"inference")
            self.assertEqual(result["recording_duration_s"],1620.)
            self.assertEqual(result["status"],"abstained")
            self.assertNotIn("estimated_xyz_m",result)


if __name__ == "__main__":
    unittest.main()
