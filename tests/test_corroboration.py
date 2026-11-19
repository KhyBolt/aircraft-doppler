import unittest

import numpy as np
import pandas as pd
from aircraft_doppler.corroboration import split_trace, local_jitter_hz, check_locked_hypothesis
from aircraft_doppler.physics import Track, forward_curve
from aircraft_doppler.wide import _trace_loss


class GuardedCorroborationTests(unittest.TestCase):
    def test_full_hann_guard_and_block_counts(self):
        t = 173.25 + np.arange(128)/2
        train, blocks = split_trace(np.column_stack((t, 100+.01*t)))
        self.assertEqual(len(blocks),4)
        self.assertGreaterEqual(len(train),32)
        for block in blocks:
            self.assertGreaterEqual(np.min(abs(train[:,0,None]-block[None,:,0])),2)
        self.assertAlmostEqual(local_jitter_hz(blocks[0],.5),.0625)

    def test_scoring_must_cover_full_window_not_only_fit_subset(self):
        t = np.arange(1.,11.)
        q = 1.-.002*t
        obs = np.column_stack((t[2:-2], (120*q)[2:-2]))
        curve = (t[1:-1],q[1:-1])
        self.assertAlmostEqual(_trace_loss(curve,obs,(80,160),np.array([0.]))[0][0],0.)
        self.assertTrue(np.isinf(_trace_loss(curve,obs,(80,160),np.array([0.]),(t[0],t[-1]))[0][0]))

    def test_fixed_hypothesis_on_guards_rejects_one_wrong_event_without_refit(self):
        def synthetic_track(label,epoch,offset):
            times = epoch + np.arange(0.,183.,2.)
            pos = np.column_stack([100*(times-epoch-90.), 800.+0*times, 1400.+0*times])
            return Track(label,times,pos+offset)
        track_a = synthetic_track("a",0.,np.array([0.,0.,0.]))
        track_b = synthetic_track("b",640.,np.array([0.,200.,200.]))
        y = [0.,0.,0.]
        clock = 15.
        tracks = [track_a,track_b]
        sources = [114.,103.]
        traces=[]
        for track, f0 in zip(tracks,sources):
            curve = forward_curve(track,y)
            ts = curve.arrival_s[0]+40+np.arange(128)/2
            traces.append(np.column_stack((ts-clock,np.interp(ts,curve.arrival_s,curve.ratio)*f0)))
        train, held = zip(*(split_trace(t) for t in traces))
        win = pd.Series(dict(x_m=0.,y_m=0.,z_m=0.,flight_pair="a|b",clock_offset_s=clock,
                             event_1_source_hz=sources[0],event_2_source_hz=sources[1]))
        _, reasons = check_locked_hypothesis(win, tracks, train, held, 343., .5)
        self.assertFalse(reasons)
        for block in held[1]:
            block[:,1] += .7
        diagnostics,reasons = check_locked_hypothesis(win, tracks, train, held, 343., .5)
        self.assertEqual([a["passing_fraction"] for a in diagnostics],[1.,0.])
        self.assertEqual(len(reasons),1)
        self.assertTrue(reasons[0].startswith("Event 2"))


if __name__ == "__main__":
    unittest.main()
