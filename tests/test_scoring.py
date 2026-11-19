import unittest
import numpy as np
import pandas as pd
from aircraft_doppler.physics import Track, forward_curve, Curve
from aircraft_doppler.audio import Spectrogram, best_frequency_score
from aircraft_doppler.search import Event, SearchSettings, search
from aircraft_doppler.io import geographic_to_cartesian


class ScoringTests(unittest.TestCase):
    def make_track(self, fid, center):
        t = np.arange(center-70,center+70,2.0)
        return Track(fid,t,np.column_stack((90*(t-center),np.full(len(t),1500),np.full(len(t),1500))))

    def trace_event(self,track,window,cpa):
        curve = forward_curve(track,[0,0,0])
        times = np.linspace(*window,301)
        trace = np.column_stack((times,130*np.interp(times,curve.arrival_s,curve.ratio))).tolist()
        return Event(window,cpa,(130,130,1),trace_points=trace)

    def test_shared_timing_penalizes_wrong_second_flight(self):
        a,b,bad = self.make_track("a",80), self.make_track("b",1000), self.make_track("bad",1004)
        e1,e2 = self.trace_event(a,(55,115),86.18),self.trace_event(b,(975,1035),1006.18)
        t,f = np.arange(0,1100,0.25),np.arange(50,250)
        spec = Spectrogram(t,f,np.zeros((len(f),len(t))),np.zeros((len(f),len(t))))
        results,_ = search([a,b,bad],pd.DataFrame([[0,0,0]],columns=["x_m","y_m","z_m"]),[e1,e2],spec,
                           SearchSettings(offset_tolerance_s=10,start_sweep_s=2,start_step_s=0.25))
        right = results[results.flight_pair == "a|b"].iloc[0]
        wrong = results[results.flight_pair == "a|bad"].iloc[0]
        self.assertGreater(right.score,wrong.score+0.1)

    def test_no_partial_curve_reward(self):
        t,f = np.arange(0,10,0.25),np.arange(50,200)
        spec = Spectrogram(t,f,np.ones((len(f),len(t))),np.ones((len(f),len(t))))
        event = Event((1,8),4,(100,100,1))
        curve = Curve(np.array([3,7]),np.array([1,1]),np.array([0,4]))
        score,_ = best_frequency_score(spec,curve,event,0)
        self.assertEqual(score,-np.inf)

    def test_unknown_engine_is_retained(self):
        a = self.make_track("unknown_engine",80)
        event = self.trace_event(a,(55,115),86.18)
        t,f = np.arange(0,160,0.25),np.arange(50,250)
        spec = Spectrogram(t,f,np.zeros((len(f),len(t))),np.zeros((len(f),len(t))))
        config = SearchSettings(known_audio_start_s=0,known_start_tolerance_s=0,start_sweep_s=2)
        result,_ = search([a],pd.DataFrame([[0,0,0]],columns=["x_m","y_m","z_m"]),[event],spec,config)
        self.assertEqual(len(result),1)
        a.engine = "jet"
        result,_ = search([a],pd.DataFrame([[0,0,0]],columns=["x_m","y_m","z_m"]),[event],spec,config)
        self.assertEqual(len(result),0)

    def test_altitude_datum_must_be_explicit(self):
        with self.assertRaises(ValueError):
            geographic_to_cartesian(pd.DataFrame({"latitude":[0],"longitude":[0],"baro_altitude":[1000]}))

    def test_silence_audit_includes_jets_but_is_opt_in(self):
        a,b,extra = self.make_track("a",80),self.make_track("b",1000),self.make_track("extra_jet",500)
        extra.engine = "jet"
        e1,e2 = self.trace_event(a,(55,115),86.18),self.trace_event(b,(975,1035),1006.18)
        t,f = np.arange(0,1100,0.25),np.arange(50,250)
        spec = Spectrogram(t,f,np.zeros((len(f),len(t))),np.zeros((len(f),len(t))))
        candidates = pd.DataFrame([[0,0,0]],columns=["x_m","y_m","z_m"])
        settings = SearchSettings(silence_radius_m=3000,start_sweep_s=0)
        result,_ = search([a,b,extra],candidates,[e1,e2],spec,settings)
        self.assertEqual(len(result),1)
        self.assertEqual(result.iloc[0].silence_extra_count,1)
        settings.reject_silence = True
        result,audit = search([a,b,extra],candidates,[e1,e2],spec,settings)
        self.assertEqual(len(result),0)
        self.assertEqual(audit["silence_rejected"],1)


if __name__ == "__main__":
    unittest.main()
