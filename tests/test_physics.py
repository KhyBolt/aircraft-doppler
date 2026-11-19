import unittest
import numpy as np
from scipy.optimize import brentq
from aircraft_doppler.physics import Track, forward_curve, closest_approach, geodetic_to_ecef


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(-100,101,2.0)

    def test_stationary_source_and_propagation_delay(self):
        tr = Track("fixed",self.t,np.tile([3430,0,0],(len(self.t),1)))
        cv = forward_curve(tr,[0,0,0])
        np.testing.assert_allclose(cv.arrival_s-cv.emission_s,10)
        np.testing.assert_allclose(cv.ratio,1)

    def test_independent_retarded_time_root(self):
        tr = Track("linear",self.t,np.column_stack((90*self.t,np.full(len(self.t),2500),np.zeros(len(self.t)))))
        cv = forward_curve(tr,[0,0,0],step_s=0.02)
        for receive in (-45,0,45):
            emit = brentq(lambda s:s+np.hypot(90*s,2500)/343-receive,-100,100)
            r = np.hypot(90*emit,2500)
            expected = 1/(1+(90*90*emit/r)/343)
            self.assertAlmostEqual(np.interp(receive,cv.arrival_s,cv.ratio),expected,places=6)
        self.assertGreater(np.interp(-45,cv.arrival_s,cv.ratio),1)
        self.assertLess(np.interp(45,cv.arrival_s,cv.ratio),1)

    def test_constant_delay_is_not_equivalent(self):
        tr = Track("linear",self.t,np.column_stack((100*self.t,np.full(len(self.t),2000),np.zeros(len(self.t)))))
        cv = forward_curve(tr,[0,0,0])
        wrong_time = cv.emission_s+2000/343
        error = np.max(abs(np.interp(cv.arrival_s,wrong_time,cv.ratio)-cv.ratio))
        self.assertGreater(error,0.01)

    def test_cpa_and_slant_range(self):
        tr = Track("linear",self.t,np.column_stack((90*self.t,np.full(len(self.t),3000),np.full(len(self.t),4000))))
        cpa = closest_approach(tr,[180,0,0])
        self.assertAlmostEqual(cpa.emission_s,2,places=6)
        self.assertAlmostEqual(cpa.range_m,5000,places=6)
        self.assertAlmostEqual(cpa.arrival_s,2+5000/343,places=6)

    def test_track_gaps_fail_instead_of_inventing_path(self):
        with self.assertRaises(ValueError):
            Track("gap",[0,1,2,100],np.zeros((4,3)))

    def test_supersonic_tracks_fail(self):
        tr = Track("supersonic",self.t,np.column_stack((400*self.t,np.full(len(self.t),2000),np.zeros(len(self.t)))))
        with self.assertRaises(ValueError):
            forward_curve(tr,[0,0,0])

    def test_wgs84_reference_points(self):
        np.testing.assert_allclose(geodetic_to_ecef(0,0,0),[6378137,0,0],atol=1e-6)
        np.testing.assert_allclose(geodetic_to_ecef(90,0,0),[0,0,6356752.314245],atol=1e-5)
        self.assertAlmostEqual(np.linalg.norm(geodetic_to_ecef(0,0,10)-geodetic_to_ecef(0,0,0)),10)

    def test_straight_track_has_mirror_location_ambiguity(self):
        track = Track("line",self.t,np.column_stack((90*self.t,np.zeros(len(self.t)),np.full(len(self.t),2000))))
        left,right = forward_curve(track,[0,1500,0]),forward_curve(track,[0,-1500,0])
        np.testing.assert_allclose(left.arrival_s,right.arrival_s)
        np.testing.assert_allclose(left.ratio,right.ratio)

    def test_initial_measurement_fit_against_independent_root(self):
        from aircraft_doppler.measurement import fit_ideal_trace
        received = np.linspace(20,140,160)
        emitted = np.array([brentq(lambda s:s+np.hypot(92*(s-75),2300)/343-r,-100,200) for r in received])
        delta = emitted-75
        rdot = 92*92*delta/np.hypot(92*delta,2300)
        frequency = 132/(1+rdot/343)
        result = fit_ideal_trace(received,frequency,[85,2000,130,80])
        self.assertAlmostEqual(result["speed_mps"],92,places=3)
        self.assertAlmostEqual(result["slant_cpa_m"],2300,places=2)
        self.assertAlmostEqual(result["f0_hz"],132,places=3)
        self.assertAlmostEqual(result["delayed_cpa_audio_s"],75+2300/343,places=3)


if __name__ == "__main__":
    unittest.main()
