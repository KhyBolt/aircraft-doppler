"""Initial ideal straight-flight curve fit to a manually traced tone.

Used only to suggest generous pruning ranges. The full-geometry search is
what distinguishes turning, climbing and accelerating aircraft.
"""
import numpy as np
from scipy.optimize import least_squares


def ideal_received_frequency(audio_time_s, speed_mps, slant_cpa_m, f0_hz, delayed_cpa_s, c=343.0):
    if not 0 < speed_mps < c or slant_cpa_m <= 0 or f0_hz <= 0:
        raise ValueError("Positive frequency, distance and subsonic speed required")
    s = np.asarray(audio_time_s)-delayed_cpa_s+slant_cpa_m/c
    denominator = c*c-speed_mps*speed_mps
    tau = (c*c*s - np.sqrt(c*c*speed_mps*speed_mps*s*s + denominator*slant_cpa_m*slant_cpa_m))/denominator
    distance = np.hypot(slant_cpa_m,speed_mps*tau)
    rdot = speed_mps*speed_mps*tau/distance
    return f0_hz/(1+rdot/c)


def fit_ideal_trace(times, frequencies, initial, c=343.0):
    times,frequencies = np.asarray(times,float),np.asarray(frequencies,float)
    if len(times) < 12 or times.shape != frequencies.shape or not np.all(np.isfinite(times)) or not np.all(np.isfinite(frequencies)) or np.any(np.diff(times) <= 0):
        raise ValueError("Trace needs >=12 finite ordered time/frequency observations")
    def residual(p):
        return ideal_received_frequency(times,*p,c)-frequencies
    lower = [1,10,1,times[0]-120]
    upper = [c*0.95,50000,10000,times[-1]+120]
    fitted = least_squares(residual,initial,bounds=(lower,upper),loss="soft_l1",f_scale=0.5,x_scale="jac")
    names = ["speed_mps","slant_cpa_m","f0_hz","delayed_cpa_audio_s"]
    result = dict(zip(names,map(float,fitted.x)))
    result.update({"rmse_hz":float(np.sqrt(np.mean(residual(fitted.x)**2))),
                   "success":bool(fitted.success),"warning":"Ideal constant-speed approximation only. Bounds are not confidence intervals; inspect residuals before pruning."})
    return result
