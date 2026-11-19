"""Stationary receiver, subsonic source, homogeneous still atmosphere.

All positions in one Cartesian frame and vertical datum, in metres.
Times can be Unix seconds or seconds relative to a documented common epoch.
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline


def geodetic_to_ecef(lat, lon, ellipsoid_height_m):
    lat, lon, h = np.broadcast_arrays(np.deg2rad(lat), np.deg2rad(lon), ellipsoid_height_m)
    if np.any(~np.isfinite(lat)) or np.any(abs(lat) > np.pi / 2) or np.any(~np.isfinite(lon)) or np.any(abs(lon) > np.pi) or np.any(~np.isfinite(h)):
        raise ValueError("Invalid geodetic coordinate")
    a, e2 = 6378137.0, 6.6943799901413165e-3
    n = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    return np.stack(((n+h)*np.cos(lat)*np.cos(lon),
                     (n+h)*np.cos(lat)*np.sin(lon),
                     (n*(1-e2)+h)*np.sin(lat)), axis=-1)


@dataclass
class Track:
    flight_id: str
    t: np.ndarray
    xyz: np.ndarray
    engine: str = "unknown"
    max_gap_s: float = 15.0
    aircraft_id: str | None = None

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float)
        self.xyz = np.asarray(self.xyz, dtype=float)
        if self.t.ndim != 1 or len(self.t) < 4 or self.xyz.shape != (len(self.t), 3):
            raise ValueError("Track needs >=4 observations and an N x 3 position matrix")
        if not np.all(np.isfinite(self.t)) or not np.all(np.isfinite(self.xyz)):
            raise ValueError("Nonfinite flight data")
        if not str(self.flight_id).strip() or "|" in self.flight_id:
            raise ValueError("Track ID is empty or contains reserved separator |")
        if not np.isfinite(self.max_gap_s) or self.max_gap_s <= 0:
            raise ValueError("Maximum track gap must be positive and finite")
        if np.any(np.diff(self.t) <= 0) or np.any(np.diff(self.t) > self.max_gap_s):
            raise ValueError("Track timestamps must increase, with no excessive data gaps; split gaps first")
        self.epoch = float(self.t[0])
        local_t = self.t-self.epoch
        self.spline = CubicSpline(local_t, self.xyz, axis=0, bc_type="natural", extrapolate=False)
        delta = np.diff(local_t)[:, None]
        slopes = self.spline(local_t, 1)
        # Cubic Bézier control hull encloses each spline segment. Its AABB
        # gives a lower bound for the interval minimum without solving roots.
        controls = np.stack((self.xyz[:-1], self.xyz[:-1]+slopes[:-1]*delta/3,
                             self.xyz[1:]-slopes[1:]*delta/3, self.xyz[1:]))
        self.interval_lo = controls.min(axis=0)
        self.interval_hi = controls.max(axis=0)

    def position(self, t):
        return self.spline(np.asarray(t)-self.epoch)

    def velocity(self, t):
        return self.spline(np.asarray(t)-self.epoch, 1)


@dataclass
class Curve:
    arrival_s: np.ndarray
    ratio: np.ndarray
    emission_s: np.ndarray


@dataclass
class CPA:
    emission_s: float
    arrival_s: float
    range_m: float
    speed_mps: float
    at_boundary: bool


def forward_curve(track, observer, c=343.0, step_s=0.25):
    """Parameterize by emission time, then map to reception time.

    t_receive = t_emit + R(t_emit)/c
    f_receive/f_source = 1 / (1 + Rdot(t_emit)/c).
    This is equivalent to retarded-time root solving, without per-bin roots.
    """
    if not np.all(np.isfinite([c, step_s])) or c <= 0 or step_s <= 0:
        raise ValueError("Sound speed and sampling step must be positive")
    te = np.linspace(track.t[0], track.t[-1], int(np.ceil((track.t[-1]-track.t[0])/step_s))+1)
    delta = track.position(te)-_observer_vector(observer)
    r = np.linalg.norm(delta, axis=1)
    v = track.velocity(te)
    if np.any(r <= 0) or np.any(np.linalg.norm(v, axis=1) >= c):
        raise ValueError("Zero range, supersonic source, or invalid interpolation")
    rdot = np.einsum("ij,ij->i", delta, v)/r
    arrival = te+r/c
    if not np.all(np.diff(arrival) > 0):
        raise ValueError("Nonmonotone arrival mapping")
    return Curve(arrival, 1/(1+rdot/c), te)


def _observer_vector(observer):
    observer = np.asarray(observer, dtype=float)
    if observer.shape != (3,) or not np.all(np.isfinite(observer)):
        raise ValueError("Observer must have three finite coordinates")
    return observer


def closest_approach(track, observer, c=343.0):
    """Global minimum over spline intervals, including endpoints.

    The squared distance of a cubic spline to a point is degree six per
    interval. Evaluate all real roots of its degree-five derivative, avoiding
    the assumption of one local minimum along a turning flight.
    """
    if not np.isfinite(c) or c <= 0:
        raise ValueError("Sound speed must be positive and finite")
    observer = _observer_vector(observer)
    dist2 = np.sum((track.xyz-observer)**2, axis=1)
    sample_best = np.argmin(dist2)
    best = (float(dist2[sample_best]), float(track.t[sample_best]))
    delta = np.maximum(np.maximum(track.interval_lo-observer, observer-track.interval_hi), 0)
    lower = np.einsum("ij,ij->i", delta, delta)
    for i in np.flatnonzero(lower <= best[0] + 1e-9*max(1.0, best[0])):
        # CubicSpline coefficients descend in power. Polynomial uses ascending.
        coords = [np.polynomial.Polynomial(track.spline.c[:, i, j][::-1]) - observer[j] for j in range(3)]
        d2 = sum((p*p for p in coords), np.polynomial.Polynomial([0.0]))
        width = track.t[i+1]-track.t[i]
        roots = d2.deriv().roots()
        offsets = [0.0, float(width)] + [float(z.real) for z in roots if abs(z.imag) < 1e-7 and 0 < z.real < width]
        for s in offsets:
            value = float(d2(s))
            if value < best[0]:
                best = (value, float(track.t[i]+s))
    r = float(np.linalg.norm(track.position(best[1])-observer))
    return CPA(best[1], best[1]+r/c, r, float(np.linalg.norm(track.velocity(best[1]))),
               bool(min(abs(best[1]-track.t[0]), abs(best[1]-track.t[-1])) < 0.01))


def interval_closest_range(track, observer, arrival_lo, arrival_hi, c=343.0, step_s=0.5):
    """Conservative sampled proximity for an optional audibility heuristic.

    Returns infinity if there are no modelled arrivals in the interval.
    This is NOT an acoustic propagation or detection model.
    """
    curve = forward_curve(track, observer, c, step_s)
    use = (curve.arrival_s >= arrival_lo) & (curve.arrival_s <= arrival_hi)
    if not np.any(use):
        return np.inf
    r = np.linalg.norm(track.position(curve.emission_s[use])-observer, axis=1)
    return float(np.min(r))
