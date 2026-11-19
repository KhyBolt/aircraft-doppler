"""From-scratch fixtures in a non-geographic Cartesian frame, entirely synthetic."""
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import lfilter
from .benchmark import XYZ, _write_json

DURATION = 1620
SAMPLE_RATE = 2000
HALF_SIDE_M = 5000
COARSE_M = 500

# Coordinates and clock offsets are reserved for the renderer and evaluator.
# Nothing here is read by the inference implementation.
@dataclass(frozen=True)
class Scene:
    slug: str
    xy: tuple
    clock_s: float
    seed: int
    render_c: float = 343.0
    mode: str = "positive"
    split: str = "pilot"
    track_seed: int | None = None


HOLDOUTS = (
    Scene("holdout-nw", (-4431., 4592.), -74.375, 5117, 342.8, split="holdout", track_seed=887),
    Scene("holdout-se", (4657., -3769.), 73.125, 1364, 343.1, split="holdout", track_seed=612),
    Scene("holdout-ne", (4309., 2833.), -51.500, 5706, 342.5, split="holdout", track_seed=442),
    Scene("holdout-sw", (-2254., -4461.), 42.375, 6972, 343.45, split="holdout", track_seed=932),
)
PILOTS_AND_CONTROLS = (
    Scene("west", (-4473.0, 1938.0), -63.375, 4317, 342.7),
    Scene("southeast", (4587.0, -4362.0), 71.625, 2753, 343.4),
    Scene("northeast", (3742.0, 4213.0), -46.250, 8627, 342.4),
    Scene("interior", (-1372.0, -2371.0), 28.875, 3079, 343.3),
    Scene("ambiguous", (-1937.0, 1743.0), -27.750, 6527, 343.0, "ambiguous", "control"),
    Scene("wrong-tracks", (717.0, -1363.0), 34.625, 8893, 343.0, "wrong_tracks", "control"),
    Scene("no-flyovers", (903.0, 1143.0), -54.375, 4051, 343.0, "no_flyovers", "control"),
)


@dataclass(frozen=True)
class PathModel:
    slug: str
    center: float
    angle: float
    x: float
    y: float
    height: float
    speed: float
    turn: float
    climb: float
    phase: float

    def state(self, t):
        u = np.asarray(t) - self.center
        angle = np.deg2rad(self.angle)
        e = np.array([np.cos(angle), np.sin(angle),0])
        n = np.array([-np.sin(angle), np.cos(angle),0])
        z = np.array([0,0,1])
        p = (np.array([self.x,self.y,self.height]) + (self.speed*u+.019*u*u)[...,None]*e
             + (self.turn*np.sin(u/68+self.phase))[...,None]*n
             + (self.climb*u+15*np.sin(u/55))[...,None]*z)
        v = ((self.speed+.038*u)[...,None]*e
             + (self.turn/68*np.cos(u/68+self.phase))[...,None]*n
             + (self.climb+15/55*np.cos(u/55))[...,None]*z)
        return p, v


PATHS = (
    PathModel("sim-17", 270., 14., 0., 1120., 1190., 102., 75., 0.35, 0.1),
    PathModel("sim-41", 1280., 113., -760., 0., 1560., 109., -136., -0.55, 0.7),
    PathModel("sim-22", 281., -7., -90., 1410., 1500., 100., 230., 0.62, 1.2),
    PathModel("sim-05", 245., 66., 900., -1950., 2450., 118., -190., -1.0, 1.6),
    PathModel("sim-37", 1300., 118., -450., 1370., 2060., 97., -200., 1.1, 0.9),
    PathModel("sim-38", 1265., 144., -520., 1100., 1575., 115., 140., 1.4, -0.4),
    PathModel("sim-51", 701., 51., 2160., -700., 1600., 113., 180., -0.6, 1.6),
    PathModel("sim-53", 1004., -42., 500., 500., 1350., 107., -168., 0.4, 1.4),
)


SCENES = (*HOLDOUTS, *PILOTS_AND_CONTROLS)


def case_paths(scene):
    if scene.track_seed is None:
        return PATHS
    rng = np.random.default_rng(scene.track_seed)
    return tuple(replace(path, angle=path.angle+rng.uniform(-13,13),
                         x=path.x+rng.uniform(-900,900),y=path.y+rng.uniform(-900,900),
                         speed=path.speed+rng.uniform(-9,9),height=path.height+rng.uniform(-180,420),
                         turn=path.turn+rng.uniform(-145,145),climb=path.climb+rng.uniform(-.27,.27))
                 for path in PATHS)


def _render(scene, paths):
    sr = SAMPLE_RATE
    rng = np.random.default_rng(scene.seed)
    t = np.arange(sr*DURATION)/sr
    observer = np.array((*scene.xy,0.))
    # Background is neither silent nor simulated airframe signals alone.
    ambient = lfilter([1], [1,-0.82], rng.standard_normal(len(t)).astype(np.float32))
    x = .037*rng.standard_normal(len(t)) + .021*ambient
    x += .044*np.sin(2*np.pi*53*t) + .025*np.sin(2*np.pi*108*t)
    x += .016*(.6+.4*np.cos(2*np.pi*t/21))*np.sin(2*np.pi*(78*t+.2*np.sin(t/3)))
    for k, center in enumerate((96, 510, 700, 1020, 1480)):
        center += rng.uniform(-13,13)
        use = abs(t-center) < 55
        u = t[use]-center
        env = np.exp(-(u/(12+3*k))**4)
        carrier = (117-9*k)
        if k % 2 == 0:
            # A slowly rising narrow-band source; distinct from descending pass.
            x[use] += .066*env*np.sin(2*np.pi*(carrier*u+0.18*u*u))
        else:
            x[use] += .090*env*np.sin(2*np.pi*carrier*u)
    for center in (172, 631, 1106, 1507):
        use = abs(t-center) <= 5
        x[use] += .091*np.exp(-((t[use]-center)/1.7)**2)*rng.standard_normal(use.sum())
    if scene.mode != "no_flyovers":
        for i, (path, f0) in enumerate(zip(paths, (121.1, 104.35))):
            f0 += rng.uniform(-1.0,1.0)
            tt = np.arange(path.center-123,path.center+124,.25)
            p, _ = path.state(tt)
            cpa = tt[np.argmin(np.sum((p-observer)**2,axis=1))]
            mid = cpa + np.linalg.norm(path.state(cpa)[0]-observer)/scene.render_c
            u = t + scene.clock_s - mid
            use = abs(u) <= 57
            tr = t[use]+scene.clock_s
            # Continuous analytic path plus independent retarded-time Newton
            # solve; scoring uses interpolated, sampled observations instead.
            te = tr-10
            for _ in range(9):
                p, v = path.state(te)
                delta = p-observer
                r = np.linalg.norm(delta,axis=1)
                te -= (te+r/scene.render_c-tr)/(1+np.sum(delta*v,axis=1)/(r*scene.render_c))
            r = np.linalg.norm(path.state(te)[0]-observer,axis=1)
            if np.max(abs(te+r/scene.render_c-tr)) > 1e-8:
                raise RuntimeError("Synthetic propagation solve did not converge")
            u = u[use]
            phase = 2*np.pi*f0*(te-path.center)+.28*np.sin((te-path.center)/3.7+scene.seed+i)
            fade = np.exp(-(u/35)**8)*np.minimum(.27,.29*1650/r)
            amplitude = fade*(.91+.08*np.cos(0.37*u+scene.seed)+.045*np.sin(u*.13))
            x[use] += amplitude*(np.sin(phase)+.20*np.sin(2*phase+0.7))
    return np.rint(np.clip(x,-1,1)*32767).astype("<i2")


def make_scene(root, scene):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    paths = case_paths(scene)
    wavfile.write(root/"synthetic.wav", SAMPLE_RATE, _render(scene, paths))
    rows = []
    # Separate sampled track input. Timestamps on a different origin from WAV,
    # with no event IDs or answer coordinates provided to inference.
    for i, path in enumerate(paths):
        dt = 2.25
        ts = path.center+np.arange(-129.375,129.,dt)
        xyz, _ = path.state(ts)
        if scene.mode in ("shifted_tracks","near_decoy") and i < 2:
            if scene.mode == "shifted_tracks":
                ts = ts + (17.25 if i == 0 else -20.5)
            else:
                # Similar speed and route, but with a different lateral
                # course and height profile, not arbitrary distant negatives.
                foil = replace(path,slug=("sim-63","sim-26")[i],angle=path.angle+(3.5,-4.7)[i],
                               height=path.height+(175,-230)[i],speed=path.speed+(2.5,-2.9)[i],
                               x=path.x+(-380,310)[i],y=path.y+(240,-330)[i],
                               turn=path.turn+(93,-101)[i],climb=path.climb+(.25,-.19)[i])
                xyz,_ = foil.state(ts)
                path = foil
        if scene.mode not in ("wrong_tracks","exact_alias") or i >= 2:
            rows.extend((path.slug, path.slug, t, *p, "unknown") for t,p in zip(ts,xyz))
        if scene.mode in ("ambiguous","exact_alias") and i < 2:
            alt = ("sim-79","sim-91")[i]
            rows.extend((alt, alt, t+17, *(p+[2000.,-1500.,0.]), "unknown") for t,p in zip(ts,xyz))
    pd.DataFrame(rows,columns=["flight_id","aircraft_id","time_s",*XYZ,"engine"]).to_csv(root/"tracks.csv",index=False)
    coords = np.arange(-HALF_SIDE_M,HALF_SIDE_M+1,COARSE_M)
    pd.DataFrame([(x,y,0) for x in coords for y in coords],columns=XYZ).to_csv(root/"candidates.csv",index=False)
    correct_pair = None if scene.mode in ("no_flyovers","wrong_tracks","shifted_tracks","near_decoy","exact_alias") else "sim-17|sim-41"
    answer = {"kind":"synthetic only", "scene":scene.slug, "mode":scene.mode, "split":scene.split, "seed":scene.seed, "track_seed":scene.track_seed,
              "stationary_virtual_receiver_xyz_m":list((*scene.xy,0.)) if scene.mode != "no_flyovers" else None,
              "clock_offset_s":scene.clock_s, "render_sound_speed_mps":scene.render_c,
              "correct_flight_pair":correct_pair,
              "expected_outcome":"point" if scene.mode == "positive" else "abstain"}
    if scene.mode in ("ambiguous","exact_alias"):
        answer["equivalent_xyz_m"] = [scene.xy[0]+2000.,scene.xy[1]-1500.,0.]
        answer["equivalent_pair"] = "sim-79|sim-91"
    _write_json(root/"truth.json", answer)
