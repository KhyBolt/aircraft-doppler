"""Reproducible STFT and bilinear, equal-duration curve scoring."""
from dataclasses import dataclass
from pathlib import Path
import math
import subprocess
import tempfile
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, resample_poly
from scipy.interpolate import RegularGridInterpolator


@dataclass
class Spectrogram:
    t: np.ndarray
    f: np.ndarray
    power: np.ndarray
    whitened: np.ndarray

    def __post_init__(self):
        for name in ("t", "f", "power", "whitened"):
            raw = np.asarray(getattr(self, name))
            if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
                raise ValueError(f"Spectrogram {name} must be a real numeric array")
            setattr(self, name, raw)
        if any(axis.ndim != 1 or len(axis) < 2 or not np.all(np.isfinite(axis))
               or axis[0] < 0 or np.any(np.diff(axis.astype(float)) <= 0) for axis in (self.t, self.f)):
            raise ValueError("Spectrogram axes must be finite, nonnegative and strictly increasing")
        shape = (len(self.f), len(self.t))
        if (self.power.shape != shape or self.whitened.shape != shape
                or not np.all(np.isfinite(self.power)) or not np.all(np.isfinite(self.whitened))
                or np.any(self.power < 0)):
            raise ValueError("Spectrogram arrays must be finite, shaped [frequency, time], with nonnegative power")

    def sampler(self):
        return RegularGridInterpolator((self.t, self.f), self.whitened.T,
                                       bounds_error=False, fill_value=np.nan)


def _decode_media(path):
    """Decode the first audio stream without downmixing or changing its rate."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "decoded.wav"
        command = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-xerror",
                   "-protocol_whitelist", "file", "-i", str(path.resolve()),
                   "-map", "0:a:0", "-c:a", "pcm_f32le", str(out)]
        try:
            subprocess.run(command, check=True, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise ValueError("FFmpeg is required to decode this media. Install ffmpeg or use a PCM WAV.") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
            if "matches no streams" in detail or "does not match any streams" in detail:
                raise ValueError(f"No audio stream in {path}") from exc
            hint = detail.splitlines()[-1] if detail else "unknown decode error"
            raise ValueError(f"FFmpeg could not decode the first audio stream in {path}: {hint}") from exc
        return wavfile.read(out)


def read_audio(path, target_rate=2000, channel=0):
    """Read PCM WAV directly, or use FFmpeg for the first media audio stream."""
    path = Path(path)
    if not isinstance(target_rate, (int, np.integer)) or target_rate < 1:
        raise ValueError("Target sample rate must be a positive integer")
    if not isinstance(channel, (int, np.integer)) or channel < 0:
        raise ValueError("Audio channel must be a nonnegative integer")
    if not path.is_file():
        raise ValueError(f"Audio input file does not exist or is not a regular file: {path}")
    if path.suffix.lower() == ".wav":
        try:
            sr, x = wavfile.read(path)
        except (ValueError, EOFError):
            sr, x = _decode_media(path)
    else:
        sr, x = _decode_media(path)
    if not len(x):
        raise ValueError(f"Audio stream contains no samples: {path}")
    if x.ndim == 1 and channel != 0:
        raise ValueError("Only channel 0 exists in mono audio")
    if x.ndim > 1:
        if channel < 0 or channel >= x.shape[1]:
            raise ValueError("Audio channel out of range")
        x = x[:, channel]
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        x = (x.astype(float)-(info.max+1)/2) / ((info.max+1)/2) if info.min == 0 else x.astype(float)/max(abs(info.min), info.max)
    else:
        x = x.astype(float)
    if not np.all(np.isfinite(x)):
        raise ValueError("Nonfinite audio")
    if sr != target_rate:
        divisor = math.gcd(int(sr), int(target_rate))
        x = resample_poly(x, target_rate//divisor, sr//divisor)
    return target_rate, x


def make_spectrogram(x, sr, window_s=2.0, hop_s=0.25, fmin=40, fmax=700):
    x = np.asarray(x)
    if x.ndim != 1 or not np.all(np.isfinite(x)) or not np.all(np.isfinite([sr, window_s, hop_s, fmin, fmax])):
        raise ValueError("Expected finite mono audio and finite STFT settings")
    nperseg, hop = round(window_s*sr), round(hop_s*sr)
    if not 1 <= hop <= nperseg <= len(x) or not 0 <= fmin < fmax <= sr/2:
        raise ValueError("Invalid STFT settings, audio too short, or frequency above Nyquist")
    f, t, z = stft(x, sr, window="hann", nperseg=nperseg,
                   noverlap=nperseg-hop, boundary=None, padded=False)
    select = (f >= fmin) & (f <= fmax)
    power = abs(z[select])**2
    db = 10*np.log10(np.maximum(power, 1e-20))
    background = np.median(db, axis=1, keepdims=True)
    scale = np.maximum(1.4826*np.median(abs(db-background), axis=1, keepdims=True), 1.0)
    w = (db-background)/scale
    # Suppress broadband impulses without rewarding individual bright columns.
    w -= np.median(w, axis=0, keepdims=True)
    return Spectrogram(t, f[select], power, np.clip(w, -5, 20))


def save_spectrogram(spec, path):
    np.savez_compressed(path, t=spec.t, f=spec.f, power=spec.power, whitened=spec.whitened)


def load_spectrogram(path):
    with np.load(path, allow_pickle=False) as data:
        required = {"t", "f", "power", "whitened"}
        if not required <= set(data.files):
            raise ValueError("Missing required spectrogram arrays")
        return Spectrogram(**{k: data[k] for k in required})


def best_frequency_score(spec, curve, event, audio_start_s, sampler=None):
    """Sweep source f0, preserving all timing from a single audio start.

    Every candidate must cover the entire scored window. Otherwise boundary
    truncation could reward a candidate that matches only a small easy part.
    """
    sampler = sampler or spec.sampler()
    audio_t = spec.t[(spec.t >= event.window_s[0]) & (spec.t <= event.window_s[1])]
    if len(audio_t) < 4:
        raise ValueError("Event window contains too few STFT frames")
    absolute_t = audio_t+audio_start_s
    if absolute_t[0] < curve.arrival_s[0] or absolute_t[-1] > curve.arrival_s[-1]:
        return -np.inf, np.nan
    ratio = np.interp(absolute_t, curve.arrival_s, curve.ratio)
    f0 = event.frequencies()
    if event.trace_points is not None:
        trace = np.asarray(event.trace_points, float)
        # Compare independently measured frames directly. In particular,
        # interpolate the *model*, not gaps in a detected ridge.
        times = trace[:, 0]
        within = (times > event.window_s[0]) & (times < event.window_s[1])
        support_t = np.concatenate(([event.window_s[0]], times[within], [event.window_s[1]]))
        absolute_t = support_t + audio_start_s
        if absolute_t[0] < curve.arrival_s[0] or absolute_t[-1] > curve.arrival_s[-1]:
            return -np.inf, np.nan
        ratio = np.interp(absolute_t, curve.arrival_s, curve.ratio)
        observed = np.interp(support_t, times, trace[:, 1])
        # Trace the fundamental, or a chosen harmonic and set f0 accordingly.
        residual = abs(f0[:,None]*ratio[None,:]-observed)/event.trace_sigma_hz
        huber = np.where(residual <= 1, 0.5*residual**2, residual-0.5)
        total = -huber.mean(axis=1)
        ix = int(np.argmax(total))
        return float(total[ix]), float(f0[ix])
    total = np.zeros(len(f0))
    weights = np.asarray(event.harmonic_weights, float)
    valid = np.ones(len(f0), bool)
    for harmonic, weight in zip(event.harmonics, weights):
        frequencies = f0[:, None]*ratio[None, :]*harmonic
        points = np.column_stack((np.broadcast_to(audio_t, frequencies.shape).ravel(), frequencies.ravel()))
        brightness = sampler(points).reshape(frequencies.shape)
        valid &= np.all(np.isfinite(brightness), axis=1)
        total += weight*np.nan_to_num(brightness).mean(axis=1)
    total /= weights.sum()
    total[~valid] = -np.inf
    ix = int(np.argmax(total))
    return float(total[ix]), float(f0[ix])
