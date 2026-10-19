#!/usr/bin/env python3
"""
Extract the 80 one-dimensional emotion-recognition features from one or more audio
files, matching emotion_audio_features.csv. Nothing is predicted or estimated:
gender / emotion columns are only written if you pass them on the command line.

Dependencies:
    pip install librosa praat-parselmouth numpy scipy pandas soundfile

Usage:
    python extract_emotion_features.py clip.wav
    python extract_emotion_features.py clip1.wav clip2.mp3 --out features.csv
    python extract_emotion_features.py ./my_audio_folder --emotion angry --gender female  # labels you supply
    python extract_emotion_features.py clip.wav --with-filename

Output columns (same order as the dataset, minus sample_id, speaker_id, emotion_id):
    [gender], <80 features>, [emotion]      (gender / emotion only if given via flags)
"""
import argparse
import glob
import os
import warnings

import librosa
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

SR = 16000          # resample everything to 16 kHz mono
N_FFT = 1024        # 64 ms analysis window
HOP = 160           # 10 ms hop
SILENCE_DB = -35.0  # frames this far below the loudest frame count as silence
MIN_PAUSE_S = 0.10  # silent runs shorter than this are not counted as pauses
F0_MIN, F0_MAX = 75, 500
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


# --------------------------------------------------------------------------- helpers
def _slope(x, y):
    """Least-squares slope of y against x (nan if not enough points)."""
    if len(x) < 3:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def _nanmean(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else np.nan


def _runs(mask):
    """Return (start, length) of every run of True in a boolean array."""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(mask) - start))
    return runs


# --------------------------------------------------------------------------- feature groups
def pitch_features(snd):
    pitch = snd.to_pitch(time_step=HOP / SR, pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
    f0 = pitch.selected_array["frequency"]
    t = pitch.xs()
    voiced = f0 > 0
    f0v, tv = f0[voiced], t[voiced]

    feats = {}
    if f0v.size:
        feats["f0_mean_hz"] = f0v.mean()
        feats["f0_median_hz"] = np.median(f0v)
        feats["f0_std_hz"] = f0v.std()
        feats["f0_range_hz"] = np.percentile(f0v, 95) - np.percentile(f0v, 5)
        feats["f0_slope_hz_per_s"] = _slope(tv, f0v)
    else:
        for k in ["f0_mean_hz", "f0_median_hz", "f0_std_hz", "f0_range_hz", "f0_slope_hz_per_s"]:
            feats[k] = np.nan
    feats["voiced_frame_ratio"] = voiced.mean()
    return feats, f0, t, voiced


def energy_features(rms, rms_db_rel):
    times = np.arange(len(rms)) * HOP / SR
    active = rms_db_rel > SILENCE_DB
    r = rms[active] if active.any() else rms
    db = 20 * np.log10(np.maximum(rms, 1e-8))
    dba = db[active] if active.any() else db
    return {
        "rms_mean": rms.mean(),
        "rms_std": rms.std(),
        "rms_max": rms.max(),
        "rms_dynamic_range_db": np.percentile(dba, 95) - np.percentile(dba, 5),
        "rms_slope": _slope(times, rms),
        "loudness_db": 20 * np.log10(max(rms.mean(), 1e-8)) + 90,
    }


def timing_features(rms, rms_db_rel, voiced):
    silent = rms_db_rel < SILENCE_DB
    active_idx = np.where(~silent)[0]
    if active_idx.size == 0:
        return {k: np.nan for k in ["speaking_rate_syl_per_s", "pause_ratio", "mean_pause_len_s",
                                    "num_pauses", "voiced_unvoiced_ratio"]}
    first, last = active_idx[0], active_idx[-1]
    inner = silent[first:last + 1]                      # ignore leading/trailing silence
    min_len = int(MIN_PAUSE_S * SR / HOP)
    pauses = [ln for _, ln in _runs(inner) if ln >= min_len]

    speech_frames = max(int((~silent).sum()), 1)
    speech_dur = speech_frames * HOP / SR

    # syllable nuclei = peaks of the smoothed loudness envelope
    env = np.convolve(rms, np.ones(5) / 5, mode="same")
    env_db = 20 * np.log10(np.maximum(env, 1e-8))
    peaks, _ = find_peaks(env_db, height=env_db.max() - 25, prominence=3,
                          distance=int(0.12 * SR / HOP))

    n_voiced = int(voiced.sum())
    n_unvoiced = max(len(voiced) - n_voiced, 1)
    return {
        "speaking_rate_syl_per_s": len(peaks) / speech_dur,
        "pause_ratio": (sum(pauses) / max(len(inner), 1)),
        "mean_pause_len_s": (np.mean(pauses) * HOP / SR) if pauses else 0.0,
        "num_pauses": len(pauses),
        "voiced_unvoiced_ratio": n_voiced / n_unvoiced,
    }


def voice_quality_features(snd, S_pow, freqs, f0_frames_voiced, f0_median):
    feats = {}
    # jitter / shimmer / HNR via Praat
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", F0_MIN, F0_MAX)
        feats["jitter_percent"] = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
        feats["shimmer_percent"] = call([snd, pp], "Get shimmer (local)",
                                        0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    except Exception:
        feats["jitter_percent"] = feats["shimmer_percent"] = np.nan
    try:
        harm = call(snd, "To Harmonicity (cc)", 0.01, F0_MIN, 0.1, 1.0)
        h = harm.values.flatten()
        feats["hnr_db"] = h[h > -100].mean() if (h > -100).any() else np.nan
    except Exception:
        feats["hnr_db"] = np.nan

    # long-term average spectrum (LTAS) of the whole clip, in dB
    ltas = S_pow.mean(axis=1)
    ltas_db = 10 * np.log10(np.maximum(ltas, 1e-12))

    band = (freqs >= 100) & (freqs <= 5000)
    feats["spectral_tilt_db_per_oct"] = _slope(np.log2(freqs[band]), ltas_db[band])

    lo = ltas[(freqs >= 50) & (freqs < 1000)].sum()
    hi = ltas[(freqs >= 1000) & (freqs <= 5000)].sum()
    feats["alpha_ratio_db"] = 10 * np.log10(max(hi, 1e-12) / max(lo, 1e-12))

    # H1-H2: first two harmonic amplitudes in the average voiced-frame spectrum
    feats["h1_h2_db"] = np.nan
    if f0_frames_voiced.any() and np.isfinite(f0_median):
        vs = S_pow[:, f0_frames_voiced].mean(axis=1)
        vs_db = 10 * np.log10(np.maximum(vs, 1e-12))

        def peak_near(f):
            m = (freqs >= 0.9 * f) & (freqs <= 1.1 * f)
            return vs_db[m].max() if m.any() else np.nan

        feats["h1_h2_db"] = peak_near(f0_median) - peak_near(2 * f0_median)
    return feats


def spectral_features(y, S_mag):
    cen = librosa.feature.spectral_centroid(S=S_mag, sr=SR)[0]
    bw = librosa.feature.spectral_bandwidth(S=S_mag, sr=SR)[0]
    roll = librosa.feature.spectral_rolloff(S=S_mag, sr=SR)[0]
    flat = librosa.feature.spectral_flatness(S=S_mag)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0]
    Sn = S_mag / (S_mag.max() + 1e-12)
    flux = np.sqrt((np.diff(Sn, axis=1) ** 2).sum(axis=0))

    out = {}
    for name, arr in [("centroid", cen), ("bandwidth", bw), ("rolloff", roll),
                      ("flux", flux), ("flatness", flat)]:
        unit = "_hz" if name in ("centroid", "bandwidth", "rolloff") else ""
        out[f"spec_{name}_mean{unit}"] = arr.mean()
        out[f"spec_{name}_std{unit}"] = arr.std()
    out["zcr_mean"] = zcr.mean()
    out["zcr_std"] = zcr.std()
    return out


def formant_features(snd, f0, f0_times, voiced, max_frames=150):
    feats = {f"formant_f{i}_hz": np.nan for i in (1, 2, 3)}
    feats.update({f"formant_f{i}_bw_hz": np.nan for i in (1, 2, 3)})
    try:
        formant = call(snd, "To Formant (burg)", HOP / SR, 5, 5500, 0.025, 50)
        tv = f0_times[voiced]
        if tv.size == 0:
            return feats
        tv = tv[np.linspace(0, tv.size - 1, min(max_frames, tv.size)).astype(int)]
        for i in (1, 2, 3):
            fr = [call(formant, "Get value at time", i, t, "Hertz", "Linear") for t in tv]
            bw = [call(formant, "Get bandwidth at time", i, t, "Hertz", "Linear") for t in tv]
            feats[f"formant_f{i}_hz"] = _nanmean(fr)
            feats[f"formant_f{i}_bw_hz"] = _nanmean(bw)
    except Exception:
        pass
    return feats


def mfcc_features(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, n_fft=N_FFT, hop_length=HOP)
    delta = librosa.feature.delta(mfcc)
    out = {}
    for i in range(13):
        out[f"mfcc{i+1}_mean"] = mfcc[i].mean()
    for i in range(13):
        out[f"mfcc{i+1}_std"] = mfcc[i].std()
    for i in range(13):
        out[f"mfcc{i+1}_delta_mean"] = delta[i].mean()
    return out


# --------------------------------------------------------------------------- main extractor
def extract_features(path):
    """Return an ordered dict of the 80 features for one audio file."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) < SR * 0.5:
        raise ValueError(f"{path}: audio shorter than 0.5 s")
    y = y / (np.max(np.abs(y)) + 1e-9)                  # peak-normalise
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=SR)

    S_mag = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    S_pow = S_mag ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    rms = librosa.feature.rms(S=S_mag, frame_length=N_FFT, hop_length=HOP)[0]
    rms_db_rel = librosa.amplitude_to_db(rms, ref=np.max)

    # align Praat pitch frames to librosa frames
    pitch_f, f0, f0_t, voiced = pitch_features(snd)
    frame_t = librosa.frames_to_time(np.arange(S_mag.shape[1]), sr=SR, hop_length=HOP)
    idx = np.clip(np.searchsorted(f0_t, frame_t), 0, len(f0_t) - 1)
    voiced_on_frames = voiced[idx]

    feats = {}
    feats.update(pitch_f)
    feats.update(energy_features(rms, rms_db_rel))
    feats.update(timing_features(rms, rms_db_rel, voiced))
    feats.update(voice_quality_features(snd, S_pow, freqs, voiced_on_frames,
                                        pitch_f["f0_median_hz"]))
    feats.update(spectral_features(y, S_mag))
    feats.update(formant_features(snd, f0, f0_t, voiced))
    feats.update(mfcc_features(y))
    return feats


def collect_files(inputs):
    files = []
    for p in inputs:
        if os.path.isdir(p):
            for ext in AUDIO_EXT:
                files += glob.glob(os.path.join(p, "**", f"*{ext}"), recursive=True)
        else:
            files.append(p)
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser(description="Extract 1D emotion features from audio files.")
    ap.add_argument("inputs", nargs="+", help="audio files and/or folders")
    ap.add_argument("--out", default="extracted_features.csv", help="output CSV path")
    ap.add_argument("--emotion", default="",
                    help="optional: write this emotion label as-is into an 'emotion' column")
    ap.add_argument("--gender", default="", choices=["", "male", "female"],
                    help="optional: write this gender as-is into a 'gender' column")
    ap.add_argument("--with-filename", action="store_true", help="add a 'file' column")
    args = ap.parse_args()

    rows = []
    for f in collect_files(args.inputs):
        try:
            feats = extract_features(f)
        except Exception as e:
            print(f"[skip] {f}: {e}")
            continue
        row = {}
        if args.with_filename:
            row["file"] = f
        if args.gender:
            row["gender"] = args.gender
        row.update(feats)
        if args.emotion:
            row["emotion"] = args.emotion
        rows.append(row)
        print(f"[ok]   {f}")

    if not rows:
        raise SystemExit("No features extracted.")
    df = pd.DataFrame(rows)
    df["num_pauses"] = df["num_pauses"].astype("Int64")
    df.round(4).to_csv(args.out, index=False)
    print(f"Saved {len(df)} row(s) x {df.shape[1]} columns -> {args.out}")


if __name__ == "__main__":
    main()
