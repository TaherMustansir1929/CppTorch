#!/usr/bin/env python3
"""
Extract a 43-feature audio dataset matching the COLUMN SCHEMA of
gender_audio_features_kaggle.csv.

Default inputs:
    /audio/male
    /audio/female

Default output:
    manually_extracted_gender_audio_features.csv

Important:
This reproduces the 43 column names and provides conventional implementations
of the features. It cannot guarantee numerical equivalence to the source CSV
unless the original dataset's exact feature-extraction code and settings are
known. In particular, pitch units/algorithm and spectral-statistic conventions
must be checked against the original source.

Install:
    python -m pip install -r requirements_gender_audio_features_43.txt

Run:
    python extract_gender_audio_features_43.py

Example:
    python extract_gender_audio_features_43.py \
      --male-dir /audio/male --female-dir /audio/female \
      --output manually_extracted_gender_audio_features.csv \
      --workers 4 --male-label 0 --female-label 1
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
from scipy.stats import skew, kurtosis
from tqdm import tqdm

SAMPLE_RATE = 16000
N_FFT = 2048
WIN_LENGTH = 1024
HOP_LENGTH = 512
N_MFCC = 13
N_MELS = 40
EPS = 1e-12
AUDIO_EXTENSIONS = {
    ".wav", ".flac", ".ogg", ".oga", ".mp3", ".m4a", ".aac",
    ".aif", ".aiff", ".opus", ".wma", ".webm",
}

FEATURE_COLUMNS = [
    "mean_spectral_centroid", "std_spectral_centroid",
    "mean_spectral_bandwidth", "std_spectral_bandwidth",
    "mean_spectral_contrast", "mean_spectral_flatness",
    "mean_spectral_rolloff", "zero_crossing_rate", "rms_energy",
    "mean_pitch", "min_pitch", "max_pitch", "std_pitch",
    "spectral_skew", "spectral_kurtosis", "energy_entropy", "log_energy",
]
for i in range(1, 14):
    FEATURE_COLUMNS.extend([f"mfcc_{i}_mean", f"mfcc_{i}_std"])
CSV_COLUMNS = FEATURE_COLUMNS + ["label"]


def _safe_mean(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else 0.0


def _safe_std(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.std(a, ddof=0)) if a.size else 0.0


def _spectral_moments(magnitude: np.ndarray) -> tuple[float, float]:
    """Mean frame-wise skewness and excess kurtosis of spectral magnitudes."""
    skews, kurts = [], []
    for frame in magnitude.T:
        if np.all(frame == frame[0]):
            skews.append(0.0)
            kurts.append(0.0)
        else:
            skews.append(float(skew(frame, bias=True, nan_policy="omit")))
            # Fisher=True gives excess kurtosis (normal distribution -> 0).
            kurts.append(float(kurtosis(frame, fisher=True, bias=True, nan_policy="omit")))
    return _safe_mean(np.asarray(skews)), _safe_mean(np.asarray(kurts))


def _energy_entropy(y: np.ndarray, segments: int = 10) -> float:
    """Mean Shannon entropy of energy proportions in equal frame subdivisions."""
    if y.size == 0:
        return 0.0
    chunks = np.array_split(y, min(segments, y.size))
    energies = np.asarray([np.sum(c * c) for c in chunks], dtype=np.float64)
    total = float(energies.sum())
    if total <= EPS:
        return 0.0
    p = energies / total
    p = p[p > 0]
    # Natural logarithm, normalized to [0, 1].
    entropy = -float(np.sum(p * np.log(p + EPS)))
    return entropy / math.log(max(2, len(chunks)))


def extract_features(audio_path: Path) -> dict[str, float]:
    # Decode to mono and resample so frequencies and frame parameters are stable.
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    if y.size < int(0.1 * SAMPLE_RATE):
        raise ValueError("audio is empty or shorter than 0.1 seconds")
    if not np.all(np.isfinite(y)):
        raise ValueError("audio contains non-finite samples")

    # Do not apply notebook-specific denoising/normalization automatically:
    # those choices alter feature values. Remove DC offset and avoid clipping.
    y = y.astype(np.float32)
    y = y - np.mean(y)
    if np.max(np.abs(y)) > EPS:
        y = y / np.max(np.abs(y))

    # Librosa's STFT uses centered frames by default.
    S_complex = librosa.stft(
        y, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, window="hann", center=True
    )
    magnitude = np.abs(S_complex)
    power = magnitude ** 2

    centroid = librosa.feature.spectral_centroid(
        S=magnitude, sr=sr
    )[0]
    bandwidth = librosa.feature.spectral_bandwidth(
        S=magnitude, sr=sr
    )[0]
    rolloff = librosa.feature.spectral_rolloff(
        S=power, sr=sr, roll_percent=0.85
    )[0]
    flatness = librosa.feature.spectral_flatness(S=power)[0]
    contrast = librosa.feature.spectral_contrast(
        S=power, sr=sr, n_bands=6, fmin=200.0
    )
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    rms = librosa.feature.rms(S=magnitude, frame_length=N_FFT)[0]

    # Pitch: YIN estimates F0 for each frame. Values are Hz, not pitch periods.
    # Restrict to a broad human-voice range; unvoiced frames are excluded.
    f0 = librosa.yin(
        y, fmin=65.0, fmax=500.0, sr=sr,
        frame_length=N_FFT, hop_length=HOP_LENGTH
    )
    # YIN returns an estimate even for unvoiced frames; retain frames whose
    # RMS is above a small threshold, and document that this is an approximation.
    frame_rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    n = min(f0.size, frame_rms.size)
    voiced_f0 = f0[:n][(frame_rms[:n] > max(1e-4, 0.01 * float(np.max(frame_rms)))) &
                        np.isfinite(f0[:n]) & (f0[:n] >= 65.0) & (f0[:n] <= 500.0)]
    if voiced_f0.size == 0:
        # Missing pitch is represented as 0 for a numeric, model-friendly CSV.
        pitch_mean = pitch_min = pitch_max = pitch_std = 0.0
    else:
        pitch_mean = float(np.mean(voiced_f0))
        pitch_min = float(np.min(voiced_f0))
        pitch_max = float(np.max(voiced_f0))
        pitch_std = float(np.std(voiced_f0, ddof=0))

    # Spectral moments use each frame's magnitude spectrum.
    spec_skew, spec_kurt = _spectral_moments(magnitude)

    # Mean frame log-energy; log of frame sum-of-squares with epsilon floor.
    frame_energy = np.sum(power, axis=0)
    log_energy = _safe_mean(np.log(frame_energy + EPS))

    # MFCC: 13 coefficients, then mean and population standard deviation per coeff.
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        n_mels=N_MELS, fmin=0.0, fmax=sr / 2
    )

    result = {
        "mean_spectral_centroid": _safe_mean(centroid),
        "std_spectral_centroid": _safe_std(centroid),
        "mean_spectral_bandwidth": _safe_mean(bandwidth),
        "std_spectral_bandwidth": _safe_std(bandwidth),
        "mean_spectral_contrast": _safe_mean(contrast),
        "mean_spectral_flatness": _safe_mean(flatness),
        "mean_spectral_rolloff": _safe_mean(rolloff),
        "zero_crossing_rate": _safe_mean(zcr),
        "rms_energy": _safe_mean(rms),
        "mean_pitch": pitch_mean,
        "min_pitch": pitch_min,
        "max_pitch": pitch_max,
        "std_pitch": pitch_std,
        "spectral_skew": spec_skew,
        "spectral_kurtosis": spec_kurt,
        "energy_entropy": _energy_entropy(y),
        "log_energy": log_energy,
    }
    for i in range(N_MFCC):
        result[f"mfcc_{i+1}_mean"] = _safe_mean(mfcc[i])
        result[f"mfcc_{i+1}_std"] = _safe_std(mfcc[i])

    # Preserve exact requested schema and guarantee finite CSV values.
    cleaned = {}
    for col in FEATURE_COLUMNS:
        value = float(result[col])
        cleaned[col] = value if math.isfinite(value) else 0.0
    return cleaned


def find_audio_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {folder}")
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract the 43-column gender audio feature schema.")
    parser.add_argument("--male-dir", type=Path, default=Path("audio/male"))
    parser.add_argument("--female-dir", type=Path, default=Path("audio/female"))
    parser.add_argument("--output", type=Path, default=Path("manually_extracted_gender_audio_features.csv"))
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers; use 1 for lowest memory use.")
    parser.add_argument("--male-label", type=int, default=0, help="Numeric label assigned to /audio/male (default: 0).")
    parser.add_argument("--female-label", type=int, default=1, help="Numeric label assigned to /audio/female (default: 1).")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.male_label == args.female_label:
        parser.error("--male-label and --female-label must be different")

    try:
        files = [(p, args.male_label) for p in find_audio_files(args.male_dir)]
        files += [(p, args.female_label) for p in find_audio_files(args.female_dir)]
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not files:
        print("ERROR: no supported audio files found in the two input folders.", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    successes = 0
    failures: list[tuple[Path, str]] = []

    # Stream rows as they complete, but keep CSV columns in the dataset's order.
    with args.output.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        def process(item: tuple[Path, int]) -> tuple[Path, int, dict[str, float] | None, str | None]:
            path, label = item
            try:
                return path, label, extract_features(path), None
            except Exception as exc:
                return path, label, None, f"{type(exc).__name__}: {exc}"

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process, item) for item in files]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
                path, label, features, error = future.result()
                if error:
                    failures.append((path, error))
                    print(f"\nWARNING: skipped {path}: {error}", file=sys.stderr)
                    continue
                row = {column: features[column] for column in FEATURE_COLUMNS}
                row["label"] = label
                writer.writerow(row)
                successes += 1

    print(f"\nWrote {successes} rows to: {args.output.resolve()}")
    print(f"Feature columns: {len(FEATURE_COLUMNS)}; total CSV columns: {len(CSV_COLUMNS)}")
    print(f"Skipped files: {len(failures)}")
    if failures:
        print("Review warnings above for files that could not be decoded.")
    if successes == 0:
        print("ERROR: no audio files could be processed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
