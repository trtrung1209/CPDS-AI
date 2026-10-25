#!/usr/bin/env python3
"""
Calculate and compare the average Root-Mean-Square (RMS) volume of audio samples.
Useful for dataset sanity checking, volume bias detection, and research reporting.
"""

import argparse
import sys
from pathlib import Path

# Project root bootstrap
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def compute_rms(file_path: Path, sample_rate: int = 16000) -> float:
    """Load an audio file and compute its RMS amplitude."""
    import librosa
    import numpy as np
    signal, _ = librosa.load(str(file_path), sr=sample_rate)
    if len(signal) == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal**2)))


def analyze_directory(directory: Path, max_samples: int = 100, sample_rate: int = 16000) -> tuple[float, int]:
    """Calculate mean RMS across audio files in a given directory."""
    if not directory.is_dir():
        print(f"[WARN] Directory not found: {directory}", file=sys.stderr)
        return 0.0, 0

    audio_files = sorted(list(directory.rglob("*.wav")) + list(directory.rglob("*.mp3")))
    if not audio_files:
        print(f"[WARN] No audio files found in: {directory}", file=sys.stderr)
        return 0.0, 0

    import numpy as np
    selected_files = audio_files[:max_samples]
    rms_values = [compute_rms(f, sample_rate=sample_rate) for f in selected_files]
    mean_rms = float(np.mean(rms_values)) if rms_values else 0.0
    return mean_rms, len(selected_files)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute and compare RMS audio energy between classes (e.g., cry vs noise)."
    )
    parser.add_argument(
        "--cry-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "test_audio" / "cry",
        help="Path to directory containing distress/cry audio samples",
    )
    parser.add_argument(
        "--noise-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "test_audio" / "noise",
        help="Path to directory containing background noise audio samples",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Maximum number of audio files to sample per directory (default: 100)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Sampling rate to load audio at (default: 16000 Hz)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  CPDS-AI: Audio Dataset RMS Energy Analysis")
    print("=" * 60)
    print(f"Sampling limit : {args.samples} files per class")
    print(f"Target SR      : {args.sr} Hz\n")

    cry_rms, cry_count = analyze_directory(args.cry_dir, max_samples=args.samples, sample_rate=args.sr)
    noise_rms, noise_count = analyze_directory(args.noise_dir, max_samples=args.samples, sample_rate=args.sr)

    print(f"Cry class   ({cry_count:3d} samples) : Avg RMS = {cry_rms:.5f}")
    print(f"Noise class ({noise_count:3d} samples) : Avg RMS = {noise_rms:.5f}")

    if noise_rms > 0:
        import math
        ratio = cry_rms / noise_rms
        db_diff = 20 * math.log10(ratio) if ratio > 0 else 0
        print("-" * 60)
        print(f"Energy Ratio (Cry / Noise) : {ratio:.2f}x ({db_diff:+.2f} dB)")
        if abs(db_diff) > 12.0:
            print("[NOTE] Significant volume disparity detected (>12 dB). Consider normalization in training.")
        else:
            print("[OK] Energy distribution between classes is reasonably balanced.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
