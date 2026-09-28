"""Create a small, deterministic evaluation set from the public training sources."""

import argparse
import csv
import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf


CRY_REPOSITORY = "https://github.com/gveres/donateacry-corpus.git"
ESC50_REPOSITORY = "https://github.com/karolpiczak/ESC-50.git"
CRY_EXTENSIONS = {".wav", ".caf", ".mp3", ".flac", ".ogg", ".3gp", ".m4a"}
NOISE_CATEGORIES = {
    "engine", "car_horn", "siren", "rain", "wind", "door_wood_knock",
    "dog", "cat", "laughing", "breathing", "coughing", "sneezing", "snoring",
    "footsteps", "chirping_birds", "thunderstorm",
}
NOISE_WINDOW_SECONDS = 2.0

# The eval holdout is separated entirely from the train pool to ensure zero data leakage.
EVAL_HOLDOUT_PER_CLASS = 60


def split_holdout_and_train_pool(population: list, seed: int, holdout_size: int) -> tuple:
    """Shuffle deterministically, then split into (eval_holdout, train_pool) -- disjoint by construction.
    This algorithm must exactly match the one in 01_audio_training.ipynb to guarantee disjoint sets.
    """
    shuffled = list(population)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:holdout_size], shuffled[holdout_size:]


def clone_if_missing(repository: str, destination: Path) -> None:
    """Create a shallow local clone only when the requested cache is absent."""
    if destination.is_dir():
        return
    subprocess.run(["git", "clone", "--depth", "1", repository, str(destination)], check=True)


def convert_to_wav(source: Path, destination: Path) -> bool:
    """Convert one source clip to the evaluation format used by inference."""
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(source), "-ac", "1", "-ar", "16000", str(destination)],
            check=True,
        )
        return destination.is_file()
    except subprocess.CalledProcessError:
        return False


def select_loudest_window(y: np.ndarray, sr: int, duration: float = 2.0, hop_fraction: float = 0.1) -> np.ndarray:
    """Return the `duration`-second window with highest RMS energy (pad short clips instead)."""
    target_len = int(sr * duration)
    if len(y) <= target_len:
        return np.pad(y, (0, target_len - len(y)))
    hop = max(1, int(target_len * hop_fraction))
    best_start, best_energy = 0, -1.0
    for start in range(0, len(y) - target_len + 1, hop):
        segment = y[start:start + target_len]
        energy = float(np.sum(segment.astype(np.float64) ** 2))
        if energy > best_energy:
            best_energy, best_start = energy, start
    return y[best_start:best_start + target_len]


def convert_noise_to_wav(source: Path, destination: Path, sr: int = 16000, duration: float = NOISE_WINDOW_SECONDS) -> None:
    """Load an ESC-50 clip, keep only its loudest window, save as mono WAV."""
    y, loaded_sr = librosa.load(str(source), sr=sr, duration=None)
    trimmed = select_loudest_window(y, loaded_sr, duration=duration)
    sf.write(str(destination), trimmed, loaded_sr)


def prepare_audio_evaluation_data(output_dir: Path, cache_dir: Path, samples_per_class: int, seed: int, overwrite: bool) -> dict:
    """Build balanced WAV evaluation data and return its reproducibility manifest."""
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive.")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite to replace it.")
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    cry_cache, esc50_cache = cache_dir / "donateacry-corpus", cache_dir / "ESC-50"
    cache_dir.mkdir(parents=True, exist_ok=True)
    clone_if_missing(CRY_REPOSITORY, cry_cache)
    clone_if_missing(ESC50_REPOSITORY, esc50_cache)

    cry_source = sorted(path for path in cry_cache.rglob("*") if path.suffix.lower() in CRY_EXTENSIONS)
    if not cry_source:
        raise ValueError("Donate-a-cry contains no supported audio files.")
    cry_holdout, _cry_train_pool = split_holdout_and_train_pool(cry_source, seed, EVAL_HOLDOUT_PER_CLASS)
    selected_cry = cry_holdout[:min(samples_per_class, len(cry_holdout))]

    metadata_path, audio_dir = esc50_cache / "meta" / "esc50.csv", esc50_cache / "audio"
    with metadata_path.open(encoding="utf-8", newline="") as metadata_file:
        rows = list(csv.DictReader(metadata_file))
    noise_source = [audio_dir / row["filename"] for row in rows if row["category"] in NOISE_CATEGORIES]
    noise_holdout, _noise_train_pool = split_holdout_and_train_pool(noise_source, seed, EVAL_HOLDOUT_PER_CLASS)
    selected_noise = noise_holdout[:min(len(selected_cry), len(noise_holdout))]

    cry_output, noise_output = output_dir / "cry", output_dir / "noise"
    cry_output.mkdir(parents=True, exist_ok=True)
    noise_output.mkdir(parents=True, exist_ok=True)
    
    valid_cries = 0
    for source in selected_cry:
        if convert_to_wav(source, cry_output / f"cry_{valid_cries:03d}.wav"):
            valid_cries += 1

    valid_noises = 0
    for source in selected_noise:
        try:
            convert_noise_to_wav(source, noise_output / f"noise_{valid_noises:03d}.wav")
            valid_noises += 1
        except Exception:
            pass

    manifest = {
        "seed": seed,
        "cry_count": valid_cries,
        "noise_count": valid_noises,
        "sources": [CRY_REPOSITORY, ESC50_REPOSITORY],
        "noise_categories": sorted(NOISE_CATEGORIES),
        "noise_prep": "loudest-2s-RMS-window",
        "eval_holdout_per_class": EVAL_HOLDOUT_PER_CLASS,
        "split_method": "deterministic-shuffle-holdout-prefix",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic public audio data for model evaluation.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/test_audio"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/cpds-ai-audio"))
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = prepare_audio_evaluation_data(args.output_dir, args.cache_dir, args.samples_per_class, args.seed, args.overwrite)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
