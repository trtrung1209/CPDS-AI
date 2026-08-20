"""Create a small, deterministic evaluation set from the public training sources."""

import argparse
import csv
import json
import random
import shutil
import subprocess
from pathlib import Path


CRY_REPOSITORY = "https://github.com/gveres/donateacry-corpus.git"
ESC50_REPOSITORY = "https://github.com/karolpiczak/ESC-50.git"
CRY_EXTENSIONS = {".wav", ".caf", ".mp3", ".flac", ".ogg", ".3gp", ".m4a"}
NOISE_CATEGORIES = {"engine", "car_horn", "siren", "rain", "wind", "airplane", "helicopter", "train"}


def clone_if_missing(repository: str, destination: Path) -> None:
    """Create a shallow local clone only when the requested cache is absent."""
    if destination.is_dir():
        return
    subprocess.run(["git", "clone", "--depth", "1", repository, str(destination)], check=True)


def convert_to_wav(source: Path, destination: Path) -> None:
    """Convert one source clip to the evaluation format used by inference."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(source), "-ac", "1", "-ar", "16000", str(destination)],
        check=True,
    )


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

    random_generator = random.Random(seed)
    cry_source = sorted(path for path in cry_cache.rglob("*") if path.suffix.lower() in CRY_EXTENSIONS)
    if not cry_source:
        raise ValueError("Donate-a-cry contains no supported audio files.")
    selected_cry = random_generator.sample(cry_source, min(samples_per_class, len(cry_source)))

    metadata_path, audio_dir = esc50_cache / "meta" / "esc50.csv", esc50_cache / "audio"
    with metadata_path.open(encoding="utf-8", newline="") as metadata_file:
        rows = list(csv.DictReader(metadata_file))
    noise_source = [audio_dir / row["filename"] for row in rows if row["category"] in NOISE_CATEGORIES]
    selected_noise = random_generator.sample(noise_source, min(len(selected_cry), len(noise_source)))

    cry_output, noise_output = output_dir / "cry", output_dir / "noise"
    cry_output.mkdir(parents=True, exist_ok=True)
    noise_output.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(selected_cry):
        convert_to_wav(source, cry_output / f"cry_{index:03d}.wav")
    for index, source in enumerate(selected_noise):
        shutil.copy2(source, noise_output / f"noise_{index:03d}.wav")

    manifest = {"seed": seed, "cry_count": len(selected_cry), "noise_count": len(selected_noise), "sources": [CRY_REPOSITORY, ESC50_REPOSITORY]}
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
