"""Evaluate the fused CPDS-AI pipeline across a synthesized dataset of edge cases."""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

from src.inference.run_inference import run as run_dual


def evaluate_fused_model(vision_model: Path, audio_model: Path, vision_test_dir: Path, audio_test_dir: Path) -> Dict[str, Any]:
    """Dynamically construct dual-modal test cases and evaluate the full pipeline."""
    random.seed(42)
    
    # 1. Parse vision labels to separate Child vs Adult images
    vision_labels = list((vision_test_dir / "labels").glob("*.txt"))
    child_images = []
    adult_images = []
    
    for lbl in vision_labels:
        content = lbl.read_text(encoding="utf-8")
        # Class 1 is Child, 0 is Adult
        has_child = any(line.startswith("1 ") for line in content.strip().splitlines())
        has_adult = any(line.startswith("0 ") for line in content.strip().splitlines())
        img_path = vision_test_dir / "images" / f"{lbl.stem}.jpg"
        
        if img_path.is_file():
            if has_child:
                child_images.append(img_path)
            elif has_adult:
                adult_images.append(img_path)
                
    # 2. Gather audio files
    cry_audios = list((audio_test_dir / "cry").glob("*.wav"))
    noise_audios = list((audio_test_dir / "noise").glob("*.wav"))
    
    # 3. Build Scenarios
    scenarios = []
    def add_scenarios(img_list: list, aud_list: list, expected: bool, name: str, count: int = 25) -> None:
        if not img_list or not aud_list:
            return
        # Ensure we always take a deterministic subset
        shuffled_imgs = list(img_list)
        shuffled_auds = list(aud_list)
        random.shuffle(shuffled_imgs)
        random.shuffle(shuffled_auds)
        
        for i in range(min(count, len(shuffled_imgs), len(shuffled_auds))):
            scenarios.append({
                "image": str(shuffled_imgs[i]),
                "audio": str(shuffled_auds[i]),
                "expected_alarm": expected,
                "scenario_name": name
            })
            
    add_scenarios(child_images, cry_audios, True, "Child + Crying")
    add_scenarios(child_images, noise_audios, False, "Child + Silent/Noise")
    add_scenarios(adult_images, cry_audios, False, "Adult + Crying (Other source)")
    add_scenarios(adult_images, noise_audios, False, "Adult + Noise")
    
    failures = []
    correct = 0
    
    print(f"[INFO] Running Fused Inference Evaluation on {len(scenarios)} generated pairs...")
    
    for s in scenarios:
        res = run_dual(
            image_path=s["image"],
            audio_path=s["audio"],
            vision_model=str(vision_model),
            audio_model=str(audio_model)
        )
        
        triggered = res["alarm_triggered"]
        if triggered == s["expected_alarm"]:
            correct += 1
        else:
            failures.append({
                "scenario": s["scenario_name"],
                "image": Path(s["image"]).name,
                "audio": Path(s["audio"]).name,
                "expected_alarm": s["expected_alarm"],
                "actual_alarm": triggered,
                "vision_child_conf": round(res["vision"]["confidence"] if res["vision"]["child_detected"] else 0.0, 3),
                "audio_cry_conf": round(res["audio"]["confidence"] if res["audio"]["is_crying"] else 0.0, 3),
            })
            
    total_expected_alarms = sum(1 for s in scenarios if s["expected_alarm"])
    total_expected_silence = sum(1 for s in scenarios if not s["expected_alarm"])
    
    fnr = sum(1 for f in failures if f["expected_alarm"] and not f["actual_alarm"]) / max(1, total_expected_alarms)
    fpr = sum(1 for f in failures if not f["expected_alarm"] and f["actual_alarm"]) / max(1, total_expected_silence)
    
    report = {
        "total_scenarios": len(scenarios),
        "correct": correct,
        "false_negative_rate": float(fnr),
        "false_positive_rate": float(fpr),
        "failures": failures
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fused model pipeline for CPDS-AI.")
    parser.add_argument("--vision-model", type=Path, required=True)
    parser.add_argument("--audio-model", type=Path, required=True)
    parser.add_argument("--vision-test-dir", type=Path, required=True)
    parser.add_argument("--audio-test-dir", type=Path, required=True)
    args = parser.parse_args()
    
    try:
        report = evaluate_fused_model(args.vision_model, args.audio_model, args.vision_test_dir, args.audio_test_dir)
        print(json.dumps(report, indent=2))
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
