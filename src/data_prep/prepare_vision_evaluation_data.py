"""Prepare a deterministic vision evaluation dataset from Roboflow."""

import argparse
import json
import os
import shutil
from pathlib import Path


def prepare_vision_evaluation_data(output_dir: Path, workspace: str = "timii-owolabi-pwfjm", project: str = "child-adult-detection-bgjzk", version: int = 10) -> dict:
    """Download the Roboflow dataset and extract ONLY the held-out test split."""
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY environment variable is required to download the vision dataset. Please set it and try again.")
    
    # Lazy import to avoid ultralytics/roboflow overhead if not used
    from roboflow import Roboflow
    
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ds_version = proj.version(version)
    dataset = ds_version.download("yolov8")
    
    source_test = Path(dataset.location) / "test"
    if not source_test.is_dir():
        raise FileNotFoundError(f"Roboflow dataset does not contain a 'test' split at {source_test}.")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    if images_dir.exists():
        shutil.rmtree(images_dir)
    if labels_dir.exists():
        shutil.rmtree(labels_dir)
        
    shutil.copytree(source_test / "images", images_dir)
    shutil.copytree(source_test / "labels", labels_dir)
    
    # Clean up the huge downloaded dataset (Train, Valid, Test) because we only need the test split!
    if Path(dataset.location).exists():
        shutil.rmtree(dataset.location)
    
    yaml_content = f"""path: {output_dir.absolute()}
train: images
val: images
test: images
nc: 2
names: ['adult', 'child']
"""
    (output_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")
    
    images_count = len(list(images_dir.glob("*.jpg")))
    manifest = {
        "source": f"roboflow:{workspace}/{project}/{version}/test",
        "images_count": images_count,
        "split": "test",
        "description": "Held-out test set from Roboflow. Disjoint from train set by Roboflow split.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare vision evaluation data.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/test_vision"))
    args = parser.parse_args()
    try:
        manifest = prepare_vision_evaluation_data(args.output_dir)
        print(json.dumps(manifest, indent=2))
    except (RuntimeError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
