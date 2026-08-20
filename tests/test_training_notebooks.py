import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def notebook_source(filename):
    notebook = json.loads((PROJECT_ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_vision_notebook_does_not_embed_a_roboflow_api_key():
    source = notebook_source("02_vision_training.ipynb")

    assert "ROBOFLOW_API_KEY" in source
    assert "UserSecretsClient" in source
    assert "api_key=\"" not in source


def test_vision_export_uses_training_result_not_a_hard_coded_runs_path():
    source = notebook_source("02_vision_training.ipynb")

    assert "results.save_dir" in source
    assert "best_weights = run_dir / \"weights\" / \"best.pt\"" in source
    assert "artifacts_dir / \"yolov8n-adult-child.onnx\"" in source
    assert "onnx.checker.check_model" in source


def test_audio_notebook_downloads_esc50_and_excludes_crying_baby():
    source = notebook_source("01_audio_training.ipynb")

    assert "esc50.csv" in source
    assert "crying_baby" in source
    assert "VEHICLE_NOISE_CATEGORIES" in source
    assert "random.seed(SEED)" in source
    assert "dataset_manifest.json" in source
    assert 'git clone --depth 1 https://github.com/karolpiczak/ESC-50.git' in source
    assert "convert_to_wav" in source
    assert 'AUDIO_EXTENSIONS = {".wav"}' in source
    assert 'dynamo=False' in source
    assert 'map_location="cpu"' in source
    assert 'model = model.cpu().eval()' in source


def test_notebooks_are_valid_json_and_contain_no_executed_outputs():
    for filename in ("01_audio_training.ipynb", "02_vision_training.ipynb"):
        notebook = json.loads((PROJECT_ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
