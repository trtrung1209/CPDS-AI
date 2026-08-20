from pathlib import Path
from src.utils import get_next_run_dir, save_result

def test_get_next_run_dir(tmp_path):
    # tmp_path is provided by pytest to create temporary directories
    run_dir1 = get_next_run_dir(base_dir=tmp_path)
    assert run_dir1.name == "run1"
    assert run_dir1.exists()

    run_dir2 = get_next_run_dir(base_dir=tmp_path)
    assert run_dir2.name == "run2"
    assert run_dir2.exists()

def test_save_result(tmp_path):
    run_dir = get_next_run_dir(base_dir=tmp_path)
    file_path = save_result(run_dir, "output.txt", "Child Detected: True")
    
    assert file_path.exists()
    with open(file_path, "r") as f:
        content = f.read()
    assert content == "Child Detected: True"
    assert list(run_dir.glob("tmp*")) == []
