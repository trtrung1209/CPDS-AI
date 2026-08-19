import os
from pathlib import Path

def get_next_run_dir(base_dir="runs"):
    """
    Finds the next available run directory (e.g., runs/run1, runs/run2).
    Creates the directory and returns its Path.
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    run_id = 1
    while True:
        run_dir = base_path / f"run{run_id}"
        if not run_dir.exists():
            run_dir.mkdir(parents=True)
            return run_dir
        run_id += 1

def save_result(run_dir, filename, content):
    """
    Saves a result file into the specified run directory.
    """
    file_path = run_dir / filename
    with open(file_path, "w") as f:
        f.write(content)
    return file_path
