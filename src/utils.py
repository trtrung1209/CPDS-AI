from pathlib import Path
from tempfile import NamedTemporaryFile

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
    Atomically save a UTF-8 result file into the specified run directory.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    file_path = run_dir / filename
    with NamedTemporaryFile("w", encoding="utf-8", dir=run_dir, delete=False) as temporary_file:
        temporary_file.write(content)
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(file_path)
    return file_path
