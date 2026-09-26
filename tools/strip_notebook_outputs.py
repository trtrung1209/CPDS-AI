"""Strip execution outputs and metadata from Jupyter notebooks before committing."""

import json
import sys
from pathlib import Path


def strip_notebook(filepath: Path) -> bool:
    """Remove outputs, execution_count, and execution metadata from a notebook. Returns True if changed."""
    with filepath.open(encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    for cell in nb.get("cells", []):
        # Remove cell IDs added by Kaggle/Papermill
        if "id" in cell:
            del cell["id"]
            changed = True
        # Remove execution metadata
        for key in ("papermill", "execution", "tags"):
            if key in cell.get("metadata", {}):
                del cell["metadata"][key]
                changed = True
        # Clear code cell outputs and execution counts
        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                cell["outputs"] = []
                changed = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True

    # Remove top-level Kaggle/Papermill metadata
    for key in ("papermill", "widgets"):
        if key in nb.get("metadata", {}):
            del nb["metadata"][key]
            changed = True

    if changed:
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)

    return changed


def main():
    project_root = Path(__file__).resolve().parent.parent
    notebooks = list((project_root / "notebooks").glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found.")
        return

    for nb_path in sorted(notebooks):
        if strip_notebook(nb_path):
            print(f"Cleaned: {nb_path.name}")
        else:
            print(f"Already clean: {nb_path.name}")


if __name__ == "__main__":
    main()
