import json
import os
from pathlib import Path


BASE_DIR = Path(os.getenv("BASE_COE", "."))
TARGET_DIRS = [
    BASE_DIR / "output" / "probe" / "sandbox",
    BASE_DIR / "output" / "baseline" / "sandbox",
]


def should_delete(path: Path, is_probe_dir: bool) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    args = obj.get("args")
    if not isinstance(args, dict):
        return True
    if is_probe_dir and args.get("mode") == "meta_attn":
        return True
    return "target_dataset" not in args


def cleanup_missing_target_dataset() -> None:
    deleted = 0
    for directory in TARGET_DIRS:
        if not directory.exists():
            continue
        is_probe_dir = directory == (BASE_DIR / "output" / "probe" / "sandbox")
        for path in directory.glob("*.json"):
            if should_delete(path, is_probe_dir=is_probe_dir):
                path.unlink(missing_ok=True)
                deleted += 1
    print(f"Deleted {deleted} files.")


if __name__ == "__main__":
    cleanup_missing_target_dataset()
