from pathlib import Path

from src.utils.config import load_config


def status_line(path: Path) -> str:
    return "[OK]" if path.exists() else "[MISSING]"


def main() -> None:
    config = load_config("configs/base.yaml")

    raw_root = Path(config["dataset"]["raw_root_dir"])

    expected_paths = [
        raw_root,
        Path(config["dataset"]["raw_splits"]["train"]["images_dir"]),
        Path(config["dataset"]["raw_splits"]["train"]["category_masks_dir"]),
        Path(config["dataset"]["raw_splits"]["train"]["id_file"]),
        Path(config["dataset"]["raw_splits"]["val"]["images_dir"]),
        Path(config["dataset"]["raw_splits"]["val"]["category_masks_dir"]),
        Path(config["dataset"]["raw_splits"]["val"]["id_file"]),
    ]

    print("=" * 60)
    print("CIHP Raw Structure Check")
    print("=" * 60)

    all_ok = True
    for path in expected_paths:
        exists = path.exists()
        all_ok = all_ok and exists
        print(f"{status_line(path)} {path}")

    print("\nExpected layout:")
    print("- data/raw/CIHP/Training/Images")
    print("- data/raw/CIHP/Training/Category_ids")
    print("- data/raw/CIHP/Training/train_id.txt")
    print("- data/raw/CIHP/Validation/Images")
    print("- data/raw/CIHP/Validation/Category_ids")
    print("- data/raw/CIHP/Validation/val_id.txt")

    if all_ok:
        print("\nStatus: raw CIHP structure looks correct.")
    else:
        print("\nStatus: raw CIHP structure is incomplete.")
        print("Download and unzip the official CIHP dataset into data/raw/CIHP.")
    

if __name__ == "__main__":
    main()