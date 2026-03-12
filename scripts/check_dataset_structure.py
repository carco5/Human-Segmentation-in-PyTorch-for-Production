from pathlib import Path

from src.utils.config import load_config


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def count_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for item in directory.iterdir() if item.is_file())


def main() -> None:
    config = load_config("configs/base.yaml")

    processed_root = Path(config["dataset"]["processed_root_dir"])

    train_images = Path(config["dataset"]["splits"]["train"]["images_dir"])
    train_masks = Path(config["dataset"]["splits"]["train"]["masks_dir"])
    val_images = Path(config["dataset"]["splits"]["val"]["images_dir"])
    val_masks = Path(config["dataset"]["splits"]["val"]["masks_dir"])

    expected_directories = [
        processed_root,
        train_images,
        train_masks,
        val_images,
        val_masks,
    ]

    print("=" * 60)
    print("Dataset Structure Check")
    print("=" * 60)

    for directory in expected_directories:
        existed_before = directory.exists()
        ensure_directory(directory)

        if existed_before:
            print(f"[OK] Directory already exists: {directory}")
        else:
            print(f"[CREATED] Directory created: {directory}")

    print("\nFile counts:")
    print(f"- train/images: {count_files(train_images)}")
    print(f"- train/masks:  {count_files(train_masks)}")
    print(f"- val/images:   {count_files(val_images)}")
    print(f"- val/masks:    {count_files(val_masks)}")

    print("\nExpected convention:")
    print("- Each image must have a corresponding mask with the same file stem.")
    print("- Example: sample_001.jpg <-> sample_001.png")
    print("- Masks must be binary or convertible to binary.")

    print("\nStatus: dataset directory scaffold is ready.")


if __name__ == "__main__":
    main()