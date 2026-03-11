from importlib.metadata import version, PackageNotFoundError
import sys

try:
    import torch
except ImportError as exc:
    raise SystemExit(f"PyTorch is not installed correctly: {exc}")


PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "matplotlib",
    "pillow",
    "opencv-python",
    "PyYAML",
    "tqdm",
    "scikit-learn",
    "fastapi",
    "httpx",
    "pytest",
    "ruff",
    "black",
]


def get_package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> None:
    print("=" * 60)
    print("Human Segmentation in PyTorch for Production")
    print("Environment Check")
    print("=" * 60)

    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    print("\nInstalled package versions:")
    for package in PACKAGES:
        print(f"- {package}: {get_package_version(package)}")


if __name__ == "__main__":
    main()