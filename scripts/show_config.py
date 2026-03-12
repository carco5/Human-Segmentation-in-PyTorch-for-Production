from pprint import pprint

from src.utils.config import load_config


def main() -> None:
    config = load_config("configs/base.yaml")

    print("=" * 60)
    print("Loaded project configuration")
    print("=" * 60)
    pprint(config)

    print("\nQuick summary:")
    print(f"- Project name: {config['project']['name']}")
    print(f"- Task: {config['project']['task']}")
    print(f"- Image size: {config['data']['image_size']}")
    print(f"- Batch size: {config['training']['batch_size']}")
    print(f"- Device: {config['training']['device']}")


if __name__ == "__main__":
    main()