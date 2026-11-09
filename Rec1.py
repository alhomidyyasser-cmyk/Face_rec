"""Display an image from this project with OpenCV."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = PROJECT_ROOT / "images" / "baby.png"


def project_path(value: str) -> Path:
    """Resolve a command-line path relative to the project directory."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open an image in an OpenCV window.")
    parser.add_argument(
        "image",
        nargs="?",
        default=DEFAULT_IMAGE,
        type=project_path,
        help="Image to display (default: images/baby.png).",
    )
    parser.add_argument("--title", default="Image viewer", help="Window title.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    cv2.imshow(args.title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
