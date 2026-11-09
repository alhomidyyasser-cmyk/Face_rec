"""Run COCO object detection on an image using OpenCV's DNN module."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = PROJECT_ROOT / "images" / "baby.png"
MODEL_PATH = PROJECT_ROOT / "files" / "frozen_inference_graph.pb"
CONFIG_PATH = PROJECT_ROOT / "files" / "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
LABELS_PATH = PROJECT_ROOT / "files" / "thing.names"


def project_path(value: str) -> Path:
    """Resolve a command-line path relative to the project directory."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def confidence(value: str) -> float:
    """Validate the detection confidence threshold supplied to argparse."""
    threshold = float(value)
    if not 0.0 < threshold <= 1.0:
        raise argparse.ArgumentTypeError("confidence must be greater than 0 and at most 1")
    return threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect COCO objects in an image.")
    parser.add_argument(
        "image",
        nargs="?",
        default=DEFAULT_IMAGE,
        type=project_path,
        help="Image to analyze (default: images/baby.png).",
    )
    parser.add_argument(
        "--confidence",
        type=confidence,
        default=0.6,
        help="Minimum confidence for a detection, from 0 to 1 (default: 0.6).",
    )
    parser.add_argument(
        "--output",
        type=project_path,
        help="Optional path for the annotated image.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open an OpenCV window (useful on headless systems).",
    )
    return parser.parse_args()


def load_labels(labels_path: Path) -> List[str]:
    if not labels_path.is_file():
        raise FileNotFoundError(f"Could not find labels file: {labels_path}")

    labels = [label.strip() for label in labels_path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise ValueError(f"The labels file is empty: {labels_path}")
    return labels


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Could not find {description}: {path}")


def annotate(image, model: cv2.dnn_DetectionModel, labels: List[str], threshold: float) -> int:
    class_ids, confidences, boxes = model.detect(image, confThreshold=threshold)
    detections = 0

    for class_id, confidence_score, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
        class_id = int(class_id)
        label = labels[class_id - 1] if 0 < class_id <= len(labels) else f"class {class_id}"
        x, y, width, height = (int(value) for value in box)

        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{label}: {float(confidence_score):.0%}",
            (x + 10, max(y + 30, 30)),
            cv2.FONT_HERSHEY_COMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        detections += 1

    return detections


def save_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save annotated image to: {path}")


def main() -> int:
    args = parse_args()
    require_file(args.image, "input image")
    require_file(MODEL_PATH, "model file")
    require_file(CONFIG_PATH, "model configuration")

    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"Could not decode image: {args.image}")

    labels = load_labels(LABELS_PATH)
    model = cv2.dnn_DetectionModel(str(MODEL_PATH), str(CONFIG_PATH))
    model.setInputSize(320, 320)
    model.setInputScale(1.0 / 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)

    detections = annotate(image, model, labels, args.confidence)
    print(f"Found {detections} detection(s) at confidence >= {args.confidence:.0%}.")

    if args.output:
        save_image(args.output, image)
        print(f"Saved annotated image to {args.output}")

    if not args.no_display:
        cv2.imshow("Object detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


