# OpenCV Object Detection Demo

A small Python project that runs object detection on still images using OpenCV's DNN module and a bundled SSD MobileNet v3 model trained on the COCO dataset.

> **Note:** The repository name mentions face recognition, but the included model performs general object detection. It can identify COCO categories such as people, cars, dogs, bicycles, and many common objects; it is not a face-recognition system.

## Features

- Detects COCO objects in PNG, JPEG, and other image formats supported by OpenCV.
- Draws bounding boxes, labels, and confidence scores on detected objects.
- Uses project-relative paths, so scripts work when launched from another directory.
- Can display results interactively or write an annotated image for headless environments.
- Includes a minimal image viewer for quickly checking sample images.

## Project layout

```text
.
├── Rec1.py                         # OpenCV image viewer
├── Rec2.py                         # Object-detection command-line program
├── requirements.txt                # Python dependency list
├── files/
│   ├── frozen_inference_graph.pb    # TensorFlow SSD MobileNet v3 weights
│   ├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt  # OpenCV model config
│   └── thing.names                 # COCO class labels
└── images/                         # Sample input images
```

## Requirements

- Python 3.8 or later
- `pip`
- A desktop environment for the default OpenCV display window. Use `--no-display` on servers or CI systems.

## Installation

Create and activate a virtual environment, then install the dependency:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage

Run object detection on the included sample image:

```powershell
py Rec2.py
```

Choose an image and set a confidence threshold:

```powershell
py Rec2.py images/car.png --confidence 0.65
```

Save an annotated result without opening a GUI window:

```powershell
py Rec2.py images/baby.png --output output/baby-detections.png --no-display
```

Use the image viewer:

```powershell
py Rec1.py images/LL3.png
```

Use `py Rec2.py --help` or `py Rec1.py --help` to see all available options.

## Model and labels

`Rec2.py` loads the TensorFlow frozen graph and OpenCV `.pbtxt` configuration from `files/`. The model uses the class names in `files/thing.names`. Keep these three files together when moving or packaging the project.

## Troubleshooting

- **`py` reports “No installed Python found” or `python` is not recognized:** Install Python from [python.org](https://www.python.org/downloads/) and enable the option to add Python to `PATH`, then reopen your terminal.
- **No display available:** Add `--no-display --output output/result.png` to the `Rec2.py` command.
- **Input image cannot be read:** Confirm the image path is correct. Relative paths are interpreted from the project directory.
- **Model file not found:** Restore the three required files in the `files/` directory.

## License

No license file is currently included. Add one before redistributing or accepting external contributions.
