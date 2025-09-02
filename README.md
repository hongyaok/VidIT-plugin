# Video Inference Tool (VidIT)

A FiftyOne plugin for running YOLO object detection on videos and visualizing the results.

## Features
- Upload a video and a YOLO model (.pt)
- Specify detection classes and confidence threshold
- Annotate video frames with detected objects
- View the annotated video in a custom FiftyOne panel

## Requirements
- [FiftyOne](https://voxel51.com/docs/fiftyone/)
- [ultralytics](https://github.com/ultralytics/ultralytics) (YOLO)
- OpenCV (`opencv-python`)
- torch

## Installation
1. Clone or copy this plugin folder into your FiftyOne plugins directory, or load it via the FiftyOne App UI.
2. Install dependencies:
   ```sh
   pip install fiftyone ultralytics opencv-python torch
   ```

## Usage
1. Open FiftyOne App.
2. Go to the Operators menu and select **Video Inference Tool**.
3. Upload a video and YOLO model, set classes and confidence, then run.
4. After processing, click **Open Video Panel** to view the annotated video.

## File Structure
- `video_detection.py` — main operator and panel code
- `__init__.py` — plugin registration
- `fiftyone.yml` — plugin manifest
- `assets/` — static assets (optional)
