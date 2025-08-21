import os
import cv2
import torch
import tempfile
import numpy as np
import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
from ultralytics import YOLO
from typing import List


def extract_video_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    success, frame = cap.read()
    while success:
        frames.append(frame)
        success, frame = cap.read()
    cap.release()
    return frames, fps


def run_inference_on_frames(model: YOLO, frames: List[np.ndarray], class_order: List[str], conf_thres: float, device: str):
    results_per_frame = []
    names = model.names
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = model.predict(rgb, verbose=False, device=device, conf=conf_thres)[0]
        frame_dets = []
        if res and res.boxes is not None:
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), conf, cid in zip(boxes_xyxy, confs, clss):
                label = names.get(cid, str(cid)) if isinstance(names, dict) else names[cid]
                if class_order:
                    try:
                        label_index = class_order.index(label)
                    except ValueError:
                        label_index = -1
                else:
                    label_index = -1
                frame_dets.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "label": label,
                    "label_index": label_index,
                })
        results_per_frame.append(frame_dets)
    return results_per_frame


def draw_boxes(frames: List[np.ndarray], results_per_frame):
    drawn = []
    for frame, dets in zip(frames, results_per_frame):
        f = frame.copy()
        for det in dets:
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = det["label"]
            conf = det["confidence"]
            cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(f, f"{label}:{conf:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        drawn.append(f)
    return drawn


def write_video(frames: List[np.ndarray], fps: float, out_path: str):
    if not frames:
        raise ValueError("No frames to write")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return out_path

class VideoDetection(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="video_detection",
            label="Video Detection",
            description="Run YOLO11 detection on a video and show annotated result (GPU if available)",
            icon="/assets/one.png"
        )

    def resolve_input(self, ctx):
        """Define the operator's input form.

        Notes:
        - Keys should be simple (no spaces); these become ctx.params keys
        - Use appropriate field helpers instead of enum() for free-form inputs
        - File inputs use file() with accept extensions
        """

        inputs = types.Object()

        # Video file path
        inputs.file(
            "video_path",
            label="Video File",
            description="Path to the input video (.mp4/.avi/.mov)",
            required=True,
            accept=[".mp4", ".avi", ".mov"],
        )

        # YOLO model weights (.pt)
        inputs.file(
            "model_path",
            label="YOLO Model (.pt)",
            description="Path to YOLO *.pt weights (YOLOv8/YOLO11)",
            required=True,
            accept=[".pt"],
        )

        # Device selection (auto picks cuda if available)
        device_values = ["auto", "cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        inputs.enum(
            "device",
            values=device_values,
            label="Device",
            description="Computation device",
            default="auto",
        )

        # Classes filter (optional)
        inputs.string(
            "classes",
            label="Classes",
            description="Optional comma-separated list of class names to keep (leave blank for all)",
            required=False,
            default="",
        )

        # Confidence threshold
        inputs.float(
            "confidence",
            label="Confidence Threshold",
            description="Detection confidence threshold (0-1)",
            required=False,
            default=0.25,
            min=0.0,
            max=1.0,
        )

        return types.Property(inputs=inputs)

    def execute(self, ctx):
        # Retrieve parameters using the keys defined above
        video_path = ctx.params.get("video_path")
        model_path = ctx.params.get("model_path")
        device_choice = ctx.params.get("device", "auto")
        classes_str = ctx.params.get("classes", "") or ""
        conf_thres = ctx.params.get("confidence", 0.25)

        # Normalize types
        try:
            conf_thres = float(conf_thres)
        except Exception:
            conf_thres = 0.25

        class_order = [c.strip() for c in classes_str.split(",") if c.strip()]

        if device_choice == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_choice

        frames, fps = extract_video_frames(video_path)
        model = YOLO(model_path)
        results = run_inference_on_frames(model, frames, class_order, conf_thres, device)
        drawn = draw_boxes(frames, results)

        tmpdir = tempfile.mkdtemp(prefix="fo_vid_")
        out_path = os.path.join(tmpdir, "annotated.mp4")
        write_video(drawn, fps, out_path)

        # Return outputs; FiftyOne will display or pass to panels
        return {"output_video": out_path, "fps": fps, "frame_count": len(frames), "device": device}

class VideoPlayerPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="VideoPlayerPanel",
            label="Video Inference Tool",
            description="Displays processed video",
        )

    def render(self, ctx):
        # Expect the operator to have been run and stored path in session state
        output_video = (
            getattr(ctx.session, "state", {}).get("output_video")
            if hasattr(ctx.session, "state") else None
        ) or ctx.params.get("output_video")
        if not output_video or not os.path.exists(output_video):
            return foo.PanelHTML(
                "<p>No video available. Run Video Detection operator.</p>"
            )
        # Simple HTML5 video tag
        rel_path = output_video.replace("\\", "/")
        html = f"""
        <video width='100%' controls>
            <source src='file://{rel_path}' type='video/mp4'>
            Your browser does not support the video tag.
        </video>
        """
        return foo.PanelHTML(html)
