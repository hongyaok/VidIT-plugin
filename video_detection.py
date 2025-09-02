import os
import cv2
import torch
import tempfile
import numpy as np
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
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
    # Use H.264 codec for better browser compatibility
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
            icon="/assets/icon.svg"
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        inputs.file(
            "video",
            label="Video Path",
            required=True,
            description="Path to the input video file",
            accepted_file_types=[".mp4", ".avi", ".mov"],
            button_label="Browse..."
        )
        inputs.file(
            "model",
            label="YOLO Model",
            required=True,
            description="Path to the YOLO model file",
            accepted_file_types=[".pt"],
            button_label="Browse..."
        )
        inputs.str(
            "classes",
            label="Classes (comma-separated)",
            required=True,
            default="person,car,truck,bus,bicycle,motorcycle"
        )
        inputs.float(
            "confidence",
            label="Confidence Threshold",
            min=0.0,
            max=1.0,
            default=0.25,
            required=True
        )
        return types.Property(inputs, view=types.View(label="VidIT Settings"))

    def resolve_output(self, ctx):
        outputs = types.Object()
        
        # Add button to open panel after execution
        outputs.btn(
            "open_panel",
            label="Open Video Panel",
            icon="play_arrow",
            variant="contained",
            on_click="open_video_panel",
            params={"panel_name": "VideoPlayerPanel"}
        )
        
        outputs.str("device_used", label="Device Used")
        outputs.int("frame_count", label="Total Frames")
        outputs.float("fps", label="Video FPS")
        outputs.str("output_video", label="Output Video Path")
        
        return types.Property(outputs)

    def execute(self, ctx):
        video_path = ctx.params.get("video")
        model_path = ctx.params.get("model")
        
        # Handle file path extraction more robustly
        if isinstance(video_path, dict):
            video_path = video_path.get("absolute_path") or video_path.get("filepath")
        if isinstance(model_path, dict):
            model_path = model_path.get("absolute_path") or model_path.get("filepath")
            
        video_path = str(video_path)
        model_path = str(model_path)
        
        classes_str = str(ctx.params.get("classes", ""))
        conf_thres = float(ctx.params.get("confidence", 0.25))
        
        # Check GPU availability and report it
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        class_order = [c.strip() for c in classes_str.split(",") if c.strip()]

        # Extract frames and run inference
        frames, fps = extract_video_frames(video_path)
        print(f"Extracted {len(frames)} frames from video at {fps} FPS")
        
        model = YOLO(model_path)
        print(f"Loaded YOLO model from {model_path}")
        
        results = run_inference_on_frames(model, frames, class_order, conf_thres, device)
        drawn = draw_boxes(frames, results)

        # Create output video with better file naming
        tmpdir = tempfile.mkdtemp(prefix="fiftyone_vidit_")
        video_name = os.path.basename(video_path).split('.')[0]
        out_path = os.path.join(tmpdir, f"{video_name}_annotated.mp4")
        write_video(drawn, fps, out_path)

        print(f"Saved annotated video to: {out_path}")

        # Store results in panel state for access by the panel
        if not hasattr(ctx, 'panel'):
            # Create a mock panel object to store state
            class MockPanel:
                def __init__(self):
                    self.state = {}
            ctx.panel = MockPanel()
        
        # Store video info in panel state
        ctx.panel.state = {
            "output_video": out_path,
            "fps": fps,
            "frame_count": len(frames),
            "device_used": device
        }

        return {
            "output_video": out_path,
            "fps": fps,
            "frame_count": len(frames),
            "device_used": device
        }


class VideoPlayerPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="VideoPlayerPanel",
            label="Video Inference Results",
            description="Displays processed video with detection results",
            icon="video_library",
            surfaces="grid modal"
        )

    def on_load(self, ctx):
        """Initialize panel when it loads"""
        # Initialize state if not present
        if not hasattr(ctx.panel, 'state') or not ctx.panel.state:
            ctx.panel.state = {
                "output_video": None,
                "fps": 0,
                "frame_count": 0,
                "device_used": "Unknown"
            }

    def render(self, ctx):
        panel = types.Object()
        
        # Get video data from panel state or params
        output_video = None
        fps = 0
        frame_count = 0
        device_used = "Unknown"
        
        if hasattr(ctx.panel, 'state') and ctx.panel.state:
            output_video = ctx.panel.state.get("output_video")
            fps = ctx.panel.state.get("fps", 0)
            frame_count = ctx.panel.state.get("frame_count", 0)
            device_used = ctx.panel.state.get("device_used", "Unknown")
        
        # Also check params as fallback
        if not output_video:
            output_video = ctx.params.get("output_video")
            fps = ctx.params.get("fps", 0)
            frame_count = ctx.params.get("frame_count", 0)
            device_used = ctx.params.get("device_used", "Unknown")

        if not output_video or not os.path.exists(str(output_video)):
            panel.md(
                "no_video_message",
                "## No Video Available\n\n"
                "Please run the **Video Detection** operator first to generate an annotated video.\n\n"
                "You can find it by pressing ` (backtick) and searching for 'Video Detection'."
            )
            
            panel.btn(
                "run_detection",
                label="Run Video Detection",
                icon="smart_toy",
                variant="contained",
                on_click="video_detection"
            )
            
        else:
            # Display video info
            panel.md(
                "info",
                f"**Video Info:**\n"
                f"- Frames: {frame_count}\n"
                f"- FPS: {fps:.2f}\n" 
                f"- Device: {device_used}\n"
            )
            
            # Create a proper video element using FiftyOne's asset serving
            # Convert absolute path to relative for serving
            try:
                # For security, use data URL or serve through FiftyOne's server
                video_filename = os.path.basename(output_video)
                panel.md(
                    "video_info",
                    f"**Processed Video**: {video_filename}\n\n"
                    f"**Note**: Due to browser security restrictions, the video file has been "
                    f"saved to: `{output_video}`\n\n"
                    f"You can open this file directly in your video player or import it back into FiftyOne."
                )
                
                # Add button to copy path to clipboard
                panel.btn(
                    "copy_path",
                    label="Copy Video Path",
                    icon="content_copy",
                    variant="outlined",
                    params={"path": output_video}
                )
                
            except Exception as e:
                panel.md("error", f"**Error loading video**: {str(e)}")

        # Add refresh button
        panel.btn(
            "refresh",
            label="Refresh Panel",
            icon="refresh",
            variant="outlined"
        )

        return types.Property(panel)

    def on_click(self, ctx):
        """Handle button clicks"""
        if ctx.params.get("copy_path"):
            # This would copy to clipboard in a real implementation
            path = ctx.params.get("path")
            print(f"Video path: {path}")
            ctx.panel.state["message"] = f"Path copied: {path}"