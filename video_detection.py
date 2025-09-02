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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return out_path

class OpenVideoPanel(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="open_video_panel",
            label="Open Video Panel",
            description="Opens the Video Inference Tool panel",
            icon="video_library",
        )

    def resolve_placement(self, ctx):
        # Place a button in the Samples grid header actions
        return types.Placement(
            types.Places.SAMPLES_GRID_SECONDARY_ACTIONS,
            types.Button(
                label="Open Video Panel",
                icon="video_library",
            ),
        )

    def execute(self, ctx):
        # Use the builtin open_panel operator to open the panel by label
        # The label here must match the PanelConfig.label below: "Video Inference Tool"
        ctx.ops.open_panel(
            name="Video Inference Tool",
            is_active=True,
            layout="horizontal",
        )
        return {"status": "opened"}

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
        
        # Add informational outputs
        outputs.str("device_used", label="Device Used")
        outputs.int("frame_count", label="Total Frames")
        outputs.float("fps", label="Video FPS")
        outputs.str("output_video", label="Output Video Path")
        outputs.str("panel_instruction", label="Next Step")
        
        return types.Property(outputs)

    def execute(self, ctx):
        video_path = ctx.params.get("video")
        model_path = ctx.params.get("model")
        
        # Handle file path extraction
        if isinstance(video_path, dict):
            video_path = video_path.get("absolute_path") or video_path.get("filepath")
        if isinstance(model_path, dict):
            model_path = model_path.get("absolute_path") or model_path.get("filepath")
            
        video_path = str(video_path)
        model_path = str(model_path)
        
        classes_str = str(ctx.params.get("classes", ""))
        conf_thres = float(ctx.params.get("confidence", 0.25))
        
        print(f"Processing: {video_path}")
        print(f"Model: {model_path}")
        print(f"Classes: {classes_str}")
        print(f"Confidence: {conf_thres}")
        
        # Check GPU availability and report it
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        class_order = [c.strip() for c in classes_str.split(",") if c.strip()]

        # Process video
        frames, fps = extract_video_frames(video_path)
        print(f"Extracted {len(frames)} frames at {fps} FPS")
        
        model = YOLO(model_path)
        results = run_inference_on_frames(model, frames, class_order, conf_thres, device)
        drawn = draw_boxes(frames, results)

        # Save output video
        tmpdir = tempfile.mkdtemp(prefix="fo_vid_")
        out_path = os.path.join(tmpdir, "annotated.mp4")
        write_video(drawn, fps, out_path)

        print(f"Saved annotated video to: {out_path}")

        # Return results for the operator output and panel access
        return {
            "output_video": out_path,
            "fps": fps,
            "frame_count": len(frames),
            "device_used": device,
            "panel_instruction": "Open the 'Video Inference Tool' panel to view results"
        }


class VideoPlayerPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="VideoInferenceTool", 
            label="Video Inference Tool",
            description="Displays processed video results",
            icon="video_library"
        )

    def render(self, ctx):
        # Try to get the most recent temporary video file
        # This is a simple approach that looks for recent video files
        import glob
        import time
        
        # Look for recent video files in temp directories
        temp_patterns = [
            "/tmp/fo_vid_*/annotated.mp4",
            "/tmp/fiftyone_vidit_*/*.mp4", 
            "/var/folders/*/fo_vid_*/annotated.mp4",  # macOS temp
        ]
        
        recent_videos = []
        current_time = time.time()
        
        for pattern in temp_patterns:
            try:
                files = glob.glob(pattern)
                for file in files:
                    # Check if file was created in the last 30 minutes
                    if os.path.exists(file) and (current_time - os.path.getctime(file)) < 1800:
                        recent_videos.append((file, os.path.getctime(file)))
            except:
                continue
        
        # Sort by creation time, most recent first
        recent_videos.sort(key=lambda x: x[1], reverse=True)
        
        if not recent_videos:
            return foo.PanelHTML(
                "<div style='padding: 20px;'>"
                "<h3>No Video Available</h3>"
                "<p>Please run the Video Detection operator first.</p>"
                "<p><strong>Steps:</strong></p>"
                "<ol>"
                "<li>Press ` (backtick) to open operator browser</li>"
                "<li>Search for 'Video Detection'</li>"
                "<li>Configure your settings and click Execute</li>"
                "<li>Return to this panel to view results</li>"
                "</ol>"
                "<button onclick='location.reload()' style='margin-top: 10px; padding: 8px 16px;'>"
                "Refresh Panel"
                "</button>"
                "</div>"
            )
        
        # Get the most recent video
        latest_video_path, created_time = recent_videos[0]
        
        # Get video info
        try:
            cap = cv2.VideoCapture(latest_video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            else:
                fps = 0
                frame_count = 0
        except:
            fps = 0
            frame_count = 0
        
        # Format creation time
        import datetime
        created_str = datetime.datetime.fromtimestamp(created_time).strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
        <div style="padding: 20px; font-family: Arial, sans-serif;">
            <h2>🎬 Video Inference Results</h2>
            
            <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                <h3>📊 Video Statistics</h3>
                <p><strong>📁 File:</strong> {os.path.basename(latest_video_path)}</p>
                <p><strong>⏰ Created:</strong> {created_str}</p>
                <p><strong>📼 Frames:</strong> {frame_count:,}</p>
                <p><strong>🎯 FPS:</strong> {fps:.2f}</p>
                <p><strong>📍 Location:</strong> <code>{latest_video_path}</code></p>
            </div>
            
            <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                <h3>📂 Access Your Video</h3>
                <p>Due to browser security restrictions, the video cannot be displayed directly here. 
                Use one of these methods to view your annotated video:</p>
                
                <div style="margin: 10px 0;">
                    <button onclick="navigator.clipboard.writeText('{latest_video_path}')" 
                            style="padding: 8px 16px; margin-right: 10px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">
                        📋 Copy File Path
                    </button>
                    <button onclick="location.reload()" 
                            style="padding: 8px 16px; background-color: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer;">
                        🔄 Refresh Results
                    </button>
                </div>
                
                <h4>💡 Quick Access Options:</h4>
                <ul style="margin: 10px 0;">
                    <li><strong>Option 1:</strong> Copy the path above and open in your video player (VLC, QuickTime, etc.)</li>
                    <li><strong>Option 2:</strong> Open terminal/command prompt and navigate to the file location</li>
                    <li><strong>Option 3:</strong> Import the video into a new FiftyOne dataset for viewing</li>
                </ul>
            </div>
            
            <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px;">
                <h3>🖥️ Command Line Access</h3>
                <p>Open the video directly from terminal:</p>
                <pre style="background-color: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto;">
# macOS
open "{latest_video_path}"

# Linux  
xdg-open "{latest_video_path}"

# Windows
start "{latest_video_path}"</pre>
            </div>
        </div>
        """
        
        return foo.PanelHTML(html)