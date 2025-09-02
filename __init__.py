
from .video_detection import VideoDetection, VideoPlayerPanel

def register(p):
    """Register the plugin operators and panels"""
    p.register(VideoDetection)
    p.register(VideoPlayerPanel)