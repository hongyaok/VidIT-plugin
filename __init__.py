
from .video_detection import VideoDetection, VideoPlayerPanel, OpenVideoPanel

def register(p):
    """Register the plugin operators and panels"""
    p.register(VideoDetection)
    p.register(OpenVideoPanel) 
    p.register(VideoPlayerPanel)