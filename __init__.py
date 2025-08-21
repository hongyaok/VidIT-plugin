
from .video_detection import VideoDetection, VideoPlayerPanel

def register(p):
    p.register(VideoDetection)
    p.register(VideoPlayerPanel)