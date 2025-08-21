
from .video_detection_operator import VideoDetection, VideoPlayerPanel

def register(p):
    p.register(VideoDetection)
    p.register(VideoPlayerPanel)