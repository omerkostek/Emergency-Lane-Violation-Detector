# 
# Defines WebState — the shared-memory object that the pipeline
# background thread and the FastAPI server use to communicate.
# Thread-safe frame updates are done via a Lock 
# lifecycle events (stop, pause, confirm) use threading.Event flags.
# 

import threading
import cv2

class WebState:
    """Shared state between the video pipeline thread and the web server.
    
    Lifecycle states (self.state):
        idle → detecting → calibrating → tracking → done
        (tracking can also be reported as 'paused' by the API layer)
    """

    def __init__(self):
        # Frame buffer
        # Holds the latest JPEG-encoded frame as bytes.
        self.frame_buffer = None
        self.frame_lock = threading.Lock()

        # Control events
        self.stop_event = threading.Event()         # Signal pipeline to exit
        self.pause_event = threading.Event()        # Freeze frame output
        self.confirm_event = threading.Event()      # User confirmed lane calibration
        self.next_frame_event = threading.Event()   # User requested next calibration frame

        # Pipeline metadata
        self.state = "idle"     # Current lifecycle state string
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0.0
        self.frame_count = 0

    def push_frame(self, frame):
        """Encode a BGR OpenCV frame to JPEG and store it in the buffer.
        Called from the pipeline thread; safe to call at full frame rate."""
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self.frame_lock:
            self.frame_buffer = jpeg.tobytes()
