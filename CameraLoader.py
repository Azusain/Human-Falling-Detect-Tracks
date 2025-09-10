import cv2
import time

from threading import Thread, Lock
import sys

from loguru import logger

# Configure loguru for async logging
logger.remove()  # Remove default handler
logger.add(
    "logs/camera_loader_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    compression="zip",
    enqueue=True,  # Enable async logging
    backtrace=True,
    diagnose=True,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}"
)
# Keep console output with proper colors
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> | {message}",
    level="DEBUG",
    enqueue=True
)


class CamLoader:
    def __init__(self, camera, preprocess=None, ori_return=False):
        self.stream = cv2.VideoCapture(camera)
        self.valid = self.stream.isOpened()
        
        if not self.valid:
            logger.warning(f"warning: cannot open camera {camera}")
            return
            
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.frame_size = (int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.stopped = False
        self.ret = False
        self.frame = None
        self.ori_frame = None
        self.read_lock = Lock()
        self.ori = ori_return
        self.preprocess_fn = preprocess
        self.error_msg = None
        self.t = None

    def start(self):
        if not self.valid:
            raise RuntimeError("Camera is not valid, cannot start")
            
        if self.t is not None and self.t.is_alive():
            return self  # 已经启动了
            
        self.stopped = False
        self.ret = False
        self.error_msg = None
        
        self.t = Thread(target=self.update, daemon=True)
        self.t.start()
        
        # 10s timeout limit.
        max_wait_time = 10.0  
        check_interval = 0.1
        max_checks = int(max_wait_time / check_interval)
        
        for i in range(max_checks):
            if self.ret:
                logger.info(f"camera started successfully after {i * check_interval:.1f}s")
                return self
            if self.error_msg:
                self.stop()
                raise RuntimeError(f'Camera initialization failed: {self.error_msg}')
            time.sleep(check_interval)
        
        self.stop()
        raise TimeoutError('Cannot get a frame from camera within timeout period')

    def update(self):
        consecutive_failures = 0
        max_failures = 10  
        
        try:
            # discard the first 5 frames.
            for _ in range(5):
                ret, _ = self.stream.read()
                if ret:
                    break
                time.sleep(0.1)
            
            while not self.stopped:
                ret, frame = self.stream.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        self.error_msg = f"Failed to read {max_failures} consecutive frames"
                        break
                    time.sleep(0.01)  
                    continue
                
                consecutive_failures = 0  
                
                with self.read_lock:
                    self.ori_frame = frame.copy()
                    processed_frame = frame
                    
                    if self.preprocess_fn is not None:
                        try:
                            processed_frame = self.preprocess_fn(frame)
                        except Exception as e:
                            self.error_msg = f"Preprocessing failed: {str(e)}"
                            break
                    
                    self.ret = True
                    self.frame = processed_frame
                    
        except Exception as e:
            self.error_msg = f"Update thread error: {str(e)}"
        finally:
            if self.error_msg:
                with self.read_lock:
                    self.ret = False

    def grabbed(self):
        return (hasattr(self, 'ret') and self.ret and 
                hasattr(self, 'stopped') and not self.stopped)

    def getitem(self):
        if not self.grabbed():
            return None if not self.ori else (None, None)
            
        with self.read_lock:
            if self.frame is None:
                return None if not self.ori else (None, None)
                
            frame = self.frame.copy()
            ori_frame = self.ori_frame.copy() if self.ori_frame is not None else None
            
        if self.ori:
            return frame, ori_frame
        else:
            return frame

    def stop(self):
        if hasattr(self, 'stopped') and self.stopped:
            return
            
        if hasattr(self, 'stopped'):
            self.stopped = True
        
        if hasattr(self, 't') and self.t is not None and self.t.is_alive():
            self.t.join(timeout=2.0)  
            
        if hasattr(self, 'stream') and self.stream and self.stream.isOpened():
            self.stream.release()


    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


