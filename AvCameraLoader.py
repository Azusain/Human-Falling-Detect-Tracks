import av
import time
import numpy as np
import cv2
from threading import Thread, Lock, Event
import sys
from loguru import logger

# Configure loguru for async logging
logger.remove()  # Remove default handler
logger.add(
    "logs/av_camera_loader_{time:YYYY-MM-DD}.log",
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


class AvCamLoader:
    """
    camera loader using PyAV library for improved stability
    compared to cv2.VideoCapture, av provides better error handling,
    network stream stability and codec support
    """
    
    def __init__(self, camera_source, preprocess=None, ori_return=False, 
                 buffer_size=1, timeout=10.0, reconnect_attempts=0):
        self.camera_source = camera_source
        self.preprocess_fn = preprocess
        self.ori_return = ori_return
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.reconnect_attempts = reconnect_attempts
        
        # stream objects
        self.container = None
        self.stream = None
        
        # frame data
        self.frame = None
        self.ori_frame = None
        self.ret = False
        self.read_lock = Lock()
        
        # threading
        self.stopped = Event()
        self.thread = None
        self.error_msg = None
        
        # stream properties
        self.fps = 25.0  # default fps
        self.frame_size = (640, 480)  # default size
        self.valid = False
        
        # connection stats
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.consecutive_failures = 0
        self.max_consecutive_failures = 30
        
        logger.debug(f"initializing av camera loader for source: {camera_source}")
        
        # try to open stream during initialization
        self._open_stream()
        
    def _open_stream(self):
        """open the av container and find video stream"""
        try:
            # configure options for rtsp/network streams
            options = {
                'rtsp_transport': 'tcp',              # use tcp for better stability
                'max_delay': '5000000',               # 5 second max delay
                'stimeout': str(int(self.timeout * 1000000)),   # connect timeout (us)
                'rw_timeout': str(int(3.0 * 1000000)),          # read/write timeout (us)
                'reconnect': '1',                      # enable reconnect
                'reconnect_streamed': '1',
                'reconnect_delay_max': '3',
                'buffer_size': str(self.buffer_size * 1024 * 1024),  # buffer size in bytes
                'fflags': 'nobuffer',                 # reduce buffering
                'flags': 'low_delay',                 # low delay mode
                'probesize': '32',                    # smaller probe size
                'analyzeduration': '1000000',         # 1 second analyze duration
            }
            
            logger.debug(f"opening container with options: {options}")
            self.container = av.open(self.camera_source, options=options, timeout=self.timeout)
            
            # find video stream
            video_streams = [s for s in self.container.streams if s.type == 'video']
            if not video_streams:
                raise ValueError("no video stream found in source")
                
            self.stream = video_streams[0]
            logger.info(f"found video stream: {self.stream.codec_context.name}, "
                       f"{self.stream.codec_context.width}x{self.stream.codec_context.height}")
            
            # get stream properties
            if self.stream.average_rate:
                self.fps = float(self.stream.average_rate)
            elif self.stream.base_rate:
                self.fps = float(self.stream.base_rate)
            else:
                self.fps = 25.0  # default fallback
                
            self.frame_size = (self.stream.codec_context.width, self.stream.codec_context.height)
            
            # configure stream for low-latency
            self.stream.thread_type = 'AUTO'
            
            self.valid = True
            logger.success(f"stream opened successfully: {self.frame_size[0]}x{self.frame_size[1]} @ {self.fps:.1f}fps")
            return True
            
        except Exception as e:
            error_detail = f"failed to open stream '{self.camera_source}': {type(e).__name__}: {e}"
            logger.error(error_detail)
            self.error_msg = str(e)
            self.valid = False
            return False
    
    def _close_stream(self):
        """safely close the av container"""
        try:
            if self.container:
                self.container.close()
                self.container = None
                self.stream = None
                logger.debug("container closed")
        except Exception as e:
            logger.warning(f"error closing container: {e}")
    
    def start(self):
        """start the camera capture thread"""
        if not self.valid:
            raise RuntimeError(f"failed to initialize camera: {self.error_msg}")
            
        if self.thread and self.thread.is_alive():
            logger.warning("capture thread already running")
            return self
            
        self.stopped.clear()
        self.ret = False
        self.error_msg = None
        self.consecutive_failures = 0
        
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        # wait for first frame with timeout
        max_wait = 10.0
        check_interval = 0.1
        waited = 0.0
        
        while waited < max_wait:
            if self.ret:
                logger.success(f"first frame received after {waited:.1f}s")
                return self
            if self.error_msg:
                self.stop()
                raise RuntimeError(f"camera initialization failed: {self.error_msg}")
            
            time.sleep(check_interval)
            waited += check_interval
            
        self.stop()
        raise TimeoutError("failed to receive first frame within timeout")
    
    def _capture_loop(self):
        """main capture loop running in separate thread"""
        logger.info("capture loop started")
        reconnect_count = 0
        last_check = time.time()
        
        try:
            while not self.stopped.is_set():
                try:
                    # idle watchdog: if no frames for >3s, force reconnect
                    if time.time() - self.last_frame_time > 3.0:
                        raise RuntimeError("no frames received for >3s, triggering reconnect")
                    
                    # decode frames from container
                    for packet in self.container.demux(self.stream):
                        if self.stopped.is_set():
                            break
                            
                        for frame in packet.decode():
                            if self.stopped.is_set():
                                break
                                
                            try:
                                # convert to numpy array (bgr format for opencv compatibility)
                                img = frame.to_ndarray(format='bgr24')
                                
                                # reset failure counter on successful frame
                                self.consecutive_failures = 0
                                reconnect_count = 0
                                
                                # process frame
                                self._process_frame(img)
                                
                                # update stats
                                self.frame_count += 1
                                self.last_frame_time = time.time()
                                
                                # frame processed successfully
                                
                            except Exception as e:
                                logger.error(f"frame processing error: {e}")
                                self.consecutive_failures += 1
                                
                        if self.consecutive_failures >= self.max_consecutive_failures:
                            raise RuntimeError(f"too many consecutive failures: {self.consecutive_failures}")
                            
                except (av.AVError, RuntimeError, OSError) as e:
                    logger.error(f"stream error: {e}")
                    self.consecutive_failures += 1
                    
                    # attempt reconnection (0 means unlimited)
                    if (self.reconnect_attempts == 0 or reconnect_count < self.reconnect_attempts) and not self.stopped.is_set():
                        reconnect_count += 1
                        attempts_str = "unlimited" if self.reconnect_attempts == 0 else str(self.reconnect_attempts)
                        logger.warning(f"attempting reconnection {reconnect_count}/{attempts_str}")
                        
                        self._close_stream()
                        time.sleep(3.0)  # 3 second reconnect interval
                        
                        if self._open_stream():
                            logger.success(f"reconnected successfully (attempt {reconnect_count})")
                            continue
                    
                    # failed to reconnect
                    self.error_msg = f"stream failed after {reconnect_count} reconnection attempts: {e}"
                    break
                    
        except Exception as e:
            self.error_msg = f"capture loop error: {e}"
            logger.error(f"capture loop failed: {e}")
        finally:
            with self.read_lock:
                self.ret = False
            self._close_stream()
            logger.info("capture loop ended")
    
    def _process_frame(self, img):
        """process and store frame data"""
        with self.read_lock:
            # store original frame
            self.ori_frame = img.copy()
            
            # apply preprocessing if provided
            processed_frame = img
            if self.preprocess_fn is not None:
                try:
                    processed_frame = self.preprocess_fn(img)
                except Exception as e:
                    logger.error(f"preprocessing failed: {e}")
                    # use original frame if preprocessing fails
                    processed_frame = img
            
            self.frame = processed_frame
            self.ret = True
    
    def grabbed(self):
        """check if a frame is available"""
        return self.ret and not self.stopped.is_set() and self.error_msg is None
    
    def getitem(self):
        """get the current frame"""
        if not self.grabbed():
            return None if not self.ori_return else (None, None)
        
        with self.read_lock:
            if self.frame is None:
                return None if not self.ori_return else (None, None)
            
            # return copies to avoid threading issues
            frame = self.frame.copy()
            ori_frame = self.ori_frame.copy() if self.ori_frame is not None else None
        
        if self.ori_return:
            return frame, ori_frame
        else:
            return frame
    
    def stop(self):
        """stop the capture thread and release resources"""
        if self.stopped.is_set():
            return
            
        logger.info("stopping camera loader...")
        self.stopped.set()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                logger.warning("capture thread did not stop gracefully")
        
        self._close_stream()
        
        with self.read_lock:
            self.ret = False
            self.frame = None
            self.ori_frame = None
        
        logger.success("camera loader stopped")
    
    def get_stats(self):
        """get capture statistics"""
        return {
            'frame_count': self.frame_count,
            'fps': self.fps,
            'frame_size': self.frame_size,
            'last_frame_time': self.last_frame_time,
            'consecutive_failures': self.consecutive_failures,
            'valid': self.valid,
            'error_msg': self.error_msg
        }
    
    def __enter__(self):
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
