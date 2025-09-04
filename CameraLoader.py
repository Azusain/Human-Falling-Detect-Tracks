import cv2
import time

from threading import Thread, Lock

from sympy import true


class CamLoader:
    """Use threading to capture a frame from camera for faster frame load.
    Recommend for camera or webcam.

    Args:
        camera: (int, str) Source of camera or video.,
        preprocess: (Callable function) to process the frame before return.
    """
    def __init__(self, camera, preprocess=None, ori_return=False):
        self.stream = cv2.VideoCapture(camera)
        if not self.stream.isOpened():
            self.valid = False
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.frame_size = (int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.stopped = False
        self.ret = False
        self.frame = None
        self.ori_frame = None
        self.read_lock = Lock()
        self.ori = ori_return
        self.valid = True
        self.preprocess_fn = preprocess

    def start(self):
        self.t = Thread(target=self.update, args=())  # , daemon=True)
        self.t.start()
        c = 0
        while not self.ret:
            time.sleep(0.1)
            c += 1
            if c > 20:
                self.stop()
                raise TimeoutError('Can not get a frame from camera!!!')
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            self.read_lock.acquire()
            self.ori_frame = frame.copy()
            if ret and self.preprocess_fn is not None:
                frame = self.preprocess_fn(frame)

            self.ret, self.frame = ret, frame
            self.read_lock.release()

    def grabbed(self):
        """Return `True` if can read a frame."""
        return self.ret

    def getitem(self):
        self.read_lock.acquire()
        frame = self.frame.copy()
        ori_frame = self.ori_frame.copy()
        self.read_lock.release()
        if self.ori:
            return frame, ori_frame
        else:
            return frame

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        if self.t.is_alive():
            self.t.join()
        self.stream.release()

    def __del__(self):
        if self.stream.isOpened():
            self.stream.release()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream.isOpened():
            self.stream.release()


