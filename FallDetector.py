import asyncio
import os
import torch
import numpy as np
from Detection.Utils import ResizePadding
from CameraLoader import CamLoader
from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG
from loguru import logger
import sys
import os

# Configure loguru for async logging
logger.remove()  # Remove default handler
logger.add(
    "logs/fall_detection_{time:YYYY-MM-DD}.log",
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
import threading

import ResultHandler

# worker.
g_tasks = {}
g_tasks_lock = threading.Lock()
def fall_detection_task(task_id, rtsp_address, device):
    while True:
        Start(task_id=task_id, camera=rtsp_address, device=device)
        with g_tasks_lock:
            if not task_id in g_tasks or g_tasks[task_id] is True:
                logger.warning(f"task {task_id} done")
                break

def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


def build_preproc(inp_dets):
    resize_fn = ResizePadding(inp_dets, inp_dets)
    def _preproc(image):
        return resize_fn(image)
    return _preproc


def Start(
        task_id,
        camera,
        detection_input_size=384,
        pose_input_size='224x160',
        pose_backbone='resnet50',
        device='cuda'):
    inp_dets = detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    h, w = map(int, pose_input_size.split('x'))
    pose_model = SPPE_FastPose(pose_backbone, h, w, device=device)

    tracker = Tracker(max_age=30, n_init=3)
    action_model = TSSTG(device=device)

    preproc = build_preproc(inp_dets)

    cam_source = camera
    if isinstance(cam_source, str) and os.path.isfile(cam_source):
        logger.error("not implemented")
        exit(-1)
    else:
        logger.info(f"streaming mode: {cam_source}")
        try:
            cam = CamLoader(cam_source, preprocess=preproc)
            if not cam.valid:
                logger.warning("failed to open stream...")
                return
            cam.start()
        except:
            logger.warning("failed to open stream...")
            return
          
    f = 0
    while cam.grabbed():
        f += 1
        frame = cam.getitem()
        # 检测
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

        # 预测并融合 tracker 结果
        tracker.predict()
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        # 关键点/姿态
        detections = []
        if detected is not None:
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]

        # update tracking.
        tracker.update(detections)

        # action rocognition &  handle.
        for track in tracker.tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)

            action = 'pending..'
            score = 0.0
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action = action_model.class_names[out[0].argmax()]
                score = out[0].max() * 100
            
            if track.time_since_update == 0:
                ResultHandler.handle(task_id, action, track_id, bbox, score, cam.ori_frame)

        # check task status.
        with g_tasks_lock:
            if task_id in g_tasks and g_tasks[task_id] is True:
                del g_tasks[task_id]
                if task_id in ResultHandler.g_images:
                    del ResultHandler.g_images[task_id]
                break
    # release resource.
    cam.stop()
