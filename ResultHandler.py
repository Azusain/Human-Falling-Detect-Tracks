import cv2
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import os
import sys
import numpy as np
from loguru import logger
import threading
from collections import deque

# add parent directory to path to import shared_state
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import shared_state
except ImportError:
    shared_state = None

# TODO: make it configurable.
g_images = {}
g_images_lock = threading.Lock()

# global person classifier instance and person class index (will be set by FallDetector)
g_person_classifier = None
g_person_class_index = None

# get person confidence threshold from environment variable, default to 0.5
PERSON_CONFIDENCE_THRESHOLD = float(os.environ.get('PERSON_CONFIDENCE_THRESHOLD', '0.6'))

# action Classes:
# Standing
# Walking
# Sitting
# Lying Down
# Stand up
# Sit down
# Fall Down    
# pending..    

def is_action_safe(action: str) -> bool:
    return (
        not action.__contains__("Fall Down")
        and not action.__contains__("Lying Down")
        and not action.__contains__("Sitting")
        and not action.__contains__("Sit down")
    )
    
# fuck your 384 input.
# TODO: track_id is unused.
def handle(task_id, action: str, track_id, bbox, score, ori_frame):
    if is_action_safe(action):
        return
    # de-scale to original size.
    orig_h, orig_w = ori_frame.shape[:2]    
    scale = 384 / max(orig_h, orig_w)
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    pad_h = 384 - new_h
    pad_w = 384 - new_w
    top = pad_h // 2
    left = pad_w // 2
    x1_orig = (bbox[0] - left) / scale
    y1_orig = (bbox[1] - top) / scale
    x2_orig = (bbox[2] - left) / scale
    y2_orig = (bbox[3] - top) / scale
    
    # filter out the detection result that touch the boundary.
    if x1_orig <= 0 or y1_orig <= 0 or x2_orig >= orig_w or y2_orig >= orig_h:
        return
    x1_orig = max(0, int(x1_orig))
    y1_orig = max(0, int(y1_orig))
    x2_orig = min(orig_w - 1, int(x2_orig))
    y2_orig = min(orig_h - 1, int(y2_orig))
    
    # normalization.
    norm_x1 = x1_orig / orig_w
    norm_y1 = y1_orig / orig_h
    norm_x2 = x2_orig / orig_w
    norm_y2 = y2_orig / orig_h
    
    # crop bbox region and verify if it's a person
    cropped_region = ori_frame[y1_orig:y2_orig, x1_orig:x2_orig]
    if cropped_region.size == 0:
        return
    
    # use detection model to verify if person is present
    person_score, xyxyn, cls = g_person_classifier.Predict(cropped_region)
    
    # proper person detection check
    person_detected = False
    if person_score is not None and cls is not None:
        # convert tensors to lists for easier handling
        if hasattr(cls, 'cpu'):
            cls = cls.cpu()
        if hasattr(person_score, 'cpu'):
            person_score = person_score.cpu()
            
        # convert to lists
        cls_list = cls.tolist() if hasattr(cls, 'tolist') else (cls if isinstance(cls, list) else [cls])
        if isinstance(person_score, (list, tuple)):
            scores_list = person_score
        elif hasattr(person_score, 'tolist'):
            scores_list = person_score.tolist()
        else:
            scores_list = [person_score]
            
        # check each detection to see if any is a person (class 0) with high confidence
        for i, class_id in enumerate(cls_list):
            if i < len(scores_list):
                conf = scores_list[i]
                if int(class_id) == 0 and float(conf) > PERSON_CONFIDENCE_THRESHOLD:  # class 0 is person in COCO
                    person_detected = True
                    break
        
        if not person_detected:
            return
    
    elif not person_detected:
        return

    width = x2_orig - x1_orig
    height = y2_orig - y1_orig

    # save it in memory.
    result = {
        "score": float(score),
        "location": {
            "left": float(x1_orig),
            "top": float(y1_orig),
            "width": float(width),
            "height": float(height)
        }
    }
    # save to both local and shared state
    result_data = {
        "image": ori_frame.copy(),
        "results": result
    }
    
    # save to local state (for backward compatibility)
    with g_images_lock:
        if task_id not in g_images:
            g_images[task_id] = deque(maxlen=100)
        g_images[task_id].append(result_data)
    
    # also save to shared state if available
    if shared_state is not None:
        shared_g_images = shared_state.get_images_dict()
        shared_g_images_lock = shared_state.get_images_lock()
        
        with shared_g_images_lock:
            if task_id not in shared_g_images:
                shared_g_images[task_id] = deque(maxlen=100)
            shared_g_images[task_id].append(result_data)
            # Log successful fall detection with model type
            logger.success(f"[fall] detected {action} (score: {score:.2f}%)")

