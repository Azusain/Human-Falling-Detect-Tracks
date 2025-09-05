import cv2
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import os
import numpy as np
from loguru import logger
import threading
from collections import deque

# TODO: make it configurable.
g_images = {}
g_images_lock = threading.Lock()

TTF_PATH = "resources/simsun.ttc"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

def cv2ImgAddText(img, text, left, top, ttf_path, textColor=(0, 255, 0), textSize=40):
    if isinstance(img, np.ndarray):  
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype(ttf_path, textSize, encoding="utf-8") 
    draw.text((left, top), text, textColor, font=fontStyle)  
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

week_map = {
    0: "星期一",
    1: "星期二",
    2: "星期三",
    3: "星期四",
    4: "星期五",
    5: "星期六",
    6: "星期日",
}

# action Classes:
# Standing
# Walking
# Sitting
# Lying Down
# Stand up
# Sit down
# Fall Down    
# pending..    
# fuck your 384 input.
# TODO: track_id is unused.
def handle(task_id, action: str, track_id, bbox, score, ori_frame, camera_name="测试摄像头", save_dir="output"):
    if not action.__contains__("Fall Down") and not action.__contains__("Lying Down"):
      # logger.info(f"skip action {action}")
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
    logger.info(f"{action}: {score:.2f}% | bbox normalized: ({norm_x1:.3f}, {norm_y1:.3f}, {norm_x2:.3f}, {norm_y2:.3f})")
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
    with g_images_lock:
        if task_id not in g_images:
            g_images[task_id] = deque(maxlen=100)  
        g_images[task_id].append({
            "image": ori_frame.copy(),
            "results": result
        })
    
    # draw and save result locally.
    # result_img = ori_frame.copy()
    # cv2.rectangle(result_img, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 0), 2)
    
    # # draw confidence.
    # confidence_text = f"{action}: {score:.2f}"
    # confidence_y = max(y1_orig - 10, 30)  
    # try:
    #     result_img = cv2ImgAddText(result_img, confidence_text, x1_orig, confidence_y, 
    #                              TTF_PATH, textColor=(255, 255, 255), textSize=36)
    # except Exception as e:
    #     logger.warning(f"failed to add confidence text: {e}")
    #     # TODO: fallback to opencv default font.
    #     cv2.putText(result_img, confidence_text, (x1_orig, confidence_y), 
    #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # # datetime text at the top-right.
    # now = datetime.now()
    # date_text = now.strftime("%Y年%m月%d日 ") + week_map[now.weekday()] + now.strftime(" %H:%M:%S")
    # try:
    #     # TODO: text size.
    #     result_img = cv2ImgAddText(result_img, date_text, 10, 10, 
    #                              TTF_PATH, textColor=(255, 255, 255), textSize=48)
    # except Exception as e:
    #     logger.warning(f"failed to add datetime text: {e}")
    #     # TODO: fallback to English datetime.
    #     date_text_en = now.strftime("%Y-%m-%d %H:%M:%S")
    #     cv2.putText(result_img, date_text_en, (10, 30), 
    #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # # add camera name text at the bottom-right.
    # text_size = cv2.getTextSize(camera_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    # camera_x = orig_w - text_size[0] - 10
    # camera_y = orig_h - 10
    # try:
    #     result_img = cv2ImgAddText(result_img, camera_name, camera_x, camera_y - 30, 
    #                              TTF_PATH, textColor=(255, 255, 255), textSize=48)
    # except Exception as e:
    #     logger.warning(f"failed to add camera name: {e}")
    #     # TODO: fallback to default opencv font.
    #     cv2.putText(result_img, camera_name, (camera_x, camera_y), 
    #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # save result to local storage.
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir) 
    # timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    # filename = f"{action}_{timestamp}_track{track_id}.jpg"
    # filepath = os.path.join(save_dir, filename)
    # success = cv2.imwrite(filepath, result_img)
    # if success:
    #     logger.info(f"result saved to path: {filepath}")
    # else:
    #     logger.error(f"failed to save result: {filepath}")

