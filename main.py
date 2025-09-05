from operator import contains
import os
import torch
import argparse
import numpy as np
import cv2
from Detection.Utils import ResizePadding
from CameraLoader import CamLoader
from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG
import os

import cv2
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import os

ori_frame = None

def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


TTF_PATH = "resources/simsun.ttc"

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
def handle(action: str, track_id, bbox, score, ori_frame, camera_name="测试摄像头", save_dir="output"):
    if not action.__contains__("Fall Down") and not action.__contains__("Lying Down"):
      print(f"skip action {action}")
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
    print(f"{action}: {score:.2f} | bbox normalized: ({norm_x1:.3f}, {norm_y1:.3f}, {norm_x2:.3f}, {norm_y2:.3f})")
    
    result_img = ori_frame.copy()
    cv2.rectangle(result_img, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 0), 2)
    
    # draw confidence.
    confidence_text = f"{action}: {score:.2f}"
    confidence_y = max(y1_orig - 10, 30)  # 确保文本不会超出图像边界
    try:
        result_img = cv2ImgAddText(result_img, confidence_text, x1_orig, confidence_y, 
                                 TTF_PATH, textColor=(255, 255, 255), textSize=36)
    except Exception as e:
        print(f"failed to add confidence text: {e}")
        # TODO: fallback to opencv default font.
        cv2.putText(result_img, confidence_text, (x1_orig, confidence_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # datetime text at the top-right.
    now = datetime.now()
    date_text = now.strftime("%Y年%m月%d日 ") + week_map[now.weekday()] + now.strftime(" %H:%M:%S")
    try:
        # TODO: text size.
        result_img = cv2ImgAddText(result_img, date_text, 10, 10, 
                                 TTF_PATH, textColor=(255, 255, 255), textSize=48)
    except Exception as e:
        print(f"failed to add datetime text: {e}")
        # TODO: fallback to English datetime.
        date_text_en = now.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(result_img, date_text_en, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # add camera name text at the bottom-right.
    text_size = cv2.getTextSize(camera_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    camera_x = orig_w - text_size[0] - 10
    camera_y = orig_h - 10
    try:
        result_img = cv2ImgAddText(result_img, camera_name, camera_x, camera_y - 30, 
                                 TTF_PATH, textColor=(255, 255, 255), textSize=48)
    except Exception as e:
        print(f"failed to add camera name: {e}")
        # TODO: fallback to default opencv font.
        cv2.putText(result_img, camera_name, (camera_x, camera_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # save result to local storage.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{action}_{timestamp}_track{track_id}.jpg"
    filepath = os.path.join(save_dir, filename)
    success = cv2.imwrite(filepath, result_img)
    if success:
        print(f"result saved to path: {filepath}")
    else:
        print(f"failed to save result: {filepath}")


def build_preproc(inp_dets):
    resize_fn = ResizePadding(inp_dets, inp_dets)
    def _preproc(image):
        return resize_fn(image)
    return _preproc


def run(args):
    global ori_frame, g_current_bbox, g_inp_size
    device = args.device

    # 模型
    inp_dets = args.detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    h, w = map(int, args.pose_input_size.split('x'))
    pose_model = SPPE_FastPose(args.pose_backbone, h, w, device=device)

    tracker = Tracker(max_age=30, n_init=3)
    action_model = TSSTG(device=device)

    # 这里用工厂函数拿到线程安全的 preproc
    preproc = build_preproc(inp_dets)

    # 摄像头/文件/RTSP
    cam_source = args.camera
    if isinstance(cam_source, str) and os.path.isfile(cam_source):
        print("not implemented")
        exit(-1)
    else:
        print(f"streaming mode: {cam_source}")
        try:
          cam = CamLoader(cam_source, preprocess=preproc).start()
          if not cam.valid:
              print("failed to open stream...")
              return
        except:
              print("failed to open stream...")
              return
          
    writer = None
    f = 0
    while cam.grabbed():
        f += 1
        frame = cam.getitem()
        ori_frame = frame
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

        # 更新跟踪
        tracker.update(detections)

        # action rocognition &  handle.
        for track in tracker.tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            score = 0.0
            clr = (0, 255, 0)
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action = action_model.class_names[out[0].argmax()]
                score = out[0].max() * 100
                if action == 'Fall Down':
                    clr = (255, 0, 0)
                elif action == 'Lying Down':
                    clr = (255, 200, 0)

            # 仅在本帧有更新时画框/骨架
            if track.time_since_update == 0:
                handle(action, track_id, bbox, score, cam.ori_frame)
        if writer:
            writer.write(frame)

    cam.stop()
    if writer:
        writer.release()

if __name__ =="__main__":
  parser = argparse.ArgumentParser(description='Human Fall Detection')
  parser.add_argument('-i', '--camera',  default="rtsp://admin:hik12345+@192.168.1.84:554/Streaming/Channels/901")
  parser.add_argument('--detection_input_size', type=int, default=384)
  parser.add_argument('--pose_input_size', type=str, default='224x160')
  parser.add_argument('--pose_backbone', type=str, default='resnet50')
  parser.add_argument('-d', '--device', type=str, default='cuda', help='cuda or cpu')
  args = parser.parse_args()
  while True:
    run(args)
    print("reopen stream...")