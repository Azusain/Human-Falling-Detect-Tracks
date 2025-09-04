import os
import torch
import argparse
import numpy as np
from Detection.Utils import ResizePadding
from CameraLoader import CamLoader
from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG
import os

ori_frame = None

def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

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
def handle(action, track_id, bbox, score, ori_frame):
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
    if x1_orig <= 0 or y1_orig <= 0 or x2_orig >= orig_w or y2_orig >= orig_h:
        return
    norm_x1 = x1_orig / orig_w
    norm_y1 = y1_orig / orig_h
    norm_x2 = x2_orig / orig_w
    norm_y2 = y2_orig / orig_h
    print(f"{action}: {score:.2f} | bbox normalized: ({norm_x1:.3f}, {norm_y1:.3f}, {norm_x2:.3f}, {norm_y2:.3f})")


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
        cam = CamLoader(cam_source, preprocess=preproc).start()
        if not cam.valid:
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

# device -> 'cuda' or 'cpu'.
def Start(rtsp_addr, device='cuda'):
  parser = argparse.ArgumentParser(description='Human Fall Detection')
  parser.add_argument('-C', '--camera', default=rtsp_addr)
  parser.add_argument('--detection_input_size', type=int, default=384)
  parser.add_argument('--pose_input_size', type=str, default='224x160')
  parser.add_argument('--pose_backbone', type=str, default='resnet50')
  parser.add_argument('--device', type=str, default=device)
  args = parser.parse_args()
  while True:
    run(args)
