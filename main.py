import os
import cv2
import time
import torch
import argparse
import numpy as np

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG


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
def handle(frame, action, track_id, bbox, score):
    h, w = frame.shape[:2]
    norm_x1 = bbox[0] / w
    norm_y1 = bbox[1] / h
    norm_x2 = bbox[2] / w
    norm_y2 = bbox[3] / h
    # 这里写你的处理逻辑，比如告警/存库/消息推送等
    print(f"{action}: {score:.2f}")

# ===== 关键修复：用工厂函数创建带有 resize_fn 的预处理 =====
def build_preproc(inp_dets):
    resize_fn = ResizePadding(inp_dets, inp_dets)
    def _preproc(image):
        # 根据你的项目需要，也可以在这里做额外的变换
        return resize_fn(image)
    return _preproc


def run(args):
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
        try:
          cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
        except:
          print("failed to open file...")
          return
    else:
        print(f"streaming mode: {cam_source}")
        try:
          cam = CamLoader(cam_source, preprocess=preproc).start()
        except:
          print("failed to open stream...")
          return
          
    # 输出视频（可选）
    writer = None
    if args.save_out:
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))

    fps_time = time.time()
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
                handle(frame, action, track_id, bbox, score)
                # TODO: for testing.
                # frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                # frame = cv2.putText(frame, str(track_id), (center[0], center[1]),
                #                     cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0, 0), 2)
                # frame = cv2.putText(frame, f"{action}:{score:.2f}", (bbox[0] + 5, bbox[1] + 15),
                #                     cv2.FONT_HERSHEY_COMPLEX, 0.4, clr, 1)


        # 注意：如果视频写入需要 BGR，不要把通道翻转；窗口显示也用 BGR 就行
        if writer:
            writer.write(frame)

        # TODO: we dont need gui display.
        cv2.imshow('frame', frame)  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    if writer:
        writer.release()
    # TODO: we dont need gui dispaly.
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Human Fall Detection')
    parser.add_argument('-C', '--camera', default="")
    parser.add_argument('--detection_input_size', type=int, default=384)
    parser.add_argument('--pose_input_size', type=str, default='224x160')
    parser.add_argument('--pose_backbone', type=str, default='resnet50')
    parser.add_argument('--show_detected', default=False, action='store_true')
    parser.add_argument('--show_skeleton', default=True, action='store_true')
    parser.add_argument('--save_out', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    while True:
      run(args)
