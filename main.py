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

#source = '../Data/test_video/test7.mp4'
#source = '../Data/falldata/Home/Videos/video (2).avi'  # hard detect
source = '../Data/falldata/Home/Videos/video (1).avi'
#source = 2


def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


# ===== 你的回调：在任何状态下都调用 =====
def handle(frame, action, track_id):
    # 这里写你的处理逻辑，比如告警/存库/消息推送等
    print(f"[HANDLE] track={track_id}, action={action}")

# ===== 关键修复：用工厂函数创建带有 resize_fn 的预处理 =====
def build_preproc(inp_dets):
    resize_fn = ResizePadding(inp_dets, inp_dets)
    def _preproc(image):
        # 根据你的项目需要，也可以在这里做额外的变换
        return resize_fn(image)
    return _preproc


def run_demo(args):
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
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        print(f"streaming mode: {cam_source}")
        cam = CamLoader(int(cam_source) if (isinstance(cam_source, str) and cam_source.isdigit()) else cam_source,
                        preprocess=preproc).start()

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

            if args.show_detected:
                for bb in detected[:, 0:5]:
                    frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

        # 更新跟踪
        tracker.update(detections)

        # 动作识别 + 可视化 + 你的 handle
        for track in tracker.tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            clr = (0, 255, 0)
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                action = f'{action_name}: {out[0].max() * 100:.2f}%'
                if action_name == 'Fall Down':
                    clr = (255, 0, 0)
                elif action_name == 'Lying Down':
                    clr = (255, 200, 0)

            # —— 在每个 track 上调用你的处理函数（不管当前状态是否 pending）——
            handle(frame, action, track_id)

            # 仅在本帧有更新时画框/骨架
            if track.time_since_update == 0:
                if args.show_skeleton and len(track.keypoints_list) > 0:
                    frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0, 0), 2)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.4, clr, 1)

        # 显示/FPS
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        fps = 1.0 / max(1e-6, (time.time() - fps_time))
        frame = cv2.putText(frame, f"{f}, FPS: {fps:.2f}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        fps_time = time.time()

        # 注意：如果视频写入需要 BGR，不要把通道翻转；窗口显示也用 BGR 就行
        if writer:
            writer.write(frame)

        cv2.imshow('frame', frame)  # BGR 显示也没问题
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Human Fall Detection')
    parser.add_argument('-C', '--camera', default=source)
    parser.add_argument('--detection_input_size', type=int, default=384)
    parser.add_argument('--pose_input_size', type=str, default='224x160')
    parser.add_argument('--pose_backbone', type=str, default='resnet50')
    parser.add_argument('--show_detected', default=False, action='store_true')
    parser.add_argument('--show_skeleton', default=True, action='store_true')
    parser.add_argument('--save_out', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    run_demo(args)
