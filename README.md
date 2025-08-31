# Human Falling Detection and Tracking

Using Tiny-YOLO oneclass to detect each person in the frame and use [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) to get skeleton-pose and then use [ST-GCN](https://github.com/yysijie/st-gcn) model to predict action from every 30 frames of each person tracks.

Which now support 7 actions: **Standing, Walking, Sitting, Lying Down, Stand up, Sit down, Fall Down**.

<div align="center">
    <img src="sample1.gif" width="416">
</div>

## 🆕 Updates in This Fork

- ✅ **CPU Support**: Fixed CUDA hardcoded issues for running on CPU-only machines
- ✅ **Model Loading**: Fixed PyTorch model loading compatibility for different devices
- ✅ **Git LFS Support**: Added Git LFS support for large model files (*.pth)
- ✅ **Dependencies**: Updated requirements and fixed compatibility issues
- 🎯 **Tested**: Successfully tested with real fall detection videos

## Prerequisites

- Python >= 3.6
- PyTorch >= 1.3.1
- OpenCV
- NumPy
- Scipy
- Matplotlib

Original test run on: i7-8750H CPU @ 2.20GHz x12, GeForce RTX 2070 8GB, CUDA 10.2
**Now also supports CPU-only execution.**

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Azusain/Human-Falling-Detect-Tracks.git
cd Human-Falling-Detect-Tracks
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio opencv-python scipy matplotlib numpy
```

3. Pre-trained models:
**All pre-trained models are now included in the repository via Git LFS!**

The repository includes all necessary model files:
- `Models/yolo-tiny-onecls/best-model.pth` (34.7MB) - Tiny-YOLO person detection
- `Models/yolo-tiny-onecls/yolov3-tiny-onecls.cfg` - YOLO configuration file
- `Models/sppe/fast_res50_256x192.pth` (162.6MB) - ResNet50 pose estimation
- `Models/sppe/fast_res101_320x256.pth` (238.9MB) - ResNet101 pose estimation
- `Models/TSSTG/tsstg-model.pth` (24.7MB) - ST-GCN action recognition

**Note**: If you have Git LFS installed, these files will be automatically downloaded when you clone the repository.

## Data

This project has trained a new Tiny-YOLO oneclass model to detect only person objects and to reducing model size. Train with rotation augmented [COCO](http://cocodataset.org/#home) person keypoints dataset for more robust person detection in a variant of angle pose.

For actions recognition used data from [Le2i](http://le2i.cnrs.fr/Fall-detection-Dataset?lang=fr) Fall detection Dataset (Coffee room, Home) extract skeleton-pose by AlphaPose and labeled each action frames by hand for training ST-GCN model.


## Basic Usage

### For Video Files
```bash
python main.py --camera path/to/your/video.mp4 --device cpu
```

### For Webcam (if available)
```bash
python main.py --camera 0 --device cpu
```

### With GPU (if CUDA available)
```bash
python main.py --camera path/to/your/video.mp4 --device cuda
```

### Command Line Options

- `--camera`: Video file path or camera index (default: 0)
- `--device`: Device to run on: 'cpu' or 'cuda' (default: 'cuda')
- `--detection_input_size`: Detection model input size (default: 384)
- `--pose_input_size`: Pose model input size (default: '224x160') 
- `--pose_backbone`: Pose backbone: 'resnet50' or 'resnet101' (default: 'resnet50')
- `--show_detected`: Show detection bounding boxes
- `--show_skeleton`: Show skeleton pose (default: True)
- `--save_out`: Save output video to file

## Action Classes

The system recognizes 7 different actions:

| Action | Color | Description |
|--------|--------|-------------|
| Standing | Green | Normal standing |
| Walking | Green | Normal walking |
| Sitting | Green | Normal sitting |
| **Lying Down** | Orange | Lying position (warning) |
| Stand up | Green | Standing up motion |
| Sit down | Green | Sitting down motion |
| **Fall Down** | Red | **Fall detected (alert)** |

## Architecture

### Three-Stage Detection Pipeline

1. **Human Detection**: Tiny-YOLO v3 single class model
2. **Pose Estimation**: AlphaPose (SPPE FastPose) for keypoint extraction
3. **Action Recognition**: ST-GCN model analyzing 30-frame sequences

### Multi-Object Tracking
- Kalman filter for tracking multiple persons
- Unique track ID for each person
- Temporal consistency across frames

## Performance

- **CPU Mode**: Slower but functional for testing
- **GPU Mode**: Real-time performance on modern GPUs
- **Memory**: ~2GB GPU memory for typical usage
- **Accuracy**: High accuracy due to multi-stage approach and temporal analysis

## Example Output

The system will display:
- Bounding boxes around detected persons
- Skeleton keypoints and connections
- Track IDs for each person
- Action classification with confidence
- Color-coded alerts (Red for falls, Orange for lying down)

## Troubleshooting

### Common Issues

1. **CUDA not available**: Use `--device cpu` flag
2. **Model loading errors**: Ensure Git LFS is installed and models are downloaded
3. **Video not opening**: Check video file path and codec support
4. **Slow performance**: Use GPU if available, or reduce input resolution

### Requirements Installation
```bash
# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GPU (CUDA 11.8)  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Original Credits

Original project by [GajuuzZ](https://github.com/GajuuzZ/Human-Falling-Detect-Tracks)

## References

- AlphaPose: https://github.com/Amanbhandula/AlphaPose
- ST-GCN: https://github.com/yysijie/st-gcn
- YOLO: https://github.com/ultralytics/yolov3

## License

This project follows the original license terms.
