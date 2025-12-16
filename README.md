# YOLO11 Poker Hand Detection

Real-time poker card detection model using YOLO11, optimized for iOS deployment with CoreML.

## Model Information

| Property | Value |
|----------|-------|
| Format | CoreML Package (.mlpackage) |
| Size | ~5.2 MB |
| Input | 640×640 RGB images |
| Classes | 52 (13 ranks × 4 suits) |
| Accuracy | 99%+ precision/recall |

## Contents

```
├── convert.py                    # CoreML conversion script
├── setup_environment.sh          # Environment setup
├── requirements.txt              # Python dependencies
├── detect/                       # Original trained weights
│   └── yolo11n_poker3/weights/best.pt
├── YOLO11PokerInt8LUT.mlpackage/ # Converted CoreML model
└── YOLO11PokerSpec.json          # Model specification
```

## Quick Start

### 1. Setup Environment
```bash
./setup_environment.sh
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Convert Model (if needed)
```bash
python convert.py
```

### 3. iOS Integration
1. Add `YOLO11PokerInt8LUT.mlpackage` to your Xcode project
2. Use Vision framework for inference
3. See [DAT SDK integration example](https://github.com/ebowwa/meta-wearables-dat-ios)

## Performance

- **Inference**: ~10-20ms per frame
- **Memory**: ~30-50MB additional
- **Size**: 11× smaller than YOLOv3 (5.2MB vs 62MB)

## License

- Model: AGPL-3.0 (Ultralytics YOLO11)
- Training data: Apache-2.0
