# YOLO11 Poker Detection CoreML Model

## Model Information
- **Format**: CoreML Package (.mlpackage)
- **Size**: ~5.2 MB
- **Type**: YOLO11 neural network
- **Input**: 640x640 RGB images
- **Output**: Poker card detection results

## Usage in iOS
1. Add YOLO11PokerInt8LUT.mlpackage to Xcode project
2. CoreML will automatically compile it for iOS
3. Use Vision framework for inference

## Model Details
- **Classes**: Playing cards (52 classes: 13 ranks × 4 suits)
- **Accuracy**: 99%+ precision/recall (from training)
- **Performance**: Real-time detection optimized
- **Size**: 11x smaller than YOLOv3 (62MB vs 5.2MB)

## License
AGPL-3.0 (Ultralytics YOLO11 + Poker Training)
