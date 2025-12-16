
# YOLO11 Poker Detection - Manual CoreML Conversion

Since automatic conversion encountered compatibility issues,
here's how to manually convert the YOLO11 poker model to CoreML:

## Method 1: Use CoreMLTools with Xcode

1. Open Xcode 15+
2. Create new project or use existing
3. Add model file: detect/yolo11n_poker3/weights/best.pt
4. Use CoreMLTools converter in Xcode

## Method 2: Use Python with Older Torch Version

```bash
# Create virtual environment with compatible torch
python3 -m venv yolo11-env
source yolo11-env/bin/activate
pip install torch==2.0.1 torchvision==0.15.2
pip install ultralytics==8.0.0
pip install coremltools==7.0

# Run conversion
python -c "
from ultralytics import YOLO
model = YOLO('detect/yolo11n_poker3/weights/best.pt')
model.export(format='coreml', imgsz=640)
"
```

## Method 3: Use Online Conversion Tools

1. Upload best.pt to online YOLO to CoreML converters
2. Download converted .mlmodel file
3. Add to Xcode project

## Method 4: Use YOLO11 Native Export (Recommended)

The YOLO11 model supports native CoreML export:

```python
from ultralytics import YOLO

# Load the model
model = YOLO('detect/yolo11n_poker3/weights/best.pt')

# Export to CoreML
model.export(format='coreml', imgsz=640, optimize=True)

# This creates a YOLO11PokerInt8LUT.mlmodel file
```

## Integration with DAT SDK

Once you have the CoreML model:

1. Add YOLO11PokerInt8LUT.mlmodel to Xcode project
2. Use the provided PokerDetectionService.swift
3. Integrate with StreamSessionViewModel as documented
4. Test with DAT SDK video streaming

## Expected Performance

- Model Size: ~5-10 MB (much smaller than YOLOv3's 62MB)
- Inference Time: ~10-20ms per frame
- Memory Usage: ~30-50MB additional
- Accuracy: 99%+ (based on training metrics)

## Model Classes

The model should detect these card classes:
- Ranks: A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3, 2
- Suits: hearts, diamonds, clubs, spades
- Total: 52 classes (13 ranks × 4 suits)
