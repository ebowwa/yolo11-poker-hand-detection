#!/bin/bash

echo "🎰 YOLO11 Poker Detection - Environment Setup"
echo "==============================================="

# Check if Python 3.8+ is available
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo "✅ Python version: $PYTHON_VERSION"

    # Check if version is 3.8 or higher
    if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
        echo "✅ Python version is compatible"
    else
        echo "❌ Python 3.8+ required. Current version: $PYTHON_VERSION"
        exit 1
    fi
else
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected"
    echo "📦 Creating virtual environment..."

    if command -v python3 -m venv &> /dev/null; then
        python3 -m venv venv
        source venv/bin/activate
        echo "✅ Virtual environment created and activated"
    else
        echo "❌ Cannot create virtual environment. Installing globally..."
    fi
fi

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing Python packages..."
pip install -r requirements.txt

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Run the conversion: python3 convert_to_coreml.py"
echo "  2. Check the generated CoreML model"
echo "  3. Follow the iOS integration guide"
echo ""

# Verify installations
echo "🔍 Verifying installations..."
python3 -c "
try:
    import ultralytics
    print('✅ ultralytics installed:', ultralytics.__version__)
except ImportError:
    print('❌ ultralytics not installed')

try:
    import coremltools
    print('✅ coremltools installed:', coremltools.__version__)
except ImportError:
    print('❌ coremltools not installed')

try:
    import onnx
    print('✅ onnx installed:', onnx.__version__)
except ImportError:
    print('❌ onnx not installed')

try:
    import torch
    print('✅ torch installed:', torch.__version__)
except ImportError:
    print('❌ torch not installed')
"