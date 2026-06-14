#!/bin/bash
# Download face detection/recognition models for faceunlock

set -e

MODEL_DIR="/usr/share/faceunlock/models"
mkdir -p "$MODEL_DIR"

# SCRFD face detection model
# Options: det_500m (2.4MB, fast), det_2.5g (3.1MB, balanced), det_10g (16MB, accurate)
DETECTOR="${1:-500m}"
DETECTOR_MODEL="$MODEL_DIR/det_${DETECTOR}.onnx"

if [ ! -f "$DETECTOR_MODEL" ]; then
    echo "Downloading SCRFD ${DETECTOR} detector..."
    DET_URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_${DETECTOR}.zip"
    
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "/tmp/buffalo_${DETECTOR}.zip" "$DET_URL"
    elif command -v curl &>/dev/null; then
        curl -L -o "/tmp/buffalo_${DETECTOR}.zip" "$DET_URL"
    else
        echo "Error: wget or curl required" >&2
        exit 1
    fi
    
    # Extract det model from zip
    unzip -o -j "/tmp/buffalo_${DETECTOR}.zip" "det_${DETECTOR}.onnx" -d "$MODEL_DIR/" 2>/dev/null || \
    unzip -o -j "/tmp/buffalo_${DETECTOR}.zip" "models/det_${DETECTOR}.onnx" -d "$MODEL_DIR/" 2>/dev/null || \
    echo "Warning: Could not extract detector from zip. Please download manually."
    
    rm -f "/tmp/buffalo_${DETECTOR}.zip"
    echo "Detector saved to: $DETECTOR_MODEL"
else
    echo "Detector already exists: $DETECTOR_MODEL"
fi

# AdaFace IR-50 face recognition model
RECOGNIZER_MODEL="$MODEL_DIR/adaface_ir50.onnx"
if [ ! -f "$RECOGNIZER_MODEL" ]; then
    echo ""
    echo "AdaFace IR-50 ONNX model not found at: $RECOGNIZER_MODEL"
    echo "To convert from PyTorch:"
    echo "  1. Clone https://github.com/mk-minchul/AdaFace"
    echo "  2. Download IR-50 pretrained checkpoint"
    echo "  3. Export to ONNX:"
    echo "     import torch"
    echo "     model = ...  # load AdaFace IR-50"
    echo "     dummy = torch.randn(1, 3, 160, 160)"
    echo "     torch.onnx.export(model, dummy, 'adaface_ir50.onnx',"
    echo "                       input_names=['input'], output_names=['embedding'],"
    echo "                       dynamic_axes={'input': {0: 'batch_size'}})"
else
    echo "Recognizer already exists: $RECOGNIZER_MODEL"
fi

echo ""
echo "Models directory: $MODEL_DIR"
ls -lh "$MODEL_DIR/"
