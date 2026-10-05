#!/usr/bin/env bash
# Exports YOLOv8n to ONNX with a DYNAMIC batch axis, into worker/models/yolov8n.onnx.
#
# dynamic=True matters: without it the batch size is baked into the model as 1,
# everything works for single images, and any batch > 1 fails at Run() time. The
# worker also refuses to start on such a model.
#
# Usage: scripts/export_model.sh        (PYTHON=python3.12 scripts/export_model.sh to pick one)
set -euo pipefail

cd "$(dirname "$0")/.."            # worker/
PYTHON="${PYTHON:-python3}"

# A throwaway venv keeps ultralytics (and PyTorch, which it pulls in) out of your
# global Python. It lives in worker/.venv, which is gitignored.
if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet ultralytics onnx

# yolo downloads yolov8n.pt next to where it runs and writes yolov8n.onnx beside it,
# so run inside models/.
mkdir -p models
cd models
../.venv/bin/yolo export model=yolov8n.pt format=onnx dynamic=True opset=17

echo "wrote $(pwd)/yolov8n.onnx"
