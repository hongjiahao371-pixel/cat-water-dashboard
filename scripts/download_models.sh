#!/bin/sh
# 获取运行所需模型（必需: YOLOv8n 检测模型；其余可选）
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$REPO/models"
TARGET="$MODELS/yolov8n_640_opset21.onnx"
mkdir -p "$MODELS"

if [ -f "$TARGET" ]; then
  echo "已存在: $TARGET"
else
  echo "1/3 获取 YOLOv8n 并导出 ONNX（首次会自动安装 ultralytics 并下载权重，体积较大请耐心）..."
  if ! python3 -c "import ultralytics" 2>/dev/null; then
    python3 -m pip install --user ultralytics
  fi
  (cd "$REPO" && python3 - <<'PY'
from ultralytics import YOLO
YOLO("yolov8n.pt").export(format="onnx")
PY
)
  [ -f "$REPO/yolov8n.onnx" ] && mv "$REPO/yolov8n.onnx" "$TARGET"
  echo "OK: $TARGET"
fi

echo ""
echo "2/3 可选: LocateAnything 语义复核模型（5.5GB，影子模式增强验证用）"
echo "   不需要可跳过（系统自动优雅停用该功能）"
echo "   获取与许可说明见 README「开源说明与许可」"

echo ""
echo "3/3 可选: ROI 个性化饮水分类器（强烈推荐，决定识别准确率）"
echo "   该模型与相机视角绑定，需用你自己相机的样本训练:"
echo "   见 training/README.md"
echo "完成。"
