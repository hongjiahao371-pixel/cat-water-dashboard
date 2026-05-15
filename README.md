# Cat Water Dashboard

猫咪饮水监控系统 — 基于 YOLOv8 的实时猫咪饮水检测与记录。

## 架构

```
RTSP摄像头 → QSV转码(960px/8fps) → 运动检测预过滤 → YOLOv8 ONNX推理
                                         ↓
                                   饮水状态机(>3s触发事件)
                                         ↓
                              Flask仪表盘 + EasyNVR录像回放
```

### 服务

| 服务 | 端口 | 功能 |
|------|------|------|
| cat-water-dashboard | 39095 | Flask Web 仪表盘，事件管理，录像回放 API |
| cat-water-detector | - | YOLOv8 ONNX 实时推理，饮水状态机，事件生成 |
| cat-water-camera | - | RTSP 定时抓帧 |
| cat-water-qsv-transcode | 39100 | RTSP → FLV 转码 (CPU编码, 960px/8fps) |
| cat-water-backfill | - | 离线录像回溯检测 (已停用) |

### 检测管线

- **模型**: YOLOv8n COCO (640×640, ONNX)
- **运动预过滤**: 320×180 灰度帧差, 阈值 6.0
- **两轮检测**: 全帧(0.40) + 底部裁剪(0.25)
- **饮水判定**: 猫靠近饮水机 + (猫可见 OR 水面占用) → 持续 >3s 触发事件
- **夜间模式**: 自动 HSV 检测, 4-pass 多尺度推理, CLAHE 增强

## 部署

要求: Docker + Intel GPU (/dev/dri) 用于硬件加速编码 (QSV 不可用时回退 CPU)

```bash
# 1. 准备 YOLOv8n ONNX 模型
mkdir -p models
# 下载 yolov8n ONNX 模型放入 models/

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入摄像头 RTSP 地址

# 3. 启动
docker compose up -d
```

## 环境变量 (检测器)

核心阈值可通过环境变量调整，无需修改代码:

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CAT_SCORE_THRESHOLD` | 0.40 | 猫检测置信度阈值 |
| `CAT_CROP_SCORE_THRESHOLD` | 0.25 | 底部裁剪阈值 |
| `DRINK_MIN_DURATION_SEC` | 3.0 | 最少饮水持续时间 |
| `DRINK_END_GAP_SEC` | 5.0 | 饮水结束间隙容忍 |
| `DETECTOR_MOTION_THRESHOLD` | 6.0 | 运动检测阈值 |
| `PERSON_REJECTION_ENABLED` | true | 人物排斥开关 |
| `PERSON_SCORE_REJECT_THRESHOLD` | 0.30 | 人物排斥阈值 |
| `CONSECUTIVE_FRAMES_REQUIRED` | 2 | 连续确认帧数 |

详见 `detection_worker.py` 顶部完整配置列表。

## 页面

| 路径 | 功能 |
|------|------|
| `/` | 仪表盘 — 实时状态、今日摘要、饮水趋势、事件日历 |
| `/events/` | 事件详情 — 快照、检测指标、EasyNVR 录像回放 |
| `/data/dev.html` | 开发者模式 — ROI 编辑器、模型管理、系统诊断 |
| `/push/` | 推送配置 |
| `/settings/` | 设备设置 |
| `/api/` | API 参考 |

## 技术栈

- Python 3.9+, OpenCV, ONNX Runtime
- YOLOv8n COCO (80类, 640×640)
- Flask + HTML/CSS/JS (Claude Design 风格)
- Docker Compose 部署
