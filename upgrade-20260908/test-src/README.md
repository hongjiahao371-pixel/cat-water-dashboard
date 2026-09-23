# Cat Water Dashboard

猫咪饮水监控系统 — 基于 YOLOv8 的实时猫咪饮水检测与记录。

## 架构

```
摄像头 / EasyNVR 低清流 → 转码 → YOLOv8 猫检测 → 饮水分类器与时序门
                                                        ↓
                                     实时事件 / 不确定候选
                                                        ↓
                       EasyNVR 高清录像复核 → 已确认统计 / 排除 / 人工审核
                                      ↓
             LocateAnything 关键帧语义复核（影子模式，不改统计）
```

### 服务

| 服务 | 端口 | 功能 |
|------|------|------|
| cat-water-dashboard | 39095 | Web 仪表盘，事件管理，录像回放与健康 API |
| cat-water-detector | - | YOLOv8 ONNX 实时推理，饮水状态机，事件生成 |
| cat-water-camera | - | RTSP 定时抓帧 |
| cat-water-qsv-transcode | 39100 | RTSP → FLV 转码 (CPU编码, 960px/8fps) |
| cat-water-archive-verifier | - | EasyNVR 高清录像二次复核和漏检候选恢复 |
| cat-water-locate-verifier | - | LocateAnything-3B Q6_K 对猫嘴接触水面的高清关键帧做严格三帧语义复核和 A/B 评估；默认影子模式 |
| cat-water-notifier | - | 日报、未饮水和低饮水提醒，带去重与失败重试 |
| cat-water-daily-audit | - | 凌晨先做关键帧预筛，再用低频强证据门和 2 FPS 精校扫描前一日高清录像，发现实时阶段完全遗漏的候选 |
| cat-water-model-trainer | - | 持续检查人工样本，达到门槛后交叉验证并安全发布本猫分类阈值校准 |
| cat-water-backfill | - | 离线录像回溯检测（`maintenance` profile 手动启用） |

### 检测管线

- **模型**: YOLOv8n COCO (640×640, ONNX)
- **运动预过滤**: 320×180 灰度帧差, 阈值 6.0
- **两轮检测**: 全帧(0.40) + 底部裁剪(0.25)
- **饮水判定**: 分类器达到日/夜阈值且位于饮水区候选范围；需要连续帧确认。无分类器时才使用严格几何判定。
- **证据保留**: 确认事件会写入快照、叠加标注短片和帧级元数据；旧事件可从 EasyNVR 录像动态提取。
- **高清复核**: 实时事件结束后从 EasyNVR 高清录像取证，只有同一空间目标的重复高置信锚点才确认。
- **语义复核**: 对已有高清复核结论的事件，从分类器峰值附近提取 3 张关键帧，裁剪到饮水机区域后交给 LocateAnything；模型框还必须接近标定水区并跨帧保持空间一致。结果保存到独立状态与证据目录，默认不改写正式事件和统计。
- **漏检恢复**: 未形成事件的边界候选会自动进入高清探测；确认前不计入统计，也不会自动写成训练标签。
- **每日巡检**: 每天 03:30 对前一日 EasyNVR 高清录像做低频预筛和 2 FPS 严格复核；新发现只进入审核队列，人工确认后才计入统计。
- **统计口径**: 只统计人工确认或高清确认事件；历史未复核记录保留但从趋势和汇总隔离。
- **饮水量估算**: 使用高置信舔水锚点形成的有效舔水时长乘以可配置 ml/s，并显示参考范围；设置页可根据一段称重区间内的可信事件自动计算本猫系数，并选择重算已有 v2 事件。
- **消息提醒**: 支持通用 JSON Webhook 和飞书自定义机器人 Webhook；日报、长时间未饮水、低于近 7 日基线和录像巡检新候选提醒均有去重、冷却和最多 5 次重试。
- **写接口安全**: 配置 `CAT_WATER_API_TOKEN` 后，所有 POST 接口必须携带 `X-API-Token` 或 Bearer Token；访问一次 `/?token=...` 可将令牌安全保存到当前浏览器并立即从地址栏移除。
- **长期归档**: 活动事件超过保留期后按日期写入 `data/events_archive/`；历史页和 `include_archived=1` API 可以跨活动与归档文件查询。
- **持续校准**: 标签达到 50 个确认和 20 个误报后，NAS 自动执行分层交叉验证；只有精确率、召回率和相对基线同时达标才写入校准文件，检测器会自动热读取。
- **夜间模式**: 自动 HSV 检测和 CLAHE 增强。夜间 ROI 与模型必须用真实 IR 样本校准，不能只靠降低阈值。
- **运行健康**: 相机、转码和检测器均写入带时间戳的状态文件；`/api/health` 会报告整条实时管线是否可用。

## 部署

要求: Docker；Intel GPU `/dev/dri` 为可选加速，驱动不可用时自动使用 CPU

```bash
# 1. 准备模型
mkdir -p models
# 下载 yolov8n ONNX 模型放入 models/
# 将 locate-anything-q6_k.gguf 放入 models/；必须遵守 NVIDIA 模型许可

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入摄像头 RTSP 地址

# 3. 启动
docker compose up -d

# 可选：指定事件做一次影子复核
docker compose run --rm cat-water-locate-verifier \
  python3 -u locate_verifier_worker.py --once --event-id EVENT_ID
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
| `CONSECUTIVE_FRAMES_REQUIRED` | 3 | 连续确认帧数 |
| `ARCHIVE_AUTO_RECOVERY_ENABLED` | true | 自动对未标注边界候选做高清探测 |
| `ARCHIVE_AUTO_RECOVERY_MIN_CONFIDENCE` | 0.15 | 自动高清探测最低实时分类置信度 |
| `DAILY_AUDIT_RUN_TIME` | 03:30 | 每日录像巡检开始时间 |
| `DAILY_AUDIT_PREFILTER_INTERVAL_SEC` | 10 | 高清录像第一阶段抽样间隔；命中后升至 2 FPS 复核 |
| `DAILY_AUDIT_MOTION_THRESHOLD` | 12 | 静态机位背景差分阈值；每分钟仍强制一次 YOLO 检查 |
| `NOTIFICATION_POLL_SEC` | 60 | 通知规则检查间隔 |
| `CAT_WATER_API_TOKEN` | 空 | 写接口鉴权令牌；生产环境必须设置 |
| `MODEL_TRAINING_MIN_POSITIVE` | 50 | 自动分类阈值校准所需最少确认样本 |
| `MODEL_TRAINING_MIN_NEGATIVE` | 20 | 自动分类阈值校准所需最少误报样本 |
| `LOCATE_SHADOW_MODE` | true | 只保存 LocateAnything 结论，不改写正式事件 |
| `LOCATE_MODE` | slow | CPU 解码模式；当前方案优先稳健性 |
| `LOCATE_MAX_FRAMES` | 3 | 每个候选事件最多复核的高清关键帧数 |
| `LOCATE_MIN_COHERENT_PAIRS` | 2 | 强确认所需的最少跨帧空间一致对 |
| `LOCATE_MIN_COMPLETE_FRAMES` | 3 | 强确认所需的完整取证帧数 |
| `LOCATE_MIN_LOCALIZED_HITS` | 2 | 口鼻小框强证据所需的最少命中帧数 |
| `LOCATE_MAX_LOCALIZED_BOX_AREA_RATIO` | 0.12 | 口鼻定位框允许占整帧的最大比例；整猫大框不走该快速确认分支 |
| `LOCATE_REQUIRE_ALL_FRAMES` | true | 只有全部关键帧均命中才允许输出强确认 |
| `LOCATE_MAX_LOAD` | 3.5 | 1 分钟负载超过此值时推迟语义复核 |
| `LOCATE_MIN_INTERVAL_SEC` | 1800 | 两次后台复核之间的最短冷却时间，避免持续占满 CPU |

详见 `detection_worker.py` 顶部完整配置列表。

## 生产验证

- 2026-07-13 对 2026-07-12 的 EasyNVR 留存录像完成全日回扫：1418 个一分钟文件、8508 个预筛帧、0 个解码或分析错误。
- 23 个文件进入 1 FPS 时序门，9 个文件进入 2 FPS 精校；已知事件按录像锚点或实际开始时间去重。
- 原始候选按活动区间跨文件归并后，人工视觉确认 5 次实时阶段完全遗漏的饮水；当日可信统计由 5 次 / 68.6 ml 更新为 10 次 / 170.7 ml（视频估算，尚未针对本猫称重标定）。
- 通知发送已用临时 Webhook 完成端到端验证；生产 Webhook 留空时服务保持健康但不对外发送。
- 称重校准接口已用当日 10 次可信事件、379.4 秒有效舔水做只读预览验证，正确计算 0.45 ml/秒；未应用测试结果或改写生产统计。
- 2026-07-13 安全、长期归档、巡检候选提醒和持续模型校准闭环补齐后，本地回归扩展为 47 项。

## 页面

| 路径 | 功能 |
|------|------|
| `/` | 仪表盘 — 实时状态、今日摘要、饮水趋势、事件日历 |
| `/events/` | 活动与长期归档事件查询；带 id 时显示快照、检测指标和 EasyNVR 录像回放 |
| `/api/transcode-status` | 转码连接、重连和存活状态 |
| `/settings/` | 设备设置、饮水量称重校准与历史 v2 事件重算 |
| `/review/` | 边界候选审核与自动高清恢复状态 |
| `/push/` | Webhook、日报和异常提醒策略及测试发送 |
| `/api/health` | 实时检测与高清复核链路健康状态 |
| `/api/notification-status` | 通知配置、重试队列和最近发送状态 |
| `/api/daily-audit-status` | 每日录像巡检进度和结果摘要 |
| `/api/model-training-status` | 本猫分类阈值持续校准状态、样本门槛和交叉验证结果 |
| `/api/locate-verifier-status` | LocateAnything 模型、策略版本、影子模式、处理阶段，以及按人工/自动真值和昼夜拆分的全样本 A/B 指标 |
| `/api/events?include_archived=1` | 跨活动数据与日期归档查询历史事件 |
| `/api/events/archive-status` | 长期归档文件、数量和日期范围 |
| `/api/volume-calibrations` | 最近 50 次饮水量校准记录 |
| `/api/` | API 参考 |

## 技术栈

- Python 3.9+, OpenCV, ONNX Runtime, LocateAnything-3B Q6_K/ggml
- YOLOv8n COCO (80类, 640×640)
- Python `ThreadingHTTPServer` + HTML/CSS/JS
- Docker Compose 部署
