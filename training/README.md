# ROI 饮水分类器训练指南（按相机定制）

仓库自带的 `models/roi_drinking_20260908.onnx` 是针对**某一台相机**训练的，
换相机/换视角必须用你自己的样本重训。本目录是参考实现与流程说明。

## 1. 收集样本
系统会自动把不确定的候选帧存入审核队列（/`review/` 页，含 ROI 裁剪图）。
人工标注为 drinking / not_drinking / no_cat（`data/review_candidate_labels.json`）。
经验值：正样本 ≥100、负样本 ≥40 后开始训练效果较好。

## 2. 训练（参考实现）
`train_roi_reference.py`：MobileNetV3-Small 迁移学习，输入为固定 ROI 区域
裁剪（归一化坐标见 `models/roi_drinking.json` 的 `roi` 字段），224×224，
ImageNet 归一化。按日期划分 train/validation，未标注帧永远不进入训练。
需要 torch/torchvision（建议在电脑上跑，NAS CPU 也可以只是慢）。

## 3. 导出与部署
导出 ONNX（opset ≥17，输入 1×3×224×224，输出单 logit），
放到 `models/` 后编辑 `models/roi_drinking.json`：
`model` 填文件名，`sha256` 填文件哈希，`threshold` 建议从 0.9 起步
按验证集选点（参见文件内 `threshold_selection` 的记录格式），然后重启
detector / archive-verifier / daily-audit 三个容器。

## 4. 验收
用 `scripts/run_replay_regression.sh` 的方法在你自己的录像上建立基线：
断言事件级 TP/FP 与秒级召回/精确率，之后每晚自动回归。
