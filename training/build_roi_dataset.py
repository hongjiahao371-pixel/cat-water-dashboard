#!/usr/bin/env python3
"""从审核队列的人工标注构建 ROI 分类器训练数据集。

在 cat-water 容器内执行（有 cv2 与挂载的数据目录）:
    docker exec cat-water python3 /app/training/build_roi_dataset.py

读取 data/review_candidate_labels.json 的人工标签（drinking/not_drinking/
no_cat），从对应的整帧快照裁出 ROI 区域（与 roi_drinking.json 的 roi 一致），
输出:
    data/roi_dataset/images/<label>/<id>.jpg
    data/roi_dataset/manifest.json

之后把 data/roi_dataset/ 拷到有 torch 的机器上，按 training/README.md 用
training/train_roi_reference.py 训练并导出 ONNX，再用
scripts/deploy_roi_model.py 部署回系统。

no_cat 样本同样作为负样本（ROI 区域无饮水行为）。无法读取的样本自动跳过。
"""
import json
import os
from pathlib import Path

import cv2

DATA = Path(os.environ.get("DATA_DIR", "/app/data"))
MEDIA = Path(os.environ.get("MEDIA_DIR", "/app/media"))
ROI_CFG = Path(os.environ.get("ROI_CONFIG", "/app/models/roi_drinking.json"))
OUT = Path(os.environ.get("ROI_DATASET_DIR", str(DATA / "roi_dataset")))

LABEL_MAP = {"drinking": 1, "not_drinking": 0, "no_cat": 0}


def load_roi():
    try:
        return tuple(json.loads(ROI_CFG.read_text())["roi"])
    except Exception:
        return (0.08, 0.12, 0.94, 0.82)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    roi = load_roi()
    labels = json.loads((DATA / "review_candidate_labels.json").read_text())
    counts = {}
    records = []
    skipped = 0
    for sid, info in sorted(labels.items()):
        if not isinstance(info, dict):
            continue
        lab = info.get("label")
        if lab not in LABEL_MAP:
            continue
        # 永久库优先（标注时保存的副本），候选轮转清理后仍可重建数据集
        src = None
        for cand_path in (DATA / "roi_dataset_raw" / lab / f"{sid}.jpg",
                          MEDIA / "review_candidates" / f"{sid}.jpg"):
            if cand_path.is_file():
                src = cand_path
                break
        if src is None:
            skipped += 1
            continue
        img = cv2.imread(str(src))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]
        x1, y1 = int(w * roi[0]), int(h * roi[1])
        x2, y2 = int(w * roi[2]), int(h * roi[3])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            skipped += 1
            continue
        out_dir = OUT / "images" / lab
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{sid}.jpg"
        cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({"id": sid, "image": str(out.relative_to(OUT)),
                        "label": LABEL_MAP[lab]})
        counts[lab] = counts.get(lab, 0) + 1
    manifest = {"roi": list(roi), "records": records}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    if not records:
        print("警告: 0 条可用样本 - 标注对应的快照可能已被候选轮转清理。")
        print("从现在起新打的标签会在标注时立即保存样本（server 已支持）。")
    print("dataset:", dict(counts), "total", len(records), "skipped", skipped)
    print("output:", OUT)


if __name__ == "__main__":
    main()
