#!/usr/bin/env python3
"""把训练好的 ROI ONNX 模型部署进系统。

用法:
    python3 deploy_roi_model.py <onnx文件> [--threshold 0.8] [--apply]

默认 dry-run：只显示将写入的配置，加 --apply 才真正写入。
写入内容: models/roi_drinking.json 的 model/sha256/threshold/enabled，
旧配置自动备份为 roi_drinking.json.bak-<时间戳>。
部署后需重启 detector / archive-verifier / daily-audit 三个容器。

注意: threshold 与相机视角强相关，建议从 0.9 起步、按验证集选取
（选取方法见 training/README.md）。
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

MODELS = Path(__file__).resolve().parent / "models"
CFG = MODELS / "roi_drinking.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", help="训练导出的 ONNX 模型文件")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--apply", action="store_true", help="真正写入（默认 dry-run）")
    args = ap.parse_args()

    onnx = Path(args.onnx)
    if not onnx.is_file():
        sys.exit(f"模型文件不存在: {onnx}")
    if not 0.5 <= args.threshold <= 0.995:
        sys.exit("threshold 必须在 0.5-0.995 之间")

    sha = hashlib.sha256(onnx.read_bytes()).hexdigest()
    name = onnx.name

    cfg = {}
    if CFG.exists():
        try:
            cfg = json.loads(CFG.read_text())
        except ValueError:
            cfg = {}

    new_cfg = dict(cfg)
    new_cfg.update({
        "model": name,
        "sha256": sha,
        "threshold": round(args.threshold, 3),
        "enabled": True,
    })
    if not args.apply:
        print("== DRY-RUN（加 --apply 写入）==")
        print(json.dumps(new_cfg, indent=1, ensure_ascii=False))
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if CFG.exists():
        shutil.copy2(CFG, str(CFG) + f".bak-{stamp}")
    CFG.write_text(json.dumps(new_cfg, indent=1, ensure_ascii=False))
    print(f"已写入 {CFG}")
    print("模型文件请一并放入 models/ 目录，然后重启: docker restart "
          "cat-water-detector cat-water-archive-verifier cat-water-daily-audit")


if __name__ == "__main__":
    main()
