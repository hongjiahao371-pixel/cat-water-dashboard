#!/usr/bin/env python3
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


DATA_DIR = Path(os.environ.get('MODEL_TRAINING_DATA_DIR', '/app/data'))
POSITIVE_DIR = DATA_DIR / 'positive'
NEGATIVE_DIR = DATA_DIR / 'negative'
STATUS_FILE = DATA_DIR / 'model_training_status.json'
CALIBRATION_FILE = DATA_DIR / 'classifier_calibration.json'
MODEL_PATH = Path(os.environ.get(
    'DRINKING_CLASSIFIER_PATH', '/app/models/cat_drinking_classifier.onnx'))
MIN_POSITIVE = int(os.environ.get('MODEL_TRAINING_MIN_POSITIVE', '50'))
MIN_NEGATIVE = int(os.environ.get('MODEL_TRAINING_MIN_NEGATIVE', '20'))
MIN_PRECISION = float(os.environ.get('MODEL_TRAINING_MIN_PRECISION', '0.90'))
MIN_RECALL = float(os.environ.get('MODEL_TRAINING_MIN_RECALL', '0.80'))
BASELINE_THRESHOLD = float(os.environ.get('DRINKING_CLASSIFIER_THRESHOLD', '0.60'))
POLL_SEC = float(os.environ.get('MODEL_TRAINING_POLL_SEC', '1800'))
IMAGE_SIZE = int(os.environ.get('DRINKING_CLASSIFIER_SIZE', '224'))
CST = timezone(timedelta(hours=8))


def now_iso():
    return datetime.now(CST).isoformat(timespec='seconds')


def read_json(path, default):
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        return payload if isinstance(payload, dict) else dict(default)
    except (OSError, ValueError):
        return dict(default)


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', dir=path.parent,
        prefix=f'.{path.name}.', suffix='.tmp', delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sample_files():
    positive = sorted(POSITIVE_DIR.glob('*.jpg')) if POSITIVE_DIR.is_dir() else []
    negative = sorted(NEGATIVE_DIR.glob('*.jpg')) if NEGATIVE_DIR.is_dir() else []
    return positive, negative


def dataset_fingerprint(positive, negative):
    digest = hashlib.sha256()
    for label, paths in (('1', positive), ('0', negative)):
        for path in paths:
            stat = path.stat()
            digest.update(f'{label}:{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n'.encode())
    return digest.hexdigest()


def classification_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int32)
    predictions = np.asarray(scores, dtype=np.float64) >= float(threshold)
    positive = labels == 1
    negative = ~positive
    tp = int(np.sum(predictions & positive))
    fp = int(np.sum(predictions & negative))
    tn = int(np.sum(~predictions & negative))
    fn = int(np.sum(~predictions & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'threshold': round(float(threshold), 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'specificity': round(specificity, 4),
        'balanced_accuracy': round((recall + specificity) / 2, 4),
        'f1': round(f1, 4),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }


def choose_threshold(labels, scores):
    candidates = {0.35, 0.90, BASELINE_THRESHOLD}
    ordered = sorted(set(float(score) for score in scores))
    candidates.update(max(0.35, min(0.90, score)) for score in ordered)
    candidates.update(
        max(0.35, min(0.90, (left + right) / 2))
        for left, right in zip(ordered, ordered[1:])
    )
    ranked = []
    for threshold in candidates:
        metrics = classification_metrics(labels, scores, threshold)
        eligible = metrics['precision'] >= MIN_PRECISION and metrics['recall'] >= MIN_RECALL
        score = (
            int(eligible), metrics['balanced_accuracy'], metrics['f1'],
            metrics['precision'], -abs(threshold - BASELINE_THRESHOLD),
        )
        ranked.append((score, threshold, metrics))
    return max(ranked, key=lambda item: item[0])[1:]


def cross_validate_threshold(labels, scores, folds=5):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    positive = list(np.flatnonzero(labels == 1))
    negative = list(np.flatnonzero(labels == 0))
    fold_count = max(2, min(int(folds), len(positive), len(negative)))
    calibrated_predictions = np.zeros(len(labels), dtype=np.float64)
    thresholds = []
    for fold in range(fold_count):
        test_indexes = positive[fold::fold_count] + negative[fold::fold_count]
        test_set = set(test_indexes)
        train_indexes = [index for index in range(len(labels)) if index not in test_set]
        threshold, _ = choose_threshold(labels[train_indexes], scores[train_indexes])
        thresholds.append(float(threshold))
        calibrated_predictions[test_indexes] = scores[test_indexes] >= threshold
    calibrated = classification_metrics(labels, calibrated_predictions, 0.5)
    baseline = classification_metrics(labels, scores, BASELINE_THRESHOLD)
    return {
        'folds': fold_count,
        'thresholds': [round(value, 4) for value in thresholds],
        'calibrated': calibrated,
        'baseline': baseline,
    }


def image_probability(session, path):
    frame = cv2.imread(str(path))
    if frame is None or frame.size == 0:
        raise ValueError(f'cannot read {path.name}')
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = np.transpose((resized - mean) / std, (2, 0, 1))[np.newaxis]
    input_name = session.get_inputs()[0].name
    logit = float(np.asarray(session.run(None, {input_name: tensor})[0]).reshape(-1)[0])
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30))))


def write_status(**payload):
    payload.setdefault('ok', True)
    payload.setdefault('updated_at', now_iso())
    payload.setdefault('updated_unix', time.time())
    write_json_atomic(STATUS_FILE, payload)


def run_once():
    positive, negative = sample_files()
    counts = {'confirmed': len(positive), 'false_positive': len(negative)}
    ready = len(positive) >= MIN_POSITIVE and len(negative) >= MIN_NEGATIVE
    if not ready:
        write_status(
            phase='waiting_data', ready=False, active=CALIBRATION_FILE.is_file(),
            **counts, min_confirmed=MIN_POSITIVE, min_false_positive=MIN_NEGATIVE,
            detail=f'还需 {max(0, MIN_POSITIVE-len(positive))} 个确认样本和 {max(0, MIN_NEGATIVE-len(negative))} 个误报样本',
        )
        return
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f'classifier model not found: {MODEL_PATH}')

    fingerprint = dataset_fingerprint(positive, negative)
    current = read_json(CALIBRATION_FILE, {})
    if current.get('dataset_fingerprint') == fingerprint:
        write_status(
            phase='active', ready=True, active=True, **counts,
            threshold=current.get('threshold'), evaluation=current.get('evaluation', {}),
            dataset_fingerprint=fingerprint,
        )
        return

    session = ort.InferenceSession(str(MODEL_PATH), providers=['CPUExecutionProvider'])
    labels = [1] * len(positive) + [0] * len(negative)
    scores = [image_probability(session, path) for path in positive + negative]
    threshold, fitted = choose_threshold(labels, scores)
    evaluation = cross_validate_threshold(labels, scores)
    calibrated = evaluation['calibrated']
    baseline = evaluation['baseline']
    accepted = (
        calibrated['precision'] >= MIN_PRECISION and calibrated['recall'] >= MIN_RECALL
        and calibrated['balanced_accuracy'] + 0.01 >= baseline['balanced_accuracy']
        and calibrated['f1'] + 0.01 >= baseline['f1']
    )
    if not accepted:
        write_status(
            phase='rejected', ready=True, active=CALIBRATION_FILE.is_file(), **counts,
            threshold=round(float(threshold), 4), fitted=fitted, evaluation=evaluation,
            dataset_fingerprint=fingerprint,
            detail='交叉验证未达到安全发布门槛，继续沿用当前阈值',
        )
        return

    payload = {
        'version': 1,
        'trained_at': now_iso(),
        'model': MODEL_PATH.name,
        'threshold': round(float(threshold), 4),
        'baseline_threshold': BASELINE_THRESHOLD,
        'dataset_fingerprint': fingerprint,
        'samples': counts,
        'evaluation': evaluation,
    }
    write_json_atomic(CALIBRATION_FILE, payload)
    write_status(
        phase='active', ready=True, active=True, **counts,
        threshold=payload['threshold'], evaluation=evaluation,
        dataset_fingerprint=fingerprint,
        detail='本猫分类阈值已通过交叉验证并自动启用',
    )


def run_forever():
    while True:
        try:
            run_once()
        except Exception as exc:
            write_status(ok=False, phase='error', ready=False, active=CALIBRATION_FILE.is_file(),
                         error=str(exc)[:240])
        time.sleep(max(60, POLL_SEC))


if __name__ == '__main__':
    run_forever()
