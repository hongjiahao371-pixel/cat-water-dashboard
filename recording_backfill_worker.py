#!/usr/bin/env python3
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2

import detection_worker as detector_mod

EASYNVR_DB_PATH = Path(os.environ.get('EASYNVR_DB_PATH', '/easynvr/configs/data.db'))
EASYNVR_RECORD_ROOT = Path(os.environ.get('EASYNVR_RECORD_ROOT', '/easynvr/temporary/gbrecords'))
BACKFILL_STATUS_FILE = Path(os.environ.get('BACKFILL_STATUS_FILE', '/app/data/backfill_status.json'))
BACKFILL_STATE_FILE = Path(os.environ.get('BACKFILL_STATE_FILE', '/app/data/backfill_state.json'))
BACKFILL_POLL_SEC = float(os.environ.get('BACKFILL_POLL_SEC', '30'))
BACKFILL_LOOKBACK_MIN = int(os.environ.get('BACKFILL_LOOKBACK_MIN', '180'))
BACKFILL_ANALYZE_FPS = float(os.environ.get('BACKFILL_ANALYZE_FPS', '4'))
BACKFILL_CHANNEL_ID = os.environ.get('BACKFILL_CHANNEL_ID', os.environ.get('EASYNVR_CHANNEL_ID_SMOOTH', 'OpD6HNpPU8j0X_profile_2'))
BACKFILL_MIN_FILE_MB = float(os.environ.get('BACKFILL_MIN_FILE_MB', '1.5'))
BACKFILL_MAX_PROCESSED_TRACK = int(os.environ.get('BACKFILL_MAX_PROCESSED_TRACK', '500'))
BACKFILL_USE_FFMPEG = os.environ.get('BACKFILL_USE_FFMPEG', '1') == '1'
BACKFILL_HWACCEL_MODE = os.environ.get('BACKFILL_HWACCEL_MODE', 'qsv').lower()
FFMPEG_BIN = os.environ.get('FFMPEG_BIN', 'ffmpeg')
FFPROBE_BIN = os.environ.get('FFPROBE_BIN', 'ffprobe')

BACKFILL_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
BACKFILL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


def now_iso():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def write_status(payload):
    payload.setdefault('updated_at', now_iso())
    BACKFILL_STATUS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def read_json(path, default):
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return json.loads(json.dumps(default))


def save_state(state):
    processed = state.get('processed', {})
    if len(processed) > BACKFILL_MAX_PROCESSED_TRACK:
        items = sorted(processed.items(), key=lambda item: item[1].get('processed_at', ''))
        state['processed'] = dict(items[-BACKFILL_MAX_PROCESSED_TRACK:])
    BACKFILL_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')


def parse_any_datetime(value):
    if not value:
        return None
    raw = str(value).strip().replace('Z', '+00:00')
    if raw.endswith(' UTC'):
        raw = raw[:-4] + '+00:00'
    if '.' in raw:
        head, tail = raw.split('.', 1)
        tz_pos = max(tail.find('+'), tail.find('-'))
        if tz_pos > 0:
            frac = tail[:tz_pos]
            suffix = tail[tz_pos:]
        else:
            frac = tail
            suffix = ''
        frac = (frac + '000000')[:6]
        raw = f'{head}.{frac}{suffix}'
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def resolve_record_path(raw_path):
    if not raw_path:
        return None
    path = Path(str(raw_path))
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([
            EASYNVR_RECORD_ROOT / path,
            EASYNVR_DB_PATH.parent.parent / path,
            Path('/easynvr') / path,
        ])
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def discover_candidates_from_db(cutoff_dt):
    items = []
    if not EASYNVR_DB_PATH.exists():
        return items
    try:
        conn = sqlite3.connect(str(EASYNVR_DB_PATH))
        cursor = conn.cursor()
        cursor.execute(
            '''
            select path, channel_id, created_at, updated_at, file
            from record_files
            where deleted_at is null
              and (? = '' or channel_id = ?)
            order by created_at asc
            ''',
            (BACKFILL_CHANNEL_ID, BACKFILL_CHANNEL_ID),
        )
        for raw_path, channel_id, created_at, updated_at, raw_file in cursor.fetchall():
            created_dt = parse_any_datetime(created_at)
            updated_dt = parse_any_datetime(updated_at)
            if cutoff_dt and created_dt and created_dt < cutoff_dt and updated_dt and updated_dt < cutoff_dt:
                continue
            resolved = resolve_record_path(raw_path)
            file_meta = {}
            if raw_file:
                try:
                    file_meta = json.loads(raw_file)
                except Exception:
                    file_meta = {}
            if not resolved and isinstance(file_meta, dict):
                for key in ('path', 'file_path', 'target', 'name'):
                    resolved = resolve_record_path(file_meta.get(key))
                    if resolved:
                        break
            items.append({
                'path': str(resolved or raw_path),
                'resolved_path': str(resolved) if resolved else '',
                'channel_id': channel_id,
                'created_at': created_dt.isoformat() if created_dt else '',
                'updated_at': updated_dt.isoformat() if updated_dt else '',
                'source': 'db',
            })
        conn.close()
    except Exception as exc:
        write_status({'ok': False, 'phase': 'db_scan_error', 'detail': str(exc)})
    return items


def discover_candidates_from_dir(cutoff_ts):
    items = []
    if not EASYNVR_RECORD_ROOT.exists():
        return items
    for path in sorted(EASYNVR_RECORD_ROOT.rglob('*.mp4')):
        try:
            stat = path.stat()
        except Exception:
            continue
        if cutoff_ts and stat.st_mtime < cutoff_ts:
            continue
        if stat.st_size < BACKFILL_MIN_FILE_MB * 1024 * 1024:
            continue
        items.append({
            'path': str(path),
            'resolved_path': str(path),
            'channel_id': BACKFILL_CHANNEL_ID,
            'created_at': datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec='seconds'),
            'updated_at': datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec='seconds'),
            'source': 'dir',
        })
    return items


def ffprobe_meta(source_path):
    cmd = [
        FFPROBE_BIN,
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,avg_frame_rate,nb_frames,width,height:format=duration',
        '-of', 'json',
        str(source_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or 'ffprobe failed').strip())
    payload = json.loads(result.stdout or '{}')
    stream = (payload.get('streams') or [{}])[0]
    fmt = payload.get('format') or {}

    def parse_ratio(value):
        if not value:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if '/' in value:
            a, b = value.split('/', 1)
            try:
                a = float(a)
                b = float(b)
                return a / b if b else 0.0
            except Exception:
                return 0.0
        try:
            return float(value)
        except Exception:
            return 0.0

    fps = parse_ratio(stream.get('avg_frame_rate')) or parse_ratio(stream.get('r_frame_rate')) or 25.0
    duration = float(fmt.get('duration') or 0.0)
    if duration <= 0:
        nb = stream.get('nb_frames')
        if nb and fps:
            try:
                duration = float(nb) / fps
            except Exception:
                duration = 0.0
    return {
        'fps': fps if fps and fps < 120 else 25.0,
        'duration_sec': duration,
        'width': int(stream.get('width') or 0),
        'height': int(stream.get('height') or 0),
    }


def ffmpeg_extract_frames(source_path, fps):
    temp_dir = Path(tempfile.mkdtemp(prefix='backfill-frames-'))
    pattern = temp_dir / 'frame-%06d.jpg'
    base = [FFMPEG_BIN, '-y', '-loglevel', 'error']
    qsv_cmd = base + [
        '-hwaccel', 'qsv',
        '-qsv_device', '/dev/dri/renderD128',
        '-c:v', 'hevc_qsv',
        '-i', str(source_path),
        '-vf', f'fps={fps}',
        '-q:v', '3',
        str(pattern),
    ]
    cpu_cmd = base + [
        '-i', str(source_path),
        '-vf', f'fps={fps}',
        '-q:v', '3',
        str(pattern),
    ]
    tried = []
    for mode, cmd in [('qsv', qsv_cmd), ('cpu', cpu_cmd)]:
        if mode == 'qsv' and (not BACKFILL_USE_FFMPEG or BACKFILL_HWACCEL_MODE != 'qsv'):
            continue
        if mode == 'cpu' and not BACKFILL_USE_FFMPEG:
            continue
        tried.append(mode)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        generated = sorted(temp_dir.glob('frame-*.jpg'))
        if result.returncode == 0 and generated:
            return temp_dir, generated, mode
        for item in generated:
            item.unlink(missing_ok=True)
    if BACKFILL_USE_FFMPEG and not tried:
        result = subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=180)
        generated = sorted(temp_dir.glob('frame-*.jpg'))
        if result.returncode == 0 and generated:
            return temp_dir, generated, 'cpu'
    shutil.rmtree(temp_dir, ignore_errors=True)
    return None, [], 'opencv'


class BackfillDetector(detector_mod.Detector):
    def __init__(self, source_label=''):
        super().__init__()
        self.source_label = source_label
        self.posted_events = []

    def post_event(self, event_id, start_ts, end_ts, peak_frame, settings):
        duration = max(1.0, end_ts - start_ts)
        estimated_ml = round(duration * float(settings.get('mlPerSec', 0.14) or 0.14), 1)
        snapshot_url = self.save_snapshot(peak_frame or self.last_frame, event_id)
        started = datetime.fromtimestamp(start_ts).astimezone()
        ended = datetime.fromtimestamp(end_ts).astimezone()
        confidence = min(0.98, 0.55 + min(duration / 35.0, 0.22) + min(self.event_peak_score / 140.0, 0.15))
        summary = f'录像补检发现猫在饮水机附近持续停留 {duration:.1f} 秒；当前基于猫检测 + 低头候选 + 占用/运动规则回放得出。'
        debug_curve_url = self.save_debug_curve(event_id, start_ts, end_ts)
        payload = {
            'id': event_id,
            'title': f"{started.strftime('%H:%M:%S')} - {ended.strftime('%H:%M:%S')} · 录像补检疑似喝水事件",
            'summary': summary,
            'start_time': started.isoformat(timespec='seconds'),
            'end_time': ended.isoformat(timespec='seconds'),
            'duration_sec': round(duration, 1),
            'estimated_ml': estimated_ml,
            'confidence': round(confidence, 2),
            'snapshot_url': snapshot_url,
            'tag': '录像补检',
            'debug_curve_url': debug_curve_url,
        }
        detector_mod.post_json('/api/events', payload)
        self.posted_events.append({'id': event_id, 'start_time': payload['start_time'], 'end_time': payload['end_time']})
        self.last_posted_at = end_ts


def replay_frame(detector, frame, ts, settings):
    detector.frame_shape = frame.shape[:2]
    detector.last_frame = frame
    occupied, annotated, score = detector.analyze_frame(frame, settings)
    detector.handle_presence(occupied, score, annotated, ts, settings)
    detector.maybe_finish_event(ts, settings)


def analyze_clip(candidate):
    source_path = Path(candidate.get('resolved_path') or candidate.get('path') or '')
    if not source_path.exists():
        return {'ok': False, 'detail': f'录像文件不存在: {source_path}'}
    clip_start_dt = parse_any_datetime(candidate.get('created_at'))
    if clip_start_dt is None:
        clip_start_dt = datetime.fromtimestamp(source_path.stat().st_mtime).astimezone()
    clip_start_ts = clip_start_dt.timestamp()
    detector_mod.STREAM_URL = str(source_path)
    detector = BackfillDetector(source_label=str(source_path))
    settings = detector_mod.read_settings()
    meta = {'fps': 25.0, 'duration_sec': 0.0, 'width': 0, 'height': 0}
    try:
        meta = ffprobe_meta(source_path)
    except Exception:
        pass

    temp_dir = None
    mode = 'opencv'
    sampled = 0
    last_ts = clip_start_ts
    try:
        if BACKFILL_USE_FFMPEG:
            temp_dir, frames, mode = ffmpeg_extract_frames(source_path, BACKFILL_ANALYZE_FPS)
            if frames:
                for idx, frame_path in enumerate(frames):
                    frame = cv2.imread(str(frame_path))
                    if frame is None:
                        continue
                    ts = clip_start_ts + idx / max(BACKFILL_ANALYZE_FPS, 0.5)
                    last_ts = ts
                    replay_frame(detector, frame, ts, settings)
                    sampled += 1
        if sampled == 0:
            cap = cv2.VideoCapture(str(source_path), cv2.CAP_FFMPEG)
            if not cap.isOpened():
                return {'ok': False, 'detail': f'无法打开录像: {source_path}'}
            fps = meta.get('fps') or cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            sample_step = max(1, int(round(fps / max(BACKFILL_ANALYZE_FPS, 0.5))))
            frame_index = 0
            mode = 'opencv'
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    if frame_index % sample_step != 0:
                        frame_index += 1
                        continue
                    ts = clip_start_ts + frame_index / fps
                    last_ts = ts
                    replay_frame(detector, frame, ts, settings)
                    sampled += 1
                    frame_index += 1
            finally:
                cap.release()
        detector.maybe_finish_event(last_ts + float(settings.get('mergeGapSec', 6) or 6) + 0.2, settings)
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

    duration_sec = meta.get('duration_sec') or (sampled / max(BACKFILL_ANALYZE_FPS, 0.5) if sampled else 0)
    return {
        'ok': True,
        'detail': f'analysis complete via {mode}',
        'events_found': len(detector.posted_events),
        'events': detector.posted_events,
        'sampled_frames': sampled,
        'fps': round(meta.get('fps') or 25.0, 2),
        'duration_sec': round(duration_sec, 1),
        'decode_mode': mode,
        'frame_shape': {'width': meta.get('width', 0), 'height': meta.get('height', 0)},
    }


def worker_loop():
    state = read_json(BACKFILL_STATE_FILE, {'processed': {}, 'last_scan_at': '', 'last_event_at': ''})
    write_status({'ok': True, 'phase': 'starting', 'detail': '录像补检 worker 启动'})
    while True:
        processed = state.setdefault('processed', {})
        cutoff_ts = time.time() - BACKFILL_LOOKBACK_MIN * 60
        cutoff_dt = datetime.fromtimestamp(cutoff_ts).astimezone()
        db_items = discover_candidates_from_db(cutoff_dt)
        dir_items = discover_candidates_from_dir(cutoff_ts)
        candidates = {}
        for item in db_items + dir_items:
            key = item.get('resolved_path') or item.get('path')
            if not key:
                continue
            candidates[key] = {**candidates.get(key, {}), **item}
        pending = []
        for key, item in sorted(candidates.items(), key=lambda kv: kv[1].get('created_at', '')):
            path = Path(item.get('resolved_path') or item.get('path') or '')
            if not path.exists() or not path.is_file():
                continue
            try:
                stat = path.stat()
            except Exception:
                continue
            mark = processed.get(str(path))
            if mark and mark.get('mtime') == stat.st_mtime and mark.get('size') == stat.st_size:
                continue
            if stat.st_size < BACKFILL_MIN_FILE_MB * 1024 * 1024:
                continue
            pending.append((path, item, stat))

        if not candidates:
            write_status({
                'ok': True,
                'phase': 'waiting_recordings',
                'detail': 'EasyNVR 录像目录和数据库中暂未发现可补检录像',
                'last_scan_at': now_iso(),
                'processed_count': len(processed),
                'pending_count': 0,
                'record_db': str(EASYNVR_DB_PATH),
                'record_root': str(EASYNVR_RECORD_ROOT),
                'hwaccel_mode': BACKFILL_HWACCEL_MODE,
            })
        elif not pending:
            write_status({
                'ok': True,
                'phase': 'idle',
                'detail': '暂无新的录像文件需要补检',
                'last_scan_at': now_iso(),
                'processed_count': len(processed),
                'pending_count': 0,
                'record_db': str(EASYNVR_DB_PATH),
                'record_root': str(EASYNVR_RECORD_ROOT),
                'hwaccel_mode': BACKFILL_HWACCEL_MODE,
            })
        else:
            for index, (path, item, stat) in enumerate(pending, start=1):
                write_status({
                    'ok': True,
                    'phase': 'processing',
                    'detail': f'正在补检录像: {path.name}',
                    'last_scan_at': now_iso(),
                    'last_clip': str(path),
                    'processed_count': len(processed),
                    'pending_count': len(pending) - index + 1,
                    'record_db': str(EASYNVR_DB_PATH),
                    'record_root': str(EASYNVR_RECORD_ROOT),
                    'hwaccel_mode': BACKFILL_HWACCEL_MODE,
                })
                result = analyze_clip(item)
                processed[str(path)] = {
                    'mtime': stat.st_mtime,
                    'size': stat.st_size,
                    'processed_at': now_iso(),
                    'source': item.get('source', ''),
                    'events_found': result.get('events_found', 0),
                    'decode_mode': result.get('decode_mode', ''),
                }
                if result.get('events'):
                    state['last_event_at'] = result['events'][-1]['end_time']
                save_state(state)
                write_status({
                    'ok': bool(result.get('ok', False)),
                    'phase': 'processed' if result.get('ok') else 'process_failed',
                    'detail': result.get('detail', ''),
                    'last_scan_at': now_iso(),
                    'last_clip': str(path),
                    'last_event_at': state.get('last_event_at', ''),
                    'processed_count': len(processed),
                    'pending_count': max(0, len(pending) - index),
                    'last_result': result,
                    'record_db': str(EASYNVR_DB_PATH),
                    'record_root': str(EASYNVR_RECORD_ROOT),
                    'hwaccel_mode': BACKFILL_HWACCEL_MODE,
                })
        state['last_scan_at'] = now_iso()
        save_state(state)
        time.sleep(BACKFILL_POLL_SEC)


if __name__ == '__main__':
    worker_loop()
