#!/usr/bin/env python3
import hashlib
import json
import math
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
BACKFILL_WRITE_EVENTS = os.environ.get('BACKFILL_WRITE_EVENTS', 'false').lower() == 'true'
BACKFILL_EVENT_END_GAP_SEC = float(os.environ.get('BACKFILL_EVENT_END_GAP_SEC', '5'))
BACKFILL_CONFIRMATION_SEC = float(os.environ.get('BACKFILL_CONFIRMATION_SEC', '0.6'))
FFMPEG_BIN = os.environ.get('FFMPEG_BIN', 'ffmpeg')
FFPROBE_BIN = os.environ.get('FFPROBE_BIN', 'ffprobe')

BACKFILL_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
BACKFILL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


def now_iso():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def write_status(payload):
    payload.setdefault('updated_at', now_iso())
    payload.setdefault('updated_unix', time.time())
    _write_json_atomic(BACKFILL_STATUS_FILE, payload)


def _write_json_atomic(path, payload):
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', dir=path.parent,
        prefix=f'.{path.name}.', suffix='.tmp', delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


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
    _write_json_atomic(BACKFILL_STATE_FILE, state)


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


def backfill_confirm_frames(sample_fps=None):
    """Keep confirmation time stable when replay sampling differs from live FPS."""
    fps = float(sample_fps or BACKFILL_ANALYZE_FPS)
    return max(1, int(math.ceil(max(0.0, BACKFILL_CONFIRMATION_SEC) * max(fps, 0.1))))


class BackfillDetector:
    """Replay the same classifier/geometry gates used by the live worker."""

    def __init__(self, source_label=''):
        self.detector = detector_mod.CatDetector()
        self.source_label = source_label
        self.posted_events = []
        self.active_since = None
        self.last_positive_at = None
        self.peak_confidence = 0.0
        self.peak_cat_score = 0.0
        self.peak_frame = None
        self.profile = 'day'
        self.trigger_source = ''
        self._cat_consecutive = 0

    def _detect_drinking(self, frame, settings):
        height, width = frame.shape[:2]
        fountain_zone = detector_mod.get_fountain_zone(settings, width, height)
        water_zone = detector_mod.get_water_zone(settings, width, height)
        result = self.detector.detect(frame, force=True, fzone=fountain_zone, wzone=water_zone)
        cat_box = result.get('cat_box', [])
        night_mode = bool(result.get('night_mode', detector_mod.is_night_mode(frame)))
        has_cat_candidate = bool(cat_box and len(cat_box) == 4)
        self._cat_consecutive = self._cat_consecutive + 1 if has_cat_candidate else 0
        if (not has_cat_candidate or self._cat_consecutive < backfill_confirm_frames() or
                not water_zone):
            return False, 0.0, '', result

        threshold = (detector_mod.NIGHT_DRINKING_CLASSIFIER_THRESHOLD if night_mode
                     else detector_mod.DRINKING_CLASSIFIER_THRESHOLD)
        geometric_floor = (detector_mod.NIGHT_GEOMETRIC_CLASSIFIER_FLOOR if night_mode
                           else detector_mod.GEOMETRIC_CLASSIFIER_FLOOR)
        classifier_drinking = False
        confidence = 0.0
        source = ''
        if self.detector._drink_clf_session is not None:
            predicted, confidence = self.detector.classify_drinking(frame, cat_box, threshold=threshold)
            if predicted:
                mouth_x, mouth_y = detector_mod.estimate_mouth_point(cat_box, water_zone)
                mouth_near = (water_zone[0] - 70 <= mouth_x <= water_zone[2] + 70 and
                              water_zone[1] - 80 <= mouth_y <= water_zone[3] + 90)
                if mouth_near or result.get('closeup_water', False):
                    classifier_drinking = True
                    source = 'classifier' if mouth_near else 'classifier_closeup'

        geometric_drinking = False
        if not classifier_drinking and (self.detector._drink_clf_session is None or confidence >= geometric_floor):
            mouth_x, mouth_y = detector_mod.estimate_mouth_point(cat_box, water_zone)
            water_surface_y = settings.get('waterSurfaceY')
            _, _, posture_ratio = detector_mod.box_metrics(cat_box)
            if water_surface_y:
                scale_y = height / max(1, settings.get('frameHeight', height))
                scaled_water_y = float(water_surface_y) * scale_y
                in_water_x = water_zone[0] - 30 <= mouth_x <= water_zone[2] + 30
                head_at_water = (scaled_water_y - detector_mod.WATER_PROXIMITY_PX * 0.4) <= mouth_y <= (scaled_water_y + detector_mod.WATER_PROXIMITY_PX)
                geometric_drinking = in_water_x and head_at_water and posture_ratio >= detector_mod.DRINKING_POSTURE_MIN_RATIO
                source = 'geometric' if geometric_drinking else source
            else:
                wx1, wy1, wx2, wy2 = water_zone
                ix1, iy1 = max(cat_box[0], wx1), max(cat_box[1], wy1)
                ix2, iy2 = min(cat_box[2], wx2), min(cat_box[3], wy2)
                overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                box_area = max(1, (cat_box[2] - cat_box[0]) * (cat_box[3] - cat_box[1]))
                geometric_drinking = overlap / box_area > 0.15 and posture_ratio >= detector_mod.DRINKING_POSTURE_MIN_RATIO
                source = 'geometric_overlap' if geometric_drinking else source

        return classifier_drinking or geometric_drinking, confidence, source, result

    def observe(self, frame, ts, settings):
        drinking, confidence, source, result = self._detect_drinking(frame, settings)
        if drinking:
            if self.active_since is None:
                self.active_since = ts
            self.last_positive_at = ts
            self.peak_confidence = max(self.peak_confidence, confidence)
            self.peak_cat_score = max(self.peak_cat_score, float(result.get('cat_score', 0) or 0))
            if self.peak_frame is None or confidence >= self.peak_confidence:
                self.peak_frame = frame.copy()
            self.profile = 'night' if result.get('night_mode') else 'day'
            self.trigger_source = source or self.trigger_source
            return
        if self.active_since is not None and self.last_positive_at is not None:
            if ts - self.last_positive_at > BACKFILL_EVENT_END_GAP_SEC:
                self.finalize(ts, settings)

    def finalize(self, ts, settings, force=False):
        if self.active_since is None or self.last_positive_at is None:
            return
        if not force and ts - self.last_positive_at <= BACKFILL_EVENT_END_GAP_SEC:
            return
        start_ts = self.active_since
        end_ts = self.last_positive_at
        duration = max(0.0, end_ts - start_ts)
        required = detector_mod.trigger_min_duration(self.trigger_source or 'unknown', self.profile == 'night')
        if duration >= required:
            digest = hashlib.sha1(f'{self.source_label}:{start_ts:.3f}'.encode('utf-8')).hexdigest()[:10]
            event_id = f"backfill-{datetime.fromtimestamp(start_ts).strftime('%Y%m%d-%H%M%S')}-{digest}"
            snapshot_url = ''
            if BACKFILL_WRITE_EVENTS and self.peak_frame is not None:
                snapshot_path = detector_mod.EVENT_DIR / f'snap-{event_id}.jpg'
                if cv2.imwrite(str(snapshot_path), self.peak_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                    snapshot_url = f'/media/events/{snapshot_path.name}'
            started = datetime.fromtimestamp(start_ts).astimezone().isoformat(timespec='seconds')
            ended = datetime.fromtimestamp(end_ts).astimezone().isoformat(timespec='seconds')
            payload = {
                'event_id': event_id,
                'title': f'录像补检 · {datetime.fromtimestamp(start_ts).strftime("%H:%M:%S")}',
                'summary': f'录像回放已通过实时相同的分类器/几何门槛，持续 {duration:.1f} 秒。',
                'start_time': started,
                'end_time': ended,
                'duration_sec': round(duration, 1),
                'estimated_ml': round(duration * float(settings.get('mlPerSec', 0.14) or 0.14), 1),
                'confidence': self.peak_confidence or self.peak_cat_score,
                'cat_confidence': round(self.peak_cat_score, 3),
                'drinking_confidence': round(self.peak_confidence, 3),
                'snapshot_url': snapshot_url,
                'tag': '录像补检',
                'trigger_source': self.trigger_source,
                'profile': self.profile,
            }
            if BACKFILL_WRITE_EVENTS:
                detector_mod.post_json('/api/events', payload)
            self.posted_events.append({
                'id': event_id,
                'start_time': started,
                'end_time': ended,
                'duration_sec': round(duration, 1),
                'dry_run': not BACKFILL_WRITE_EVENTS,
            })
        self.active_since = None
        self.last_positive_at = None
        self.peak_confidence = 0.0
        self.peak_cat_score = 0.0
        self.peak_frame = None
        self.profile = 'day'
        self.trigger_source = ''


def replay_frame(detector, frame, ts, settings):
    detector.observe(frame, ts, settings)


def analyze_clip(candidate):
    source_path = Path(candidate.get('resolved_path') or candidate.get('path') or '')
    if not source_path.exists():
        return {'ok': False, 'detail': f'录像文件不存在: {source_path}'}
    clip_start_dt = parse_any_datetime(candidate.get('created_at'))
    if clip_start_dt is None:
        clip_start_dt = datetime.fromtimestamp(source_path.stat().st_mtime).astimezone()
    clip_start_ts = clip_start_dt.timestamp()
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
        detector.finalize(last_ts + BACKFILL_EVENT_END_GAP_SEC + 0.2, settings, force=True)
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

    duration_sec = meta.get('duration_sec') or (sampled / max(BACKFILL_ANALYZE_FPS, 0.5) if sampled else 0)
    return {
        'ok': True,
        'detail': f'analysis complete via {mode}',
        'events_found': len(detector.posted_events) if BACKFILL_WRITE_EVENTS else 0,
        'candidates_found': len(detector.posted_events),
        'write_events': BACKFILL_WRITE_EVENTS,
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
            if (mark and mark.get('mtime') == stat.st_mtime and mark.get('size') == stat.st_size and
                    bool(mark.get('write_events', False)) == BACKFILL_WRITE_EVENTS):
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
                'write_events': BACKFILL_WRITE_EVENTS,
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
                'write_events': BACKFILL_WRITE_EVENTS,
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
                    'write_events': BACKFILL_WRITE_EVENTS,
                })
                result = analyze_clip(item)
                processed[str(path)] = {
                    'mtime': stat.st_mtime,
                    'size': stat.st_size,
                    'processed_at': now_iso(),
                    'source': item.get('source', ''),
                    'events_found': result.get('events_found', 0),
                    'candidates_found': result.get('candidates_found', 0),
                    'write_events': bool(result.get('write_events', False)),
                    'decode_mode': result.get('decode_mode', ''),
                }
                if result.get('write_events') and result.get('events'):
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
                    'write_events': BACKFILL_WRITE_EVENTS,
                })
        state['last_scan_at'] = now_iso()
        save_state(state)
        time.sleep(BACKFILL_POLL_SEC)


if __name__ == '__main__':
    worker_loop()
