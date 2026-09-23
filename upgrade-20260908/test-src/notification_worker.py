#!/usr/bin/env python3
import json
import os
import tempfile
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from notification_delivery import deliver, webhook_configured


DASHBOARD_API_URL = os.environ.get('DASHBOARD_API_URL', 'http://cat-water-dashboard:39095')
STATUS_FILE = Path(os.environ.get('NOTIFICATION_STATUS_FILE', '/app/data/notification_status.json'))
STATE_FILE = Path(os.environ.get('NOTIFICATION_STATE_FILE', '/app/data/notification_state.json'))
PUSH_FILE = Path(os.environ.get('NOTIFICATION_PUSH_FILE', '/app/data/push.json'))
POLL_SEC = float(os.environ.get('NOTIFICATION_POLL_SEC', '60'))
MAX_RETRIES = int(os.environ.get('NOTIFICATION_MAX_RETRIES', '5'))
DAILY_WINDOW_HOURS = float(os.environ.get('NOTIFICATION_DAILY_WINDOW_HOURS', '4'))
CST = timezone(timedelta(hours=8))


def now_iso():
    return datetime.now(CST).isoformat(timespec='seconds')


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


def load_state():
    try:
        payload = json.loads(STATE_FILE.read_text(encoding='utf-8'))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def load_push_config():
    try:
        payload = json.loads(PUSH_FILE.read_text(encoding='utf-8'))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def request_json(path, timeout=15):
    request = urllib.request.Request(
        f'{DASHBOARD_API_URL}{path}', headers={'User-Agent': 'cat-water-notifier/1.0'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def parse_datetime(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return parsed.replace(tzinfo=CST) if parsed.tzinfo is None else parsed.astimezone(CST)
    except ValueError:
        return None


def parse_clock(value):
    try:
        hour, minute = str(value or '21:30').split(':', 1)
        return max(0, min(23, int(hour))), max(0, min(59, int(minute)))
    except (TypeError, ValueError):
        return 21, 30


def daily_window_open(now, clock, sent_keys):
    hour, minute = parse_clock(clock)
    scheduled = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    key = f'daily:{now.date().isoformat()}'
    return key, scheduled <= now <= scheduled + timedelta(hours=DAILY_WINDOW_HOURS) and key not in sent_keys


def previous_daily_average(trends, today):
    values = []
    for label, count, amount in zip(
        trends.get('labels', []), trends.get('counts', []), trends.get('ml', []),
    ):
        if label < today.isoformat():
            values.append((float(count or 0), float(amount or 0)))
    values = values[-7:]
    if not values:
        return 0.0, 0.0
    return (
        sum(value[0] for value in values) / len(values),
        sum(value[1] for value in values) / len(values),
    )


def render_template(template, overview):
    values = {
        'drink_count': overview.get('drinkCount', 0),
        'estimated_total_ml': overview.get('waterMl', 0),
        'last_drink_time': overview.get('lastDrink', '--:--'),
    }
    result = str(template or '')
    for key, value in values.items():
        result = result.replace('{{' + key + '}}', str(value))
    return result.strip()


def actionable_review_candidates(queue):
    resolved = {'confirmed', 'rejected', 'covered_confirmed', 'covered_rejected'}
    if not isinstance(queue, dict):
        return []
    return [
        item for item in queue.get('items', [])
        if isinstance(item, dict) and not item.get('label')
        and (item.get('recovery_status') or '') not in resolved
        and item.get('source') == 'nightly_archive_audit'
    ]


def build_scheduled_events(now, settings, push, dashboard, state, review_queue=None):
    overview = dashboard.get('overview', {})
    trends = dashboard.get('trends', {})
    sent = state.get('sent', {})
    failed = state.get('failed', {})
    queued = {item.get('dedupe_key') for item in state.get('queue', [])}
    events = []
    push_mode = str(push.get('pushMode', '日报 + 异常提醒'))
    daily_enabled = (
        settings.get('enableDailyReport', True) and push.get('enableFeishuDaily', False)
        and push_mode != '仅异常提醒'
    )
    alert_enabled = push_mode != '仅日报'
    daily_key, daily_due = daily_window_open(now, push.get('dailyReportTime'), sent)
    if daily_enabled and daily_due and daily_key not in queued and daily_key not in failed:
        message = render_template(push.get('summaryTemplate'), overview)
        events.append({
            'type': 'daily_report', 'dedupe_key': daily_key,
            'title': f'{now:%Y-%m-%d} 猫咪饮水日报',
            'message': message or f'今日饮水 {overview.get("drinkCount", 0)} 次，视频估算 {overview.get("waterMl", 0)} ml。',
            'data': {'overview': overview},
        })

    last_drink = parse_datetime(overview.get('lastDrinkAt'))
    no_drink_hours = max(1.0, float(push.get('noDrinkHours', 8) or 8))
    cooldown_hours = max(1.0, float(push.get('alertCooldownHours', 6) or 6))
    if (alert_enabled and settings.get('enableNoDrinkAlert', True)
            and push.get('enableFeishuAlert', False) and last_drink):
        elapsed_hours = (now - last_drink).total_seconds() / 3600
        bucket = int(elapsed_hours // cooldown_hours)
        key = f'no_drink:{last_drink.isoformat()}:{bucket}'
        if (elapsed_hours >= no_drink_hours and key not in sent
                and key not in failed and key not in queued):
            events.append({
                'type': 'no_drink_alert', 'dedupe_key': key,
                'title': '猫咪长时间未饮水提醒',
                'message': f'距离最近一次已确认饮水约 {elapsed_hours:.1f} 小时，请检查饮水机和猫咪状态。',
                'data': {'last_drink_at': last_drink.isoformat(), 'elapsed_hours': round(elapsed_hours, 1)},
            })

    low_key = f'low_intake:{now.date().isoformat()}'
    if (alert_enabled and push.get('enableLowIntakeAlert', False) and daily_due
            and low_key not in sent and low_key not in failed and low_key not in queued):
        avg_count, avg_ml = previous_daily_average(trends, now.date())
        threshold = max(1.0, min(99.0, float(push.get('lowCountPercent', 30) or 30)))
        cutoff = avg_count * (1 - threshold / 100)
        current = float(overview.get('drinkCount', 0) or 0)
        if avg_count >= 2 and current < cutoff:
            events.append({
                'type': 'low_intake_alert', 'dedupe_key': low_key,
                'title': '猫咪饮水偏低提醒',
                'message': f'今日已确认饮水 {current:g} 次，低于近 7 日均值 {avg_count:.1f} 次；请结合精神、食欲和饮水机状态观察。',
                'data': {'today_count': current, 'average_count': round(avg_count, 1), 'average_ml': round(avg_ml, 1)},
            })

    if alert_enabled and push.get('enableReviewCandidateAlert', True):
        for candidate in actionable_review_candidates(review_queue):
            candidate_id = str(candidate.get('id', ''))
            key = f'review_candidate:{candidate_id}'
            if not candidate_id or key in sent or key in failed or key in queued:
                continue
            events.append({
                'type': 'review_candidate_alert',
                'dedupe_key': key,
                'title': '发现需要复核的饮水候选',
                'message': '每日录像巡检发现一段未被实时链路覆盖的高可信候选，请打开样本审核页确认。',
                'data': {
                    'sample_id': candidate_id,
                    'created_at': candidate.get('created_at', ''),
                    'classifier_confidence': candidate.get('classifier_confidence'),
                    'review_url': '/review/',
                },
            })
    return events


def enqueue(state, events, now):
    queue = state.setdefault('queue', [])
    for event in events:
        queue.append({
            **event,
            'attempts': 0,
            'next_attempt_at': now.timestamp(),
            'created_at': now.isoformat(timespec='seconds'),
        })


def process_queue(state, push, now):
    queue = state.setdefault('queue', [])
    sent = state.setdefault('sent', {})
    failed = state.setdefault('failed', {})
    remaining = []
    delivered = 0
    last_error = ''
    retry_delays = (60, 300, 900, 3600, 10800)
    for item in queue:
        if float(item.get('next_attempt_at', 0) or 0) > now.timestamp():
            remaining.append(item)
            continue
        event = {key: value for key, value in item.items()
                 if key not in ('attempts', 'next_attempt_at', 'created_at', 'dedupe_key')}
        try:
            deliver(push.get('openclawWebhook', ''), event)
            sent[item['dedupe_key']] = now.isoformat(timespec='seconds')
            state['last_success_at'] = now.isoformat(timespec='seconds')
            state['last_success_type'] = item.get('type', '')
            delivered += 1
        except Exception as exc:
            attempts = int(item.get('attempts', 0)) + 1
            last_error = str(exc)[:240]
            state['last_error'] = last_error
            state['last_error_at'] = now.isoformat(timespec='seconds')
            if attempts < MAX_RETRIES:
                item['attempts'] = attempts
                item['next_attempt_at'] = now.timestamp() + retry_delays[min(attempts - 1, len(retry_delays) - 1)]
                remaining.append(item)
            else:
                failed[item['dedupe_key']] = {
                    'failed_at': now.isoformat(timespec='seconds'),
                    'attempts': attempts,
                    'error': last_error,
                }
    state['queue'] = remaining
    if len(sent) > 500:
        state['sent'] = dict(sorted(sent.items(), key=lambda item: item[1])[-500:])
    if len(failed) > 500:
        state['failed'] = dict(sorted(
            failed.items(), key=lambda item: item[1].get('failed_at', ''))[-500:])
    if delivered and not last_error:
        state.pop('last_error', None)
        state.pop('last_error_at', None)
    return delivered, last_error


def status_error(state, error=''):
    if error:
        return error
    stored = str(state.get('last_error') or '')
    if not stored:
        return ''
    success_at = parse_datetime(state.get('last_success_at'))
    error_at = parse_datetime(state.get('last_error_at'))
    if success_at and error_at and success_at >= error_at:
        return ''
    return stored


def write_status(state, phase, configured, error=''):
    write_json_atomic(STATUS_FILE, {
        'ok': not bool(error),
        'phase': phase,
        'configured': configured,
        'pending': len(state.get('queue', [])),
        'failed': len(state.get('failed', {})),
        'last_success_at': state.get('last_success_at', ''),
        'last_success_type': state.get('last_success_type', ''),
        'last_error': status_error(state, error),
        'last_error_at': state.get('last_error_at', ''),
        'updated_at': now_iso(),
        'updated_unix': time.time(),
    })


def run_forever():
    state = load_state()
    while True:
        try:
            settings = request_json('/api/settings')
            push = load_push_config()
            configured = webhook_configured(push.get('openclawWebhook', ''))
            if not configured:
                write_status(state, 'waiting_configuration', False)
                time.sleep(max(10, POLL_SEC))
                continue
            dashboard = request_json('/api/dashboard/overview')
            review_queue = request_json('/api/review-candidates')
            now = datetime.now(CST)
            enqueue(state, build_scheduled_events(
                now, settings, push, dashboard, state, review_queue), now)
            delivered, error = process_queue(state, push, now)
            write_json_atomic(STATE_FILE, state)
            phase = 'retrying' if state.get('queue') else ('delivered' if delivered else 'idle')
            write_status(state, phase, True, error if state.get('queue') else '')
        except Exception as exc:
            write_status(state, 'poll_failed', False, str(exc)[:240])
        time.sleep(max(10, POLL_SEC))


if __name__ == '__main__':
    run_forever()
