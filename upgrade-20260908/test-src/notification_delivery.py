import json
import urllib.parse
import urllib.request


PLACEHOLDER_HOSTS = {'openclaw.local'}


def webhook_configured(url):
    try:
        parsed = urllib.parse.urlparse(str(url or '').strip())
    except ValueError:
        return False
    return bool(
        parsed.scheme in ('http', 'https') and parsed.hostname
        and parsed.hostname not in PLACEHOLDER_HOSTS
    )


def validate_webhook_url(url):
    value = str(url or '').strip()
    if not value:
        return ''
    if len(value) > 2048:
        raise ValueError('webhook URL is too long')
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError('webhook URL must use http or https')
    if parsed.username or parsed.password:
        raise ValueError('webhook URL must not contain userinfo')
    return value


def public_webhook_url(url):
    value = str(url or '').strip()
    if not value or not webhook_configured(value):
        return ''
    parsed = urllib.parse.urlparse(value)
    port = f':{parsed.port}' if parsed.port else ''
    return f'{parsed.scheme}://{parsed.hostname}{port}/***'


def delivery_body(url, event):
    host = urllib.parse.urlparse(url).hostname or ''
    message = str(event.get('message') or event.get('title') or '')
    if host.endswith('open.feishu.cn'):
        return {'msg_type': 'text', 'content': {'text': message}}
    return {
        'schema_version': 1,
        'source': 'cat-water-dashboard',
        **event,
    }


def deliver(url, event, timeout=12, opener=urllib.request.urlopen):
    target = validate_webhook_url(url)
    if not webhook_configured(target):
        raise ValueError('webhook is not configured')
    host = urllib.parse.urlparse(target).hostname or ''
    body = json.dumps(delivery_body(target, event), ensure_ascii=False).encode('utf-8')
    request = urllib.request.Request(
        target,
        data=body,
        headers={
            'Content-Type': 'application/json; charset=utf-8',
            'User-Agent': 'cat-water-notifier/1.0',
        },
        method='POST',
    )
    with opener(request, timeout=timeout) as response:
        response_body = response.read(4096).decode('utf-8', 'replace')
        status = int(getattr(response, 'status', 200) or 200)
    if status < 200 or status >= 300:
        raise RuntimeError(f'webhook returned HTTP {status}')
    if host.endswith('open.feishu.cn') and response_body:
        try:
            payload = json.loads(response_body)
        except ValueError:
            payload = {}
        code = payload.get('code', payload.get('StatusCode', 0))
        if code not in (None, 0, '0'):
            message = str(payload.get('msg') or payload.get('StatusMessage') or 'delivery failed')
            raise RuntimeError(f'Feishu webhook rejected request: {message[:160]}')
    return {'status': status, 'response': response_body[:500]}
