import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    path = ROOT / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimeSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('DETECTOR_EVENT_DIR', '/tmp/cat-water-test-events')
        os.environ.setdefault('DETECTOR_STATUS_FILE', '/tmp/cat-water-test-status.json')
        cls.stream_worker = load_module('hd_live_worker')
        cls.camera_worker = load_module('camera_worker')
        cls.detector_worker = load_module('detection_worker')
        cls.server = load_module('server')
        cls.archive_verifier = load_module('archive_verifier_worker')
        cls.notification_delivery = load_module('notification_delivery')
        cls.notification_worker = load_module('notification_worker')
        cls.daily_audit = load_module('daily_audit_worker')
        cls.classifier_calibration = load_module('classifier_calibration_worker')

    def test_http_input_has_bounded_read_timeout(self):
        previous = self.stream_worker.RTSP_URL
        self.stream_worker.RTSP_URL = 'http://example.invalid/live.flv'
        try:
            command = self.stream_worker.base_input_cmd()
        finally:
            self.stream_worker.RTSP_URL = previous
        self.assertIn('-rw_timeout', command)
        self.assertNotIn('-rtsp_transport', command)

    def test_rtsp_input_forces_tcp(self):
        previous = self.stream_worker.RTSP_URL
        self.stream_worker.RTSP_URL = 'rtsp://example.invalid/stream'
        try:
            command = self.stream_worker.base_input_cmd()
        finally:
            self.stream_worker.RTSP_URL = previous
        self.assertIn('-rtsp_transport', command)
        self.assertEqual(command[command.index('-rtsp_transport') + 1], 'tcp')

    def test_missing_stream_configuration_is_rejected(self):
        previous = self.stream_worker.RTSP_URL
        self.stream_worker.RTSP_URL = ''
        try:
            with self.assertRaisesRegex(RuntimeError, 'RTSP_URL'):
                self.stream_worker.base_input_cmd()
        finally:
            self.stream_worker.RTSP_URL = previous

    def test_stream_url_masks_userinfo(self):
        safe = self.camera_worker.safe_stream_url('rtsp://user:password@example.invalid:554/stream')
        self.assertNotIn('password', safe)
        self.assertIn('***@example.invalid', safe)

    def test_stale_status_is_not_reported_as_healthy(self):
        status = self.server._status_with_freshness(
            {'ok': True, 'phase': 'running', 'updated_unix': 0},
            stale_after_sec=10,
            missing_phase='stale',
        )
        self.assertTrue(status['stale'])
        self.assertFalse(status['ok'])
        self.assertEqual(status['phase'], 'stale')

    def test_zone_validation_rejects_empty_area(self):
        with self.assertRaisesRegex(ValueError, 'positive area'):
            self.server.normalize_zone({'x1': 4, 'y1': 2, 'x2': 4, 'y2': 8}, 100, 100)

    def test_zone_validation_accepts_bounded_rectangle(self):
        self.assertEqual(
            self.server.normalize_zone({'x1': 4, 'y1': 2, 'x2': 20, 'y2': 8}, 100, 100),
            {'x1': 4, 'y1': 2, 'x2': 20, 'y2': 8},
        )

    def test_merge_keeps_all_event_media_and_sources(self):
        base = {
            'id': 'first', 'start_time': '2026-07-11T08:00:00+08:00',
            'end_time': '2026-07-11T08:00:05+08:00', 'estimatedMl': 1,
            'metric': '5s', 'img': '/media/events/snap-first.jpg',
            'crop': '/media/events/crop-first.jpg',
            'clip': '/media/events/first.mp4', 'evidenceUrl': '/media/events/evidence-first.json',
        }
        incoming = {
            'id': 'second', 'start_time': '2026-07-11T08:00:08+08:00',
            'end_time': '2026-07-11T08:00:13+08:00', 'estimatedMl': 1,
            'metric': '5s', 'img': '/media/events/snap-second.jpg',
            'crop': '/media/events/crop-second.jpg',
            'clip': '/media/events/second.mp4', 'evidenceUrl': '/media/events/evidence-second.json',
        }
        merged = self.server.merge_event_pair(base, incoming)
        self.assertEqual(merged['clips'], ['/media/events/first.mp4', '/media/events/second.mp4'])
        self.assertEqual(merged['crops'], ['/media/events/crop-first.jpg', '/media/events/crop-second.jpg'])
        self.assertEqual(merged['evidenceUrls'], [
            '/media/events/evidence-first.json', '/media/events/evidence-second.json',
        ])
        self.assertEqual(merged['snapshots'], ['/media/events/snap-first.jpg', '/media/events/snap-second.jpg'])
        self.assertEqual(merged['sourceEventIds'], ['first', 'second'])
        self.assertEqual(merged['metric'], '10s')
        merged_without_metric = self.server.merge_event_pair({
            'id': 'third', 'start_time': '2026-07-11T09:00:00+08:00',
            'end_time': '2026-07-11T09:00:05+08:00',
        }, {
            'id': 'fourth', 'start_time': '2026-07-11T09:00:08+08:00',
            'end_time': '2026-07-11T09:00:13+08:00',
        })
        self.assertEqual(merged_without_metric['metric'], '10s')

    def test_settings_reject_unknown_fields_and_model_paths(self):
        with self.assertRaisesRegex(ValueError, 'unsupported settings'):
            self.server.normalize_settings_update({}, {'activeModel': 'anything.onnx'})
        with self.assertRaisesRegex(ValueError, 'without a path'):
            self.server.resolve_model_path('../server.py')
        with self.assertRaisesRegex(ValueError, 'must not exceed'):
            self.server.normalize_settings_update(
                self.server.DEFAULT_SETTINGS,
                {'mlPerActiveSecLow': 0.8, 'mlPerActiveSecHigh': 0.2},
            )

    def test_volume_calibration_uses_only_trusted_v2_events(self):
        data = {'overview': {}, 'trends': {}, 'events': [
            {
                'id': 'confirmed', 'start_time': '2026-07-13T08:10:00+08:00',
                'end_time': '2026-07-13T08:10:10+08:00', 'activeDrinkSec': 10,
                'volumeMethod': 'active_lapping_v2', 'label': 'confirmed',
            },
            {
                'id': 'verified', 'start_time': '2026-07-13T09:10:00+08:00',
                'end_time': '2026-07-13T09:10:10+08:00', 'activeDrinkSec': 10,
                'volumeMethod': 'archive_anchor_clusters_v2', 'verificationStatus': 'confirmed',
            },
            {
                'id': 'rejected', 'start_time': '2026-07-13T10:10:00+08:00',
                'end_time': '2026-07-13T10:10:10+08:00', 'activeDrinkSec': 100,
                'volumeMethod': 'active_lapping_v2', 'label': 'false_positive',
            },
        ]}
        result = self.server.calculate_volume_calibration({
            'start_time': '2026-07-13T08:00:00+08:00',
            'end_time': '2026-07-13T11:00:00+08:00',
            'start_weight_g': 1000,
            'end_weight_g': 991,
        }, data)
        self.assertEqual(result['event_count'], 2)
        self.assertEqual(result['active_drink_sec'], 20)
        self.assertEqual(result['calculated_ml_per_active_sec'], 0.45)
        self.assertTrue(result['can_apply'])

    def test_volume_calibration_applies_rate_and_recalculates_v2_history(self):
        server = self.server
        previous = (server.SETTINGS_FILE, server.DATA_FILE, server.VOLUME_CALIBRATIONS_FILE)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server.SETTINGS_FILE = root / 'settings.json'
            server.DATA_FILE = root / 'mock.json'
            server.VOLUME_CALIBRATIONS_FILE = root / 'volume_calibrations.json'
            try:
                server.write_json(server.SETTINGS_FILE, dict(server.DEFAULT_SETTINGS))
                server.write_json(server.DATA_FILE, {'overview': {}, 'trends': {}, 'events': [{
                    'id': 'event-1', 'start_time': '2026-07-13T08:10:00+08:00',
                    'end_time': '2026-07-13T08:10:20+08:00', 'activeDrinkSec': 20,
                    'volumeMethod': 'archive_anchor_clusters_v2', 'label': 'confirmed',
                    'estimatedMl': 9,
                }]})
                result = server.apply_volume_calibration({
                    'start_time': '2026-07-13T08:00:00+08:00',
                    'end_time': '2026-07-13T09:00:00+08:00',
                    'start_weight_g': 1000,
                    'end_weight_g': 990,
                    'recalculate_history': True,
                })
                self.assertEqual(result['settings']['mlPerActiveSec'], 0.5)
                self.assertEqual(result['recalculated_event_count'], 1)
                event = server.load_dashboard_data()['events'][0]
                self.assertEqual(event['estimatedMl'], 10.0)
                self.assertEqual(event['estimatedMlLow'], 7.5)
                self.assertEqual(event['estimatedMlHigh'], 12.5)
                self.assertTrue(server.read_json(server.VOLUME_CALIBRATIONS_FILE, {})['items'])
            finally:
                server.SETTINGS_FILE, server.DATA_FILE, server.VOLUME_CALIBRATIONS_FILE = previous

    def test_push_settings_are_validated_and_webhook_is_masked(self):
        server = self.server
        current = dict(server.DEFAULT_PUSH)
        updated = server.normalize_push_update(current, {
            'openclawWebhook': 'https://example.invalid/hooks/secret',
            'pushMode': '仅异常提醒',
            'dailyReportTime': '22:15',
            'noDrinkHours': 10,
        })
        self.assertTrue(server.public_push(updated)['webhookConfigured'])
        self.assertEqual(server.public_push(updated)['openclawWebhook'], 'https://example.invalid/***')
        with self.assertRaisesRegex(ValueError, 'dailyReportTime'):
            server.normalize_push_update(current, {'dailyReportTime': '25:00'})
        with self.assertRaisesRegex(ValueError, 'userinfo'):
            server.normalize_push_update(current, {
                'openclawWebhook': 'https://user:secret@example.invalid/hook',
            })

    def test_webhook_delivery_uses_generic_json_without_exposing_target(self):
        delivery = self.notification_delivery
        captured = {}

        class Response:
            status = 204

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, _):
                return b''

        def opener(request, timeout=0):
            captured['body'] = json.loads(request.data.decode('utf-8'))
            captured['timeout'] = timeout
            return Response()

        result = delivery.deliver(
            'https://example.invalid/hook',
            {'type': 'test', 'title': 'title', 'message': 'hello'},
            opener=opener,
        )
        self.assertEqual(result['status'], 204)
        self.assertEqual(captured['body']['source'], 'cat-water-dashboard')
        self.assertEqual(captured['body']['message'], 'hello')
        self.assertGreater(captured['timeout'], 0)

    def test_notification_scheduler_deduplicates_daily_and_no_drink_alerts(self):
        worker = self.notification_worker
        now = worker.datetime(2026, 7, 12, 22, 0, tzinfo=worker.CST)
        settings = {'enableDailyReport': True, 'enableNoDrinkAlert': True}
        push = {
            'pushMode': '日报 + 异常提醒', 'enableFeishuDaily': True,
            'enableFeishuAlert': True, 'enableLowIntakeAlert': False,
            'dailyReportTime': '21:30', 'noDrinkHours': 8,
            'alertCooldownHours': 6, 'summaryTemplate': '{{drink_count}} / {{estimated_total_ml}}',
        }
        dashboard = {'overview': {
            'drinkCount': 2, 'waterMl': 12.5, 'lastDrink': '10:00:00',
            'lastDrinkAt': '2026-07-12T10:00:00+08:00',
        }, 'trends': {}}
        events = worker.build_scheduled_events(now, settings, push, dashboard, {'sent': {}, 'queue': []})
        self.assertEqual({event['type'] for event in events}, {'daily_report', 'no_drink_alert'})
        state = {'sent': {}, 'queue': []}
        worker.enqueue(state, events, now)
        self.assertEqual(worker.build_scheduled_events(now, settings, push, dashboard, state), [])

    def test_notification_queue_retries_then_marks_delivery(self):
        worker = self.notification_worker
        now = worker.datetime(2026, 7, 12, 22, 0, tzinfo=worker.CST)
        state = {'queue': [{
            'type': 'test', 'title': 'x', 'message': 'y', 'dedupe_key': 'test:1',
            'attempts': 0, 'next_attempt_at': now.timestamp(), 'created_at': now.isoformat(),
        }], 'sent': {}}
        previous = worker.deliver
        try:
            worker.deliver = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('offline'))
            delivered, error = worker.process_queue(state, {'openclawWebhook': 'https://example.invalid'}, now)
            self.assertEqual(delivered, 0)
            self.assertIn('offline', error)
            self.assertEqual(state['queue'][0]['attempts'], 1)
            worker.deliver = lambda *_args, **_kwargs: {'status': 200}
            state['queue'][0]['next_attempt_at'] = now.timestamp()
            delivered, error = worker.process_queue(state, {'openclawWebhook': 'https://example.invalid'}, now)
            self.assertEqual((delivered, error), (1, ''))
            self.assertEqual(state['queue'], [])
            self.assertIn('test:1', state['sent'])
        finally:
            worker.deliver = previous

    def test_notification_queue_stops_after_max_retries(self):
        worker = self.notification_worker
        now = worker.datetime(2026, 7, 12, 21, 30, tzinfo=worker.CST)
        state = {'queue': [{
            'type': 'daily_report', 'dedupe_key': 'daily:2026-07-12',
            'title': '日报', 'message': '测试', 'attempts': worker.MAX_RETRIES - 1,
            'next_attempt_at': 0, 'created_at': now.isoformat(),
        }], 'sent': {}}
        previous = worker.deliver
        try:
            worker.deliver = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('offline'))
            delivered, error = worker.process_queue(
                state, {'openclawWebhook': 'https://example.invalid'}, now)
        finally:
            worker.deliver = previous
        self.assertEqual(delivered, 0)
        self.assertEqual(error, 'offline')
        self.assertEqual(state['queue'], [])
        self.assertEqual(state['failed']['daily:2026-07-12']['attempts'], worker.MAX_RETRIES)
        events = worker.build_scheduled_events(
            now,
            {'enableDailyReport': True, 'enableNoDrinkAlert': False},
            {'pushMode': '日报 + 异常提醒', 'enableFeishuDaily': True,
             'enableFeishuAlert': False, 'enableLowIntakeAlert': False,
             'dailyReportTime': '21:30'},
            {'overview': {}, 'trends': {}}, state,
        )
        self.assertEqual(events, [])

    def test_feishu_delivery_rejects_business_error(self):
        delivery = self.notification_delivery

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, _):
                return b'{"StatusCode": 19001, "StatusMessage": "invalid token"}'

        with self.assertRaisesRegex(RuntimeError, 'invalid token'):
            delivery.deliver(
                'https://open.feishu.cn/open-apis/bot/v2/hook/test',
                {'type': 'test', 'message': 'hello'}, opener=lambda *_args, **_kwargs: Response(),
            )

    def test_event_upsert_persists_active_lapping_volume_v2(self):
        server = self.server
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = {'DATA_FILE': server.DATA_FILE, 'SETTINGS_FILE': server.SETTINGS_FILE}
            server.DATA_FILE = root / 'mock.json'
            server.SETTINGS_FILE = root / 'settings.json'
            server.write_json(server.DATA_FILE, {'events': [], 'overview': {}})
            server.write_json(server.SETTINGS_FILE, {
                'mlPerActiveSec': 0.45,
                'mlPerActiveSecLow': 0.30,
                'mlPerActiveSecHigh': 0.70,
            })
            try:
                event, _ = server.upsert_event({
                    'event_id': 'active-v2',
                    'start_time': '2026-07-12T08:00:00+08:00',
                    'end_time': '2026-07-12T08:00:05+08:00',
                    'duration_sec': 5.0,
                    'active_drink_sec': 2.8,
                    'estimated_ml': 1.3,
                    'ml_per_active_sec': 0.45,
                    'volume_method': 'active_lapping_v2',
                })
                self.assertEqual(event['metric'], '5s')
                self.assertEqual(event['activeDrinkSec'], 2.8)
                self.assertEqual(event['estimatedMl'], 1.3)
                self.assertEqual(event['estimatedMlLow'], 0.8)
                self.assertEqual(event['estimatedMlHigh'], 2.0)
                self.assertEqual(event['volumeMethod'], 'active_lapping_v2')
                self.assertEqual(event['verificationStatus'], 'pending')
                self.assertFalse(server.event_counts_for_stats(event))
                event = server.update_event_verification('active-v2', {
                    'status': 'confirmed', 'anchors': 10, 'active_drink_sec': 14.7,
                })
                self.assertEqual(event['activeDrinkSec'], 14.7)
                self.assertEqual(event['estimatedMl'], 6.6)
                self.assertEqual(event['volumeMethod'], 'archive_anchor_clusters_v2')
                self.assertIn('14.7 秒', event['summary'])
            finally:
                for name, value in previous.items():
                    setattr(server, name, value)

    def test_temporal_classifier_requires_repeated_high_confidence_hits(self):
        gate = self.detector_worker.TemporalClassifierGate(
            threshold=0.9, min_hits=2, window_sec=1.0, hold_sec=2.0)
        self.assertFalse(gate.observe(0.0, True, True, 0.95))
        self.assertFalse(gate.observe(0.2, True, False, 0.99))
        self.assertTrue(gate.observe(0.6, True, True, 0.96))
        self.assertEqual(gate.last_confidence, 0.96)
        self.assertTrue(gate.observe(1.5, False, False, 0.0))
        self.assertFalse(gate.observe(2.7, False, False, 0.0))
        self.assertEqual(gate.last_confidence, 0.0)

    def test_temporal_classifier_rejects_spatially_unrelated_hits(self):
        gate = self.detector_worker.TemporalClassifierGate(
            threshold=0.9, min_hits=2, window_sec=1.0, hold_sec=2.0, min_iou=0.2)
        self.assertFalse(gate.observe(0.0, True, True, 0.96, [0, 0, 100, 100]))
        self.assertFalse(gate.observe(0.2, True, True, 0.97, [200, 200, 300, 300]))
        self.assertTrue(gate.observe(0.4, True, True, 0.98, [205, 205, 305, 305]))

    def test_drinking_candidate_duration_resets_after_gap(self):
        gate = self.detector_worker.DrinkingCandidateGate(gap_sec=0.8)
        self.assertEqual(gate.observe(0.0, True), 0.0)
        self.assertEqual(gate.observe(0.4, False), 0.0)
        self.assertIsNone(gate.observe(0.9, False))
        self.assertEqual(gate.observe(1.0, True), 1.0)

    def test_active_drink_evidence_excludes_temporal_hold_and_unrelated_boxes(self):
        evidence = self.detector_worker.ActiveDrinkEvidence(
            max_gap_sec=2.0, min_iou=0.2, max_credit_sec=0.5)
        box = [100, 10, 800, 300]
        evidence.observe(0.0, True, box)
        evidence.observe(0.4, True, [105, 10, 805, 300])
        evidence.observe(0.8, True, [110, 12, 810, 302])
        evidence.observe(3.3, False, box)
        self.assertAlmostEqual(evidence.duration(), 1.2)
        evidence.observe(4.0, True, [900, 300, 1100, 500])
        self.assertAlmostEqual(evidence.duration(), 1.2)
        evidence.observe(4.4, True, [905, 302, 1105, 502])
        self.assertAlmostEqual(evidence.duration(), 2.0)

    def test_review_recorder_persists_and_deduplicates_samples(self):
        worker = self.detector_worker
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = {
                'REVIEW_SAMPLE_DIR': worker.REVIEW_SAMPLE_DIR,
                'REVIEW_SAMPLE_FILE': worker.REVIEW_SAMPLE_FILE,
                'REVIEW_SAMPLE_ENABLED': worker.REVIEW_SAMPLE_ENABLED,
                'REVIEW_SAMPLE_COOLDOWN_SEC': worker.REVIEW_SAMPLE_COOLDOWN_SEC,
                'REVIEW_SAMPLE_MAX_ITEMS': worker.REVIEW_SAMPLE_MAX_ITEMS,
            }
            worker.REVIEW_SAMPLE_DIR = root / 'media'
            worker.REVIEW_SAMPLE_FILE = root / 'review.json'
            worker.REVIEW_SAMPLE_ENABLED = True
            worker.REVIEW_SAMPLE_COOLDOWN_SEC = 0
            worker.REVIEW_SAMPLE_MAX_ITEMS = 2
            try:
                recorder = worker.ReviewSampleRecorder()
                frame = worker.np.zeros((120, 160, 3), dtype=worker.np.uint8)
                frame[20:100, 30:130] = (30, 120, 220)
                self.assertTrue(recorder.capture(
                    frame, [30, 20, 130, 100], 0.5, 0.4, False))
                self.assertFalse(recorder.capture(
                    frame, [30, 20, 130, 100], 0.6, 0.5, False))
                payload = json.loads(worker.REVIEW_SAMPLE_FILE.read_text())
                self.assertEqual(len(payload['items']), 1)
                self.assertTrue((worker.REVIEW_SAMPLE_DIR / (
                    payload['items'][0]['id'] + '-crop.jpg')).is_file())
            finally:
                for name, value in previous.items():
                    setattr(worker, name, value)

    def test_review_labels_are_stored_separately_from_candidates(self):
        server = self.server
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = {
                'REVIEW_CANDIDATES_FILE': server.REVIEW_CANDIDATES_FILE,
                'REVIEW_LABELS_FILE': server.REVIEW_LABELS_FILE,
            }
            server.REVIEW_CANDIDATES_FILE = root / 'candidates.json'
            server.REVIEW_LABELS_FILE = root / 'labels.json'
            candidate_payload = {'version': 1, 'items': [{'id': 'sample-1'}]}
            server.REVIEW_CANDIDATES_FILE.write_text(json.dumps(candidate_payload))
            original = server.REVIEW_CANDIDATES_FILE.read_bytes()
            try:
                annotation = server.label_review_candidate('sample-1', 'not_drinking')
                self.assertEqual(annotation['label'], 'not_drinking')
                queue = server.load_review_candidates()
                self.assertEqual(queue['items'][0]['label'], 'not_drinking')
                self.assertEqual(queue['counts']['not_drinking'], 1)
                self.assertEqual(server.REVIEW_CANDIDATES_FILE.read_bytes(), original)
                with self.assertRaises(KeyError):
                    server.label_review_candidate('missing', 'no_cat')
            finally:
                for name, value in previous.items():
                    setattr(server, name, value)

    def test_review_covered_status_tracks_linked_event(self):
        server = self.server
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = {
                'DATA_FILE': server.DATA_FILE,
                'REVIEW_CANDIDATES_FILE': server.REVIEW_CANDIDATES_FILE,
                'DAILY_AUDIT_CANDIDATES_FILE': server.DAILY_AUDIT_CANDIDATES_FILE,
                'REVIEW_LABELS_FILE': server.REVIEW_LABELS_FILE,
            }
            server.DATA_FILE = root / 'mock.json'
            server.REVIEW_CANDIDATES_FILE = root / 'review.json'
            server.DAILY_AUDIT_CANDIDATES_FILE = root / 'audit.json'
            server.REVIEW_LABELS_FILE = root / 'labels.json'
            server.write_json(server.DATA_FILE, {'events': [{
                'id': 'event-current', 'sourceEventIds': ['event-old'],
                'verificationStatus': 'confirmed',
            }]})
            server.write_json(server.REVIEW_CANDIDATES_FILE, {'items': [{'id': 'sample'}]})
            server.write_json(server.DAILY_AUDIT_CANDIDATES_FILE, {'items': []})
            server.write_json(server.REVIEW_LABELS_FILE, {'sample': {
                'recovery_event_id': 'event-old', 'recovery_status': 'covered_pending',
            }})
            try:
                item = server.load_review_candidates()['items'][0]
                self.assertEqual(item['recovery_status'], 'covered_confirmed')
            finally:
                for name, value in previous.items():
                    setattr(server, name, value)

    def test_archive_verifier_classification_policy(self):
        classify = self.archive_verifier.classify_verification
        self.assertEqual(classify(True, 2, 0.95), 'confirmed')
        self.assertEqual(classify(False, 0, 0.10), 'rejected')
        self.assertEqual(classify(False, 1, 0.91), 'uncertain')
        self.assertEqual(classify(False, 0, 0.0, classifier_available=False), 'error')

    def test_archive_anchor_gate_accepts_sparse_spatially_coherent_hits(self):
        gate = self.archive_verifier.ArchiveAnchorGate(window_sec=8, min_iou=0.2)
        self.assertFalse(gate.observe(0.0, [100, 0, 800, 260]))
        self.assertFalse(gate.observe(2.0, [900, 300, 1100, 500]))
        self.assertTrue(gate.observe(4.5, [110, 2, 810, 265]))
        self.assertFalse(gate.observe(20.0, [115, 2, 815, 265]))

    def test_dominant_anchor_cluster_joins_sparse_missed_drink_hits(self):
        first, last, span = self.archive_verifier.dominant_anchor_cluster(
            [2.0, 40.0, 107.0, 119.8], window_sec=15)
        self.assertEqual((first, last), (107.0, 119.8))
        self.assertAlmostEqual(span, 12.8)
        self.assertAlmostEqual(
            self.archive_verifier.total_anchor_cluster_duration(
                [1.0, 1.5, 40.0, 42.0], window_sec=15, credit_sec=0.5),
            3.5,
        )

    def test_recovery_candidate_can_be_verified_after_normal_lookback(self):
        worker = self.archive_verifier
        timestamp = (worker.datetime.now(worker.CST) - worker.timedelta(hours=48)).isoformat()
        normal = {'id': 'normal', 'end_time': timestamp}
        recovery = {'id': 'recovery', 'end_time': timestamp, 'recoveryCandidate': True}
        eligible = worker.eligible_events([normal, recovery], {'events': {}})
        self.assertEqual([event['id'] for event in eligible], ['recovery'])

    def test_unlabeled_review_candidate_becomes_automatic_probe_after_wait(self):
        worker = self.archive_verifier
        timestamp = (worker.datetime.now(worker.CST) - worker.timedelta(minutes=10)).isoformat()
        queue = {'items': [
            {'id': 'candidate', 'created_at': timestamp, 'classifier_confidence': 0.46},
            {'id': 'low', 'created_at': timestamp, 'classifier_confidence': 0.10},
            {'id': 'labeled', 'created_at': timestamp, 'classifier_confidence': 0.90,
             'label': 'not_drinking'},
        ]}
        eligible = worker.eligible_review_candidates(queue)
        self.assertEqual([candidate['id'] for candidate in eligible], ['candidate'])

    def test_archive_verification_window_covers_variable_recording_delay(self):
        event = {
            'start_time': '2026-07-12T08:49:19+08:00',
            'end_time': '2026-07-12T08:49:29+08:00',
        }
        start, end = self.server.event_clip_window(event, archive_verification=True)
        self.assertEqual(start.strftime('%H:%M:%S'), '08:46:49')
        self.assertEqual(end.strftime('%H:%M:%S'), '08:48:29')
        self.assertLess(start, end)
        event['recoveryCandidate'] = True
        _, recovery_end = self.server.event_clip_window(event, archive_verification=True)
        self.assertEqual(recovery_end.strftime('%H:%M:%S'), '08:49:09')

    def test_daily_audit_deduplicates_using_archive_anchor_time(self):
        audit = self.daily_audit
        event = {
            'id': 'realtime',
            'start_time': '2026-07-12T19:12:31+08:00',
            'verificationStatus': 'confirmed',
            'archiveVerification': {'first_anchor_sec': 103.9},
        }
        archive_time, tolerance = audit.event_archive_time(event)
        self.assertEqual(archive_time.strftime('%H:%M:%S'), '19:11:44')
        self.assertEqual(tolerance, 45)
        candidate_time = audit.datetime.fromisoformat('2026-07-12T19:11:32+08:00')
        self.assertEqual(audit.covered_by_existing(candidate_time, [event]), 'realtime')
        event['start_time'] = '2026-07-12T08:49:19+08:00'
        event['archiveVerification']['first_anchor_sec'] = 44.105
        candidate_time = audit.datetime.fromisoformat('2026-07-12T08:49:44+08:00')
        self.assertEqual(audit.covered_by_existing(candidate_time, [event]), 'realtime')

    def test_daily_audit_recovered_event_uses_actual_start_time(self):
        audit = self.daily_audit
        candidate_time = audit.datetime.fromisoformat('2026-07-12T19:12:35+08:00')
        event = {
            'id': 'recovered-audit',
            'start_time': candidate_time.isoformat(),
            'auditRecovery': True,
            'triggerSource': 'nightly_archive_audit',
            'archiveVerification': {'first_anchor_sec': 155.0},
        }
        self.assertEqual(audit.covered_by_existing(candidate_time, [event]), 'recovered-audit')

    def test_daily_audit_idle_status_keeps_last_summary(self):
        audit = self.daily_audit
        payload = audit.idle_status_payload({'completed_dates': {
            '2026-07-12': {
                'completed_at': '2026-07-13T04:20:00+08:00',
                'summary': {'date': '2026-07-12', 'files_total': 1418, 'new_candidates': 2},
            },
        }})
        self.assertEqual(payload['phase'], 'idle')
        self.assertEqual(payload['files_total'], 1418)
        self.assertEqual(payload['new_candidates'], 2)

    def test_daily_audit_quick_gate_keeps_single_strong_anchor(self):
        audit = self.daily_audit
        floor = min(
            audit.detector_module.CLASSIFIER_TEMPORAL_ANCHOR_THRESHOLD,
            audit.detector_module.NIGHT_CLASSIFIER_TEMPORAL_ANCHOR_THRESHOLD,
        )
        self.assertTrue(audit.quick_result_needs_full_analysis({
            'status': 'uncertain', 'max_classifier': floor,
        }))
        self.assertTrue(audit.quick_result_needs_full_analysis({
            'status': 'confirmed', 'max_classifier': 0,
        }))
        self.assertFalse(audit.quick_result_needs_full_analysis({
            'status': 'rejected', 'max_classifier': floor - 0.01,
        }))

    def test_daily_audit_deduplicates_candidates_across_file_boundary(self):
        audit = self.daily_audit
        candidate_time = audit.datetime.fromisoformat('2026-07-12T01:26:02+08:00')
        payload = {'items': [{
            'id': 'audit-first', 'created_at': '2026-07-12T01:25:52+08:00',
        }]}
        self.assertEqual(
            audit.nearby_audit_candidate(candidate_time, payload), 'audit-first')
        payload['items'][0]['recording_file'] = '20260712012458-60896.mp4'
        self.assertEqual(audit.audit_candidate_for_recording(
            Path('/archive/20260712012458-60896.mp4'), payload), 'audit-first')
        payload = {'items': [{
            'id': 'audit-span', 'created_at': '2026-07-12T02:24:04+08:00',
            'active_drink_sec': 46.569, 'recording_file': 'first.mp4',
            'classifier_confidence': 0.97, 'cat_confidence': 0.8,
            'archive_verification': {'active_drink_sec': 46.569},
        }]}
        next_time = audit.datetime.fromisoformat('2026-07-12T02:24:53+08:00')
        self.assertEqual(audit.nearby_audit_candidate(
            next_time, payload, active_sec=21.079), 'audit-span')
        audit.merge_candidate_evidence(payload['items'][0], Path('second.mp4'), {
            'active_drink_sec': 21.079, 'max_classifier': 0.98, 'max_cat': 0.89,
        })
        self.assertEqual(payload['items'][0]['active_drink_sec'], 67.648)
        self.assertEqual(payload['items'][0]['recording_files'], ['first.mp4', 'second.mp4'])

    def test_daily_audit_normalization_merges_spans_and_removes_covered(self):
        audit = self.daily_audit
        payload = {'version': 1, 'items': [
            {'id': 'first', 'created_at': '2026-07-12T02:24:04+08:00',
             'active_drink_sec': 46.569, 'recording_file': 'first.mp4'},
            {'id': 'second', 'created_at': '2026-07-12T02:24:53+08:00',
             'active_drink_sec': 21.079, 'recording_file': 'second.mp4'},
            {'id': 'covered', 'created_at': '2026-07-12T08:49:44+08:00',
             'active_drink_sec': 10, 'recording_file': 'covered.mp4'},
        ]}
        events = [{
            'id': 'known', 'start_time': '2026-07-12T08:49:19+08:00',
            'verificationStatus': 'confirmed',
        }]
        result = audit.normalize_candidate_payload(payload, events)
        self.assertEqual(result, {
            'before': 3, 'covered_removed': 1, 'merged': 1, 'after': 1,
        })
        self.assertEqual(payload['items'][0]['active_drink_sec'], 67.648)

    def test_only_verified_or_human_confirmed_events_count_for_stats(self):
        server = self.server
        today = server.datetime.now(server.CST).strftime('%Y-%m-%d')
        rejected = {
            'id': 'rejected', 'start_time': f'{today}T08:00:00',
            'end_time': f'{today}T08:00:10', 'metric': '10s',
            'estimatedMl': 1.4, 'verificationStatus': 'rejected',
        }
        accepted = {
            'id': 'accepted', 'start_time': f'{today}T09:00:00',
            'end_time': f'{today}T09:00:10', 'metric': '10s',
            'estimatedMl': 1.4, 'verificationStatus': 'confirmed',
        }
        legacy = {
            'id': 'legacy', 'start_time': f'{today}T10:00:00',
            'end_time': f'{today}T10:00:10', 'metric': '10s',
            'estimatedMl': 1.4, 'volumeMethod': 'legacy_event_duration_v1',
        }
        pending = {
            'id': 'pending', 'start_time': f'{today}T11:00:00',
            'end_time': f'{today}T11:00:10', 'metric': '10s',
            'estimatedMl': 1.4, 'volumeMethod': 'active_lapping_v2',
            'verificationStatus': 'pending',
        }
        data = {'events': [rejected, accepted, legacy, pending], 'overview': {}}
        server.recalc_overview(data)
        self.assertEqual(data['overview']['drinkCount'], 1)
        self.assertEqual(data['overview']['todayPendingCount'], 1)
        self.assertEqual(data['overview']['legacyExcludedCount'], 1)
        rejected['label'] = 'confirmed'
        server.recalc_overview(data)
        self.assertEqual(data['overview']['drinkCount'], 2)

    def test_verification_update_is_atomic_and_bounded(self):
        server = self.server
        with tempfile.TemporaryDirectory() as temporary:
            previous = server.DATA_FILE
            server.DATA_FILE = Path(temporary) / 'mock.json'
            server.write_json(server.DATA_FILE, {
                'events': [{'id': 'sample', 'start_time': '2026-07-12T08:00:00'}],
                'overview': {},
            })
            try:
                event = server.update_event_verification('sample', {
                    'status': 'rejected', 'max_classifier': 2.0,
                    'samples': 20, 'reason': 'x' * 300,
                })
                self.assertEqual(event['verificationStatus'], 'rejected')
                self.assertEqual(event['archiveVerification']['max_classifier'], 1.0)
                self.assertEqual(len(event['archiveVerification']['reason']), 240)
                self.assertEqual(event['tag'], '高清复核未通过')
                event = server.update_event_verification('sample', {
                    'status': 'confirmed', 'anchors': 3, 'active_drink_sec': 6.0,
                })
                self.assertEqual(event['metric'], '6s')
                self.assertEqual(event['activeDrinkSec'], 6.0)
                self.assertEqual(event['estimatedMl'], 2.7)
                self.assertEqual(event['volumeMethod'], 'archive_anchor_clusters_v2')
                with self.assertRaisesRegex(ValueError, 'status'):
                    server.update_event_verification('sample', {'status': 'pending'})
            finally:
                server.DATA_FILE = previous

    def test_review_drinking_recovery_counts_only_after_archive_confirmation(self):
        server = self.server
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = {
                'BASE_DIR': server.BASE_DIR,
                'DATA_FILE': server.DATA_FILE,
                'REVIEW_CANDIDATES_FILE': server.REVIEW_CANDIDATES_FILE,
                'REVIEW_LABELS_FILE': server.REVIEW_LABELS_FILE,
                'SETTINGS_FILE': server.SETTINGS_FILE,
            }
            server.BASE_DIR = root
            server.DATA_FILE = root / 'mock.json'
            server.REVIEW_CANDIDATES_FILE = root / 'review.json'
            server.REVIEW_LABELS_FILE = root / 'labels.json'
            server.SETTINGS_FILE = root / 'settings.json'
            today = server.datetime.now(server.CST).strftime('%Y-%m-%d')
            server.write_json(server.DATA_FILE, {'events': [], 'overview': {}})
            server.write_json(server.REVIEW_CANDIDATES_FILE, {'items': [{
                'id': 'missed', 'created_at': f'{today}T12:38:42+08:00',
                'snapshot_url': '/media/review_candidates/missed.jpg',
                'crop_url': '/media/review_candidates/missed-crop.jpg',
                'classifier_confidence': 0.462, 'profile': 'day',
            }]})
            server.write_json(server.SETTINGS_FILE, {
                'mlPerActiveSec': 0.45,
                'mlPerActiveSecLow': 0.30,
                'mlPerActiveSecHigh': 0.70,
            })
            try:
                server.label_review_candidate('missed', 'drinking')
                recovery = server.sync_review_candidate_recovery('missed', 'drinking')
                self.assertEqual(recovery['action'], 'created')
                event = recovery['event']
                self.assertEqual(event['verificationStatus'], 'pending')
                self.assertFalse(server.event_counts_for_stats(event))
                data = server.load_dashboard_data()
                self.assertEqual(data['overview']['drinkCount'], 0)

                event = server.update_event_verification(event['id'], {
                    'status': 'confirmed', 'samples': 240, 'anchors': 2,
                    'max_classifier': 0.9434, 'anchor_span_sec': 12.8,
                    'first_anchor_sec': 107.0, 'last_anchor_sec': 119.8,
                })
                self.assertTrue(server.event_counts_for_stats(event))
                self.assertEqual(event['metric'], '13.3s')
                self.assertEqual(event['activeDrinkSec'], 13.3)
                self.assertEqual(event['estimatedMl'], 6.0)
                self.assertEqual(event['estimatedMlLow'], 4.0)
                self.assertEqual(event['estimatedMlHigh'], 9.3)
                self.assertEqual(event['volumeMethod'], 'archive_anchor_clusters_v2')
                overview = server.load_dashboard_data()['overview']
                self.assertEqual(overview['waterMl'], 6.0)
                self.assertEqual(overview['waterMlLow'], 4.0)
                self.assertEqual(overview['waterMlHigh'], 9.3)
                self.assertEqual(overview['volumeEstimateMethod'], 'active_lapping_v2')
                annotation = server.read_json(server.REVIEW_LABELS_FILE, {})['missed']
                self.assertEqual(annotation['recovery_status'], 'confirmed')

                server.label_review_candidate('missed', 'no_cat')
                result = server.sync_review_candidate_recovery('missed', 'no_cat')
                self.assertEqual(result['action'], 'suppressed')
                self.assertFalse(server.event_counts_for_stats(result['event']))
                self.assertEqual(server.load_dashboard_data()['overview']['drinkCount'], 0)
            finally:
                for name, value in previous.items():
                    setattr(server, name, value)

    def test_automatic_review_probe_is_excluded_until_archive_confirmation(self):
        server = self.server
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = {
                'DATA_FILE': server.DATA_FILE,
                'REVIEW_CANDIDATES_FILE': server.REVIEW_CANDIDATES_FILE,
                'REVIEW_LABELS_FILE': server.REVIEW_LABELS_FILE,
                'SETTINGS_FILE': server.SETTINGS_FILE,
            }
            server.DATA_FILE = root / 'mock.json'
            server.REVIEW_CANDIDATES_FILE = root / 'review.json'
            server.REVIEW_LABELS_FILE = root / 'labels.json'
            server.SETTINGS_FILE = root / 'settings.json'
            today = server.datetime.now(server.CST).strftime('%Y-%m-%d')
            server.write_json(server.DATA_FILE, {'events': [], 'overview': {}})
            server.write_json(server.REVIEW_CANDIDATES_FILE, {'items': [{
                'id': 'automatic', 'created_at': f'{today}T12:00:00+08:00',
                'snapshot_url': '/media/review_candidates/automatic.jpg',
                'crop_url': '/media/review_candidates/automatic-crop.jpg',
                'classifier_confidence': 0.51, 'profile': 'day',
            }]})
            server.write_json(server.SETTINGS_FILE, {
                'mlPerActiveSec': 0.45,
                'mlPerActiveSecLow': 0.30,
                'mlPerActiveSecHigh': 0.70,
            })
            try:
                recovery = server.sync_review_candidate_probe('automatic')
                self.assertEqual(recovery['action'], 'created')
                event = recovery['event']
                self.assertTrue(event['automatedRecovery'])
                self.assertFalse(server.event_counts_for_stats(event))
                self.assertEqual(server.load_dashboard_data()['overview']['drinkCount'], 0)
                duplicate = server.sync_review_candidate_probe('automatic')
                self.assertEqual(duplicate['action'], 'existing')

                confirmed = server.update_event_verification(event['id'], {
                    'status': 'confirmed', 'anchors': 3, 'active_drink_sec': 4.5,
                })
                self.assertTrue(server.event_counts_for_stats(confirmed))
                self.assertEqual(server.load_dashboard_data()['overview']['weekCount'], 1)
                queue = server.load_review_candidates()
                self.assertEqual(queue['items'][0]['recovery_status'], 'confirmed')
                self.assertEqual(queue['items'][0]['recovery_trigger'], 'automatic_archive_probe')
            finally:
                for name, value in previous.items():
                    setattr(server, name, value)

    def test_archive_audit_candidate_counts_only_after_human_confirmation(self):
        server = self.server
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = {
                'BASE_DIR': server.BASE_DIR,
                'DATA_FILE': server.DATA_FILE,
                'REVIEW_CANDIDATES_FILE': server.REVIEW_CANDIDATES_FILE,
                'DAILY_AUDIT_CANDIDATES_FILE': server.DAILY_AUDIT_CANDIDATES_FILE,
                'REVIEW_LABELS_FILE': server.REVIEW_LABELS_FILE,
                'SETTINGS_FILE': server.SETTINGS_FILE,
                'MEDIA_DIR': server.MEDIA_DIR,
                'POSITIVE_DIR': server.POSITIVE_DIR,
                'NEGATIVE_DIR': server.NEGATIVE_DIR,
                'LABEL_ORPHAN_DIR': server.LABEL_ORPHAN_DIR,
            }
            server.BASE_DIR = root
            server.DATA_FILE = root / 'mock.json'
            server.REVIEW_CANDIDATES_FILE = root / 'review.json'
            server.DAILY_AUDIT_CANDIDATES_FILE = root / 'audit.json'
            server.REVIEW_LABELS_FILE = root / 'labels.json'
            server.SETTINGS_FILE = root / 'settings.json'
            server.MEDIA_DIR = root / 'media'
            server.POSITIVE_DIR = root / 'positive'
            server.NEGATIVE_DIR = root / 'negative'
            server.LABEL_ORPHAN_DIR = root / 'orphans'
            (server.MEDIA_DIR / 'audit_candidates').mkdir(parents=True)
            (server.MEDIA_DIR / 'audit_candidates' / 'sample-crop.jpg').write_bytes(b'crop')
            server.write_json(server.DATA_FILE, {'events': [], 'overview': {}})
            server.write_json(server.REVIEW_CANDIDATES_FILE, {'items': []})
            server.write_json(server.DAILY_AUDIT_CANDIDATES_FILE, {'items': [{
                'id': 'sample', 'created_at': '2026-07-12T19:11:32+08:00',
                'snapshot_url': '/media/audit_candidates/sample.jpg',
                'crop_url': '/media/audit_candidates/sample-crop.jpg',
                'classifier_confidence': 0.998, 'cat_confidence': 0.7,
                'profile': 'day', 'source': 'nightly_archive_audit',
                'active_drink_sec': 25.5,
                'archive_verification': {'status': 'confirmed', 'anchors': 12},
            }]})
            server.write_json(server.SETTINGS_FILE, {
                'mlPerActiveSec': 0.45, 'mlPerActiveSecLow': 0.30,
                'mlPerActiveSecHigh': 0.70,
            })
            try:
                self.assertEqual(server.load_review_candidates()['counts']['unlabeled'], 1)
                server.label_review_candidate('sample', 'drinking')
                recovery = server.sync_review_candidate_recovery('sample', 'drinking')
                self.assertEqual(recovery['action'], 'created_confirmed')
                event = recovery['event']
                self.assertTrue(event['auditRecovery'])
                self.assertEqual(event['estimatedMl'], 11.5)
                self.assertTrue(server.event_counts_for_stats(event))
                self.assertEqual(server.load_dashboard_data()['overview']['weekCount'], 1)
                self.assertTrue((server.POSITIVE_DIR / f'{event["id"]}.jpg').is_file())

                server.label_review_candidate('sample', 'no_cat')
                recovery = server.sync_review_candidate_recovery('sample', 'no_cat')
                self.assertEqual(recovery['action'], 'suppressed')
                self.assertFalse(server.event_counts_for_stats(recovery['event']))
                self.assertEqual(server.load_dashboard_data()['overview']['weekCount'], 0)
            finally:
                for name, value in previous.items():
                    setattr(server, name, value)

    def test_labeled_samples_are_synced_and_stale_files_archived(self):
        server = self.server
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            media = root / 'media'
            event_dir = media / 'events'
            event_dir.mkdir(parents=True)
            (event_dir / 'confirmed.jpg').write_bytes(b'confirmed')
            (event_dir / 'crop-confirmed.jpg').write_bytes(b'crop-confirmed')
            (event_dir / 'false.jpg').write_bytes(b'false')
            previous = {
                'BASE_DIR': server.BASE_DIR,
                'MEDIA_DIR': server.MEDIA_DIR,
                'POSITIVE_DIR': server.POSITIVE_DIR,
                'NEGATIVE_DIR': server.NEGATIVE_DIR,
                'LABEL_ORPHAN_DIR': server.LABEL_ORPHAN_DIR,
            }
            server.BASE_DIR = root
            server.MEDIA_DIR = media
            server.POSITIVE_DIR = root / 'data' / 'positive'
            server.NEGATIVE_DIR = root / 'data' / 'negative'
            server.LABEL_ORPHAN_DIR = root / 'data' / 'label_orphans'
            server.POSITIVE_DIR.mkdir(parents=True)
            (server.POSITIVE_DIR / 'stale.jpg').write_bytes(b'stale')
            (server.POSITIVE_DIR / 'confirmed.jpg').write_bytes(b'newer-full-frame')
            try:
                result = server.sync_training_samples({'events': [
                    {
                        'id': 'confirmed', 'label': 'confirmed',
                        'img': '/media/events/confirmed.jpg',
                    },
                    {'id': 'false', 'label': 'false_positive', 'img': '/media/events/false.jpg'},
                ]})
                self.assertEqual(result['copied'], 2)
                self.assertEqual(result['orphaned'], 1)
                self.assertEqual(result['confirmed'], 1)
                self.assertEqual(result['false_positive'], 1)
                self.assertTrue((server.POSITIVE_DIR / 'confirmed.jpg').is_file())
                self.assertTrue((server.NEGATIVE_DIR / 'false.jpg').is_file())
                self.assertEqual((server.POSITIVE_DIR / 'confirmed.jpg').read_bytes(), b'crop-confirmed')
                self.assertEqual(result['source_counts'], {'crop': 1, 'snapshot': 1})
            finally:
                for name, value in previous.items():
                    setattr(server, name, value)

    def test_write_auth_accepts_only_matching_token(self):
        server = self.server
        previous = server.API_TOKEN
        handler = object.__new__(server.Handler)
        try:
            server.API_TOKEN = 'expected-token'
            handler.headers = {'X-API-Token': 'wrong'}
            self.assertFalse(handler._authenticated())
            handler.headers = {'Authorization': 'Bearer expected-token'}
            self.assertTrue(handler._authenticated())
        finally:
            server.API_TOKEN = previous

    def test_archived_events_are_queryable_without_mutating_active_data(self):
        server = self.server
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'events_archive'
            archive.mkdir()
            (archive / 'events-2026-01-02.json').write_text(json.dumps([{
                'id': 'archived-event', 'start_time': '2026-01-02T03:04:05+08:00',
            }]), encoding='utf-8')
            previous_archive = server.EVENT_ARCHIVE_DIR
            previous_data = server.DATA_FILE
            server.EVENT_ARCHIVE_DIR = archive
            server.DATA_FILE = root / 'mock.json'
            server.DATA_FILE.write_text(json.dumps({'events': [{'id': 'active-event'}]}), encoding='utf-8')
            try:
                archived = server.load_archived_events('2026-01-01', '2026-01-31')
                self.assertEqual(len(archived), 1)
                self.assertTrue(archived[0]['archived'])
                self.assertEqual(server.find_event_with_archive('archived-event')['id'], 'archived-event')
                self.assertEqual(server.archive_status()['archived_events'], 1)
                self.assertEqual(server.load_dashboard_data()['events'][0]['id'], 'active-event')
            finally:
                server.EVENT_ARCHIVE_DIR = previous_archive
                server.DATA_FILE = previous_data

    def test_notification_scheduler_includes_unresolved_audit_candidate(self):
        worker = self.notification_worker
        now = worker.datetime(2026, 7, 13, 14, 0, tzinfo=worker.CST)
        events = worker.build_scheduled_events(
            now,
            {'enableDailyReport': False, 'enableNoDrinkAlert': False},
            {'pushMode': '日报 + 异常提醒', 'enableReviewCandidateAlert': True},
            {'overview': {}, 'trends': {}},
            {'sent': {}, 'failed': {}, 'queue': []},
            {'items': [{
                'id': 'audit-new', 'source': 'nightly_archive_audit',
                'label': '', 'recovery_status': '', 'created_at': now.isoformat(),
            }]},
        )
        self.assertEqual([event['type'] for event in events], ['review_candidate_alert'])

    def test_classifier_calibration_threshold_respects_precision_and_recall(self):
        worker = self.classifier_calibration
        labels = [1, 1, 1, 1, 0, 0, 0, 0]
        scores = [0.95, 0.90, 0.82, 0.78, 0.42, 0.35, 0.21, 0.10]
        threshold, metrics = worker.choose_threshold(labels, scores)
        self.assertGreaterEqual(threshold, 0.42)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        result = worker.cross_validate_threshold(labels, scores, folds=2)
        self.assertEqual(result['calibrated']['f1'], 1.0)


if __name__ == '__main__':
    unittest.main()
