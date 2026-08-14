import unittest
from unittest.mock import patch

import numpy as np

import mobile_detection
from alert_engine import PHONE_DETECTED as PHONE_EVENT
from alert_engine import AlertEngine, AlertRule
from detectors import (
    NO_PHONE,
    PHONE_DETECTED,
    PHONE_SOURCE,
    PhoneDetector,
)
from eye_movement import UNKNOWN


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def phone_analysis(detected=True):
    prediction = {
        "class_index": 0,
        "class_name": "phone",
        "confidence": 0.84,
        "bounding_box": (10, 20, 30, 40),
        "accepted": detected,
    }
    return detected, {
        "predictions": (prediction,),
        "accepted_detections": (prediction,) if detected else (),
        "inference_size": 320,
    }


class PhoneSamplingTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.frame = np.zeros((40, 40, 3), dtype=np.uint8)

    def build_detector(self, **overrides):
        options = {
            "model": object(),
            "inference_size": 320,
            "inference_every_n_frames": 2,
            "max_reuse_age_seconds": 1.0,
            "clock": self.clock,
        }
        options.update(overrides)
        return PhoneDetector(**options)

    @patch("detectors.analyze_mobile_detection")
    def test_every_second_frame_executes_yolo_and_marks_cached_frame_unknown(
        self,
        analyze,
    ):
        analyze.return_value = phone_analysis(True)
        detector = self.build_detector()

        fresh = detector.detect(self.frame, None, timestamp=100.0)
        self.clock.now = 0.1
        cached = detector.detect(self.frame, None, timestamp=100.1)
        self.clock.now = 0.2
        next_fresh = detector.detect(self.frame, None, timestamp=100.2)

        self.assertEqual(analyze.call_count, 2)
        analyze.assert_called_with(
            self.frame,
            detector.model,
            inference_size=320,
            confirmation_size=512,
            confirmation_trigger=0.42,
        )
        self.assertEqual(fresh.state, PHONE_DETECTED)
        self.assertTrue(fresh.metadata["fresh"])
        self.assertEqual(cached.state, UNKNOWN)
        self.assertFalse(cached.suspicious)
        self.assertFalse(cached.metadata["fresh"])
        self.assertTrue(cached.metadata["reused"])
        self.assertFalse(cached.metadata["stale"])
        self.assertEqual(cached.metadata["cached_state"], PHONE_DETECTED)
        self.assertEqual(len(cached.metadata["accepted_detections"]), 1)
        self.assertTrue(next_fresh.metadata["fresh"])

    @patch("detectors.analyze_mobile_detection")
    def test_expired_cache_is_stale_and_drops_cached_boxes(self, analyze):
        analyze.return_value = phone_analysis(True)
        detector = self.build_detector(
            inference_every_n_frames=3,
            max_reuse_age_seconds=1.0,
        )

        detector.detect(self.frame, None, timestamp=100.0)
        self.clock.now = 1.1
        stale = detector.detect(self.frame, None, timestamp=101.1)

        self.assertEqual(stale.state, UNKNOWN)
        self.assertFalse(stale.metadata["reused"])
        self.assertTrue(stale.metadata["stale"])
        self.assertEqual(stale.metadata["predictions"], ())
        self.assertEqual(stale.metadata["accepted_detections"], ())

    @patch("detectors.analyze_mobile_detection")
    def test_reset_forces_fresh_inference_and_clears_session_cache(self, analyze):
        analyze.return_value = phone_analysis(False)
        detector = self.build_detector()

        first = detector.detect(self.frame, None, timestamp=100.0)
        cached = detector.detect(self.frame, None, timestamp=100.1)
        detector.reset()
        after_reset = detector.detect(self.frame, None, timestamp=200.0)

        self.assertEqual(first.state, NO_PHONE)
        self.assertEqual(cached.state, UNKNOWN)
        self.assertEqual(analyze.call_count, 2)
        self.assertTrue(after_reset.metadata["fresh"])
        self.assertEqual(after_reset.metadata["inference_sequence"], 1)

    @patch("detectors.analyze_mobile_detection")
    def test_first_qualifying_fresh_inference_replaces_cached_negative_immediately(
        self,
        analyze,
    ):
        analyze.side_effect = (phone_analysis(False), phone_analysis(True))
        detector = self.build_detector()

        first_negative = detector.detect(self.frame, None, timestamp=100.0)
        cached_negative = detector.detect(self.frame, None, timestamp=100.1)
        first_positive = detector.detect(self.frame, None, timestamp=100.2)

        self.assertEqual(first_negative.state, NO_PHONE)
        self.assertEqual(cached_negative.state, UNKNOWN)
        self.assertEqual(cached_negative.metadata["cached_state"], NO_PHONE)
        self.assertEqual(first_positive.state, PHONE_DETECTED)
        self.assertTrue(first_positive.suspicious)
        self.assertTrue(first_positive.metadata["fresh"])
        self.assertEqual(first_positive.metadata["inference_sequence"], 2)

    @patch("detectors.analyze_mobile_detection")
    def test_phone_alert_confirms_and_resolves_with_indeterminate_cached_frames(
        self,
        analyze,
    ):
        phone_present = {"value": True}
        analyze.side_effect = lambda *args, **kwargs: phone_analysis(
            phone_present["value"]
        )
        detector = self.build_detector()
        alert_engine = AlertEngine(
            rules=(
                AlertRule(
                    PHONE_EVENT,
                    PHONE_SOURCE,
                    minimum_duration_seconds=1.0,
                    clear_grace_seconds=0.5,
                    cooldown_seconds=1.0,
                ),
            ),
            clock=self.clock,
        )

        updates = []
        for current_time in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
            self.clock.now = current_time
            detection = detector.detect(
                self.frame,
                None,
                timestamp=100.0 + current_time,
            )
            updates.append(alert_engine.update((detection,)))

        confirmed = [
            event
            for update in updates
            for event in update.newly_confirmed_events
        ]
        self.assertEqual(len(confirmed), 1)

        phone_present["value"] = False
        resolved = []
        for current_time in (1.75, 2.0, 2.25):
            self.clock.now = current_time
            detection = detector.detect(
                self.frame,
                None,
                timestamp=100.0 + current_time,
            )
            update = alert_engine.update((detection,))
            resolved.extend(update.resolved_events)

        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].event_id, confirmed[0].event_id)

    def test_sampling_configuration_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            self.build_detector(inference_size=0)
        with self.assertRaises(ValueError):
            self.build_detector(confirmation_size=0)
        with self.assertRaises(ValueError):
            self.build_detector(confirmation_trigger=1.1)
        with self.assertRaises(ValueError):
            self.build_detector(inference_every_n_frames=0)
        with self.assertRaises(ValueError):
            self.build_detector(max_reuse_age_seconds=0)

    @patch("mobile_detection._analyze_mobile_detection_once")
    def test_low_confidence_phone_candidate_runs_immediate_high_resolution_pass(
        self,
        analyze_once,
    ):
        _, base_metadata = phone_analysis(False)
        base_metadata["max_phone_confidence"] = 0.44
        _, confirmation_metadata = phone_analysis(True)
        confirmation_metadata["max_phone_confidence"] = 0.72
        confirmation_metadata["inference_size"] = 512
        analyze_once.side_effect = (
            (False, base_metadata),
            (True, confirmation_metadata),
        )

        detected, metadata = mobile_detection.analyze_mobile_detection(
            self.frame,
            object(),
        )

        self.assertTrue(detected)
        self.assertEqual(analyze_once.call_count, 2)
        self.assertEqual(analyze_once.call_args_list[0].args[2], 384)
        self.assertEqual(analyze_once.call_args_list[1].args[2], 512)
        self.assertTrue(metadata["confirmation_executed"])
        self.assertEqual(metadata["inference_sizes_executed"], (384, 512))
        self.assertEqual(metadata["base_max_phone_confidence"], 0.44)
        self.assertEqual(metadata["max_phone_confidence"], 0.72)

    @patch("mobile_detection._analyze_mobile_detection_once")
    def test_sub_trigger_candidate_uses_only_fast_pass(self, analyze_once):
        _, base_metadata = phone_analysis(False)
        base_metadata["max_phone_confidence"] = 0.39
        analyze_once.return_value = (False, base_metadata)

        detected, metadata = mobile_detection.analyze_mobile_detection(
            self.frame,
            object(),
        )

        self.assertFalse(detected)
        self.assertEqual(analyze_once.call_count, 1)
        self.assertFalse(metadata["confirmation_executed"])
        self.assertEqual(metadata["inference_sizes_executed"], (384,))


if __name__ == "__main__":
    unittest.main()
