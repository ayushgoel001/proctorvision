import unittest

from alert_engine import (
    GAZE_DEVIATION,
    MULTIPLE_FACES_EVENT,
    NO_FACE_EVENT,
    AlertEngine,
    EventStatus,
)
from alert_engine import (
    PHONE_DETECTED as PHONE_EVENT,
)
from detectors import (
    FACE_PRESENCE_SOURCE,
    GAZE_SOURCE,
    HEAD_POSE_SOURCE,
    NO_PHONE,
    PHONE_DETECTED,
    PHONE_SOURCE,
    DetectionResult,
)
from eye_movement import GAZE_CENTER, GAZE_LEFT, UNKNOWN
from face_context import (
    MULTIPLE_FACES,
    NO_FACES,
    PRIMARY_ASSOCIATED,
    PRIMARY_MISSING,
    PRIMARY_PRESENT,
)
from head_pose import LOOKING_AT_SCREEN


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def frame_detections(
    *,
    gaze_state=GAZE_CENTER,
    gaze_suspicious=False,
    head_state=LOOKING_AT_SCREEN,
    head_suspicious=False,
    phone_state=NO_PHONE,
    phone_suspicious=False,
    face_state=PRIMARY_PRESENT,
    association_status=PRIMARY_ASSOCIATED,
):
    return (
        DetectionResult(GAZE_SOURCE, gaze_state, gaze_suspicious),
        DetectionResult(HEAD_POSE_SOURCE, head_state, head_suspicious),
        DetectionResult(PHONE_SOURCE, phone_state, phone_suspicious),
        DetectionResult(
            FACE_PRESENCE_SOURCE,
            face_state,
            face_state in {NO_FACES, PRIMARY_MISSING, MULTIPLE_FACES},
            metadata={"association_status": association_status},
        ),
    )


def suspicious_gaze():
    return frame_detections(gaze_state=GAZE_LEFT, gaze_suspicious=True)


class AlertEngineTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.engine = AlertEngine(clock=self.clock)

    def test_below_minimum_duration_creates_no_event(self):
        self.engine.update(suspicious_gaze())
        self.clock.advance(2.9)
        update = self.engine.update(suspicious_gaze())
        self.assertEqual(update.newly_confirmed_events, ())
        self.assertEqual(
            update.current_rule_states[GAZE_DEVIATION].status,
            EventStatus.PENDING,
        )

    def test_sustained_state_creates_exactly_one_event(self):
        self.engine.update(suspicious_gaze())
        self.clock.advance(3.0)
        update = self.engine.update(suspicious_gaze())
        self.assertEqual(len(update.newly_confirmed_events), 1)
        self.assertEqual(update.newly_confirmed_events[0].event_type, GAZE_DEVIATION)

        self.clock.advance(20.0)
        update = self.engine.update(suspicious_gaze())
        self.assertEqual(update.newly_confirmed_events, ())
        self.assertEqual(
            update.current_rule_states[GAZE_DEVIATION].status,
            EventStatus.CONFIRMED,
        )

    def test_brief_clear_frame_preserves_pending_duration(self):
        self.engine.update(suspicious_gaze())
        self.clock.advance(1.0)
        self.engine.update(frame_detections())
        self.clock.advance(0.25)
        update = self.engine.update(suspicious_gaze())
        self.assertEqual(
            update.current_rule_states[GAZE_DEVIATION].started_at,
            0.0,
        )

        self.clock.advance(1.75)
        update = self.engine.update(suspicious_gaze())
        self.assertEqual(len(update.newly_confirmed_events), 1)

    def test_pending_state_resets_after_clear_grace(self):
        self.engine.update(suspicious_gaze())
        self.clock.advance(1.0)
        self.engine.update(frame_detections())
        self.clock.advance(0.6)
        update = self.engine.update(frame_detections())
        self.assertEqual(
            update.current_rule_states[GAZE_DEVIATION].status,
            EventStatus.IDLE,
        )
        self.assertEqual(update.resolved_events, ())

    def test_confirmed_state_resolves_after_clear_grace(self):
        self.engine.update(suspicious_gaze())
        self.clock.advance(3.0)
        confirmed = self.engine.update(suspicious_gaze()).newly_confirmed_events[0]
        self.clock.advance(0.1)
        self.engine.update(frame_detections())
        self.clock.advance(0.6)
        update = self.engine.update(frame_detections())
        self.assertEqual(len(update.resolved_events), 1)
        self.assertEqual(update.resolved_events[0].event_id, confirmed.event_id)
        self.assertEqual(update.resolved_events[0].status, EventStatus.RESOLVED)
        self.assertAlmostEqual(update.resolved_events[0].started_at, 0.0)
        self.assertAlmostEqual(update.resolved_events[0].confirmed_at, 3.0)
        self.assertAlmostEqual(update.resolved_events[0].resolved_at, 3.6)
        self.assertAlmostEqual(
            update.current_rule_states[GAZE_DEVIATION].cooldown_until,
            8.6,
        )

    def test_persistent_state_does_not_duplicate_during_cooldown(self):
        self.engine.update(suspicious_gaze())
        self.clock.advance(3.0)
        self.engine.update(suspicious_gaze())
        for _ in range(5):
            self.clock.advance(5.0)
            update = self.engine.update(suspicious_gaze())
            self.assertEqual(update.newly_confirmed_events, ())

    def test_new_event_after_resolution_and_cooldown(self):
        self.engine.update(suspicious_gaze())
        self.clock.advance(3.0)
        first = self.engine.update(suspicious_gaze()).newly_confirmed_events[0]
        self.clock.advance(0.1)
        self.engine.update(frame_detections())
        self.clock.advance(0.6)
        self.engine.update(frame_detections())

        self.clock.advance(1.0)
        update = self.engine.update(suspicious_gaze())
        self.assertEqual(update.newly_confirmed_events, ())
        self.assertEqual(
            update.current_rule_states[GAZE_DEVIATION].status,
            EventStatus.RESOLVED,
        )

        self.clock.advance(4.1)
        self.engine.update(suspicious_gaze())
        self.clock.advance(3.0)
        second = self.engine.update(suspicious_gaze()).newly_confirmed_events[0]
        self.assertNotEqual(first.event_id, second.event_id)

    def test_rules_maintain_independent_state(self):
        gaze_only = suspicious_gaze()
        both = frame_detections(
            gaze_state=GAZE_LEFT,
            gaze_suspicious=True,
            phone_state=PHONE_DETECTED,
            phone_suspicious=True,
        )
        self.engine.update(gaze_only)
        self.clock.advance(1.0)
        self.engine.update(both)
        self.clock.advance(2.0)
        update = self.engine.update(both)
        self.assertEqual(
            [event.event_type for event in update.newly_confirmed_events],
            [GAZE_DEVIATION],
        )
        self.assertEqual(
            update.current_rule_states[PHONE_EVENT].status,
            EventStatus.PENDING,
        )

        self.clock.advance(1.0)
        update = self.engine.update(both)
        self.assertEqual(
            [event.event_type for event in update.newly_confirmed_events],
            [PHONE_EVENT],
        )
        self.assertEqual(
            update.current_rule_states[GAZE_DEVIATION].status,
            EventStatus.CONFIRMED,
        )

    def test_unknown_is_indeterminate_and_never_suspicious(self):
        unknown = frame_detections(gaze_state=UNKNOWN, gaze_suspicious=False)
        self.engine.update(unknown)
        self.clock.advance(30.0)
        update = self.engine.update(unknown)
        self.assertEqual(update.newly_confirmed_events, ())
        self.assertEqual(
            update.current_rule_states[GAZE_DEVIATION].status,
            EventStatus.IDLE,
        )

    def test_no_face_rule(self):
        no_face = frame_detections(
            face_state=NO_FACES,
            association_status=PRIMARY_MISSING,
        )
        self.engine.update(no_face)
        self.clock.advance(3.0)
        update = self.engine.update(no_face)
        self.assertEqual(
            [event.event_type for event in update.newly_confirmed_events],
            [NO_FACE_EVENT],
        )

    def test_multiple_faces_rule(self):
        multiple = frame_detections(
            face_state=MULTIPLE_FACES,
            association_status=PRIMARY_ASSOCIATED,
        )
        self.engine.update(multiple)
        self.clock.advance(3.0)
        update = self.engine.update(multiple)
        self.assertEqual(
            [event.event_type for event in update.newly_confirmed_events],
            [MULTIPLE_FACES_EVENT],
        )

    def test_primary_missing_and_multiple_faces_are_independent(self):
        multiple_without_primary = frame_detections(
            face_state=MULTIPLE_FACES,
            association_status=PRIMARY_MISSING,
        )
        self.engine.update(multiple_without_primary)
        self.clock.advance(3.0)
        update = self.engine.update(multiple_without_primary)
        self.assertEqual(
            [event.event_type for event in update.newly_confirmed_events],
            [NO_FACE_EVENT, MULTIPLE_FACES_EVENT],
        )

    def test_state_isolation_between_runs(self):
        self.engine.update(suspicious_gaze())
        self.clock.advance(3.0)
        self.assertEqual(
            len(self.engine.update(suspicious_gaze()).newly_confirmed_events),
            1,
        )

        second_run = AlertEngine(clock=self.clock)
        update = second_run.update(suspicious_gaze())
        self.assertEqual(update.newly_confirmed_events, ())
        self.assertEqual(
            update.current_rule_states[GAZE_DEVIATION].status,
            EventStatus.PENDING,
        )

        self.engine.reset()
        self.assertTrue(
            all(
                state.status == EventStatus.IDLE
                for state in self.engine.current_rule_states().values()
            )
        )


if __name__ == "__main__":
    unittest.main()
