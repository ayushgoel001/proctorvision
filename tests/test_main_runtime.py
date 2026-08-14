import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import main


class MainRuntimeTests(unittest.TestCase):
    @patch("main.run_application")
    def test_debug_flag_is_passed_into_runtime_without_global_args(self, run_application):
        run_application.return_value = 0

        with patch.object(sys, "argv", ["main.py", "--debug"]):
            result = main.main()

        self.assertEqual(result, 0)
        run_application.assert_called_once_with(source="0", debug=True)

    @patch("main.run_application")
    def test_source_argument_is_passed_into_runtime(self, run_application):
        run_application.return_value = 0

        with patch.object(sys, "argv", ["main.py", "--source", "Demo_vid/exam.mp4"]):
            result = main.main()

        self.assertEqual(result, 0)
        run_application.assert_called_once_with(
            source="Demo_vid/exam.mp4",
            debug=False,
        )

    def test_camera_source_is_resolved_deterministically(self):
        source = main.resolve_capture_source("2")

        self.assertEqual(source.capture_value, 2)
        self.assertEqual(source.session_label, "camera:2")
        self.assertTrue(source.is_camera)

    def test_video_source_uses_file_without_storing_external_absolute_path(self):
        with TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "sample.mp4"
            video_path.touch()

            source = main.resolve_capture_source(video_path)

        self.assertEqual(source.capture_value, str(video_path.resolve()))
        self.assertEqual(source.session_label, "video:sample.mp4")
        self.assertFalse(source.is_camera)

    def test_missing_video_source_is_rejected_clearly(self):
        with self.assertRaises(FileNotFoundError):
            main.resolve_capture_source("Demo_vid/does-not-exist.mp4")

    def test_negative_camera_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            main.resolve_capture_source("-1")


if __name__ == "__main__":
    unittest.main()
