import contextlib
import io
import unittest

from download_weights import parse_args


class WeightArgumentsTests(unittest.TestCase):
    def test_positional_yolov8_names(self):
        _, args = parse_args(['yolov8n.pt', 'yolov8s.pt', '--output', 'weights'])
        self.assertEqual(args.models, ['yolov8n.pt', 'yolov8s.pt'])

    def test_old_flag_and_defaults_still_work(self):
        self.assertEqual(parse_args([])[1].models, ['yolo11s.pt', 'yolo26s.pt'])
        self.assertEqual(parse_args(['--models','yolo11n.pt','yolo26n.pt'])[1].models,
                         ['yolo11n.pt','yolo26n.pt'])

    def test_combined_arguments_deduplicate(self):
        _, args = parse_args(['yolov8n.pt','--models','yolov8n.pt','yolo11n.pt'])
        self.assertEqual(args.models, ['yolov8n.pt','yolo11n.pt'])

    def test_typo_is_rejected_with_correct_filename(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            parse_args(['yolo8n.pt'])
        self.assertEqual(error.exception.code, 2)
        self.assertIn('yolov8n.pt', stderr.getvalue())
