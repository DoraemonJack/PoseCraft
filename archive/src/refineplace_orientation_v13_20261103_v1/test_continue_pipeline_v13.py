"""Local admission tests; never launches inference, SSH, training or Docker."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import continue_pipeline_v13 as pipeline
import watch_official_v13 as watcher


class ContinuationTests(unittest.TestCase):
    def completed(self, folder):
        root = Path(folder)
        run = root/'refine1000_v1'
        run.mkdir()
        parent = root/'parent.pt'
        parent.write_bytes(b'frozen original V11')
        (run/'final.pt').write_bytes(b'new paired-value heads')
        hashes = {}
        for name in pipeline.SOURCE_FILES:
            (root/name).write_text(name)
            hashes[name] = pipeline.sha(root/name)
        protocol = dict(architecture='RefinePlaceV13',
            phase='paired_heading_and_one_step_action_value', tasks=1000, epochs=3,
            baseline_frozen=True, full_horizon_q_learning=False, test_labels_read=False,
            validation_labels_read=False, original_full_training_resumed=False,
            parent_checkpoint=str(parent), parent_checkpoint_sha256=pipeline.sha(parent),
            source_hashes=hashes)
        summary = dict(status='training_complete', architecture='RefinePlaceV13',
            tasks=1000, epochs=3, baseline_weights_unchanged=True,
            parent_checkpoint_unchanged=True, independent_test_used=False,
            final_checkpoint_sha256=pipeline.sha(run/'final.pt'))
        pipeline.write(run/'protocol_before_training.json', protocol)
        pipeline.write(run/'summary.json', summary)
        return root, run, parent, hashes

    def test_accept_complete_even_after_trainer_exits_then_reject_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root, run, parent, hashes = self.completed(folder)
            with patch.object(pipeline, 'HERE', root), patch.object(pipeline, 'PARENT_SHA256', pipeline.sha(parent)):
                with patch.object(pipeline, 'process_token', return_value=None):
                    result = pipeline.wait_training(run, 123, hashes, timeout=1, poll_seconds=.1,
                                                    progress=lambda *a, **k: None)
                self.assertTrue(result['summary']['baseline_weights_unchanged'])
                parent.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Parent checkpoint changed'):
                    pipeline.validate_training(run, hashes)

    def test_training_stop_exit_and_timeout_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            pipeline.write(run/'progress.json', dict(status='stopped'))
            with patch.object(pipeline, 'process_token', return_value='alive'):
                with self.assertRaisesRegex(RuntimeError, 'Training stopped'):
                    pipeline.wait_training(run, 123, {}, timeout=1, poll_seconds=.1,
                                           progress=lambda *a, **k: None)
            pipeline.write(run/'progress.json', dict(status='training'))
            with patch.object(pipeline, 'process_token', return_value=None):
                with self.assertRaisesRegex(RuntimeError, 'process exited'):
                    pipeline.wait_training(run, 123, {}, timeout=1, poll_seconds=.1,
                                           progress=lambda *a, **k: None)
            with patch.object(pipeline, 'process_token', return_value='alive'), \
                 patch.object(pipeline.time, 'monotonic', side_effect=[0., 2.]):
                with self.assertRaises(TimeoutError):
                    pipeline.wait_training(run, 123, {}, timeout=1, poll_seconds=.1,
                                           progress=lambda *a, **k: None)

    def test_source_drift_and_false_freeze_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root, run, parent, hashes = self.completed(folder)
            with patch.object(pipeline, 'HERE', root), patch.object(pipeline, 'PARENT_SHA256', pipeline.sha(parent)):
                summary = pipeline.read(run/'summary.json')
                summary['baseline_weights_unchanged'] = 1  # booleans must really be bool
                pipeline.write(run/'summary.json', summary)
                with self.assertRaisesRegex(ValueError, 'baseline_weights_unchanged'):
                    pipeline.validate_training(run, hashes)
                summary['baseline_weights_unchanged'] = True
                pipeline.write(run/'summary.json', summary)
                (root/'model_v13.py').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    pipeline.validate_training(run, hashes)

    def test_command_and_watcher_failure_handshake(self):
        command = pipeline.inference_command(Path('/run/refine1000_v1'))
        self.assertEqual(pipeline.option(command, '--limit'), '300')
        self.assertEqual(Path(pipeline.option(command, '--output')).parts[-2:], ('refine1000_v1', 'test300'))
        self.assertNotIn('--resume', command)
        watcher.check_pipeline({'stage': 'waiting_for_paired_value_training'})
        watcher.check_pipeline({'stage': 'predictions_ready', 'status': 'complete'})
        with self.assertRaises(ValueError):
            watcher.check_pipeline({'stage': 'failed', 'status': 'failed'})


if __name__ == '__main__':
    unittest.main()
