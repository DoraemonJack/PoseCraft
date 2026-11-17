"""CPU-only handoff checks: no SSH, upload, Docker, model or evaluation calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import evaluate_test300_official_v13 as evaluate
import watch_official_v13 as watch


class AdmissionTests(unittest.TestCase):
    def export(self, folder):
        folder = Path(folder)
        prediction = folder/'predictions.json'
        protocol_path = folder/'protocol.json'
        summary_path = folder/'summary.json'
        rows = [dict(entry_id=i, dataset_idx=-1, tx=1., ty=2., tz=3., rotation=float(i%8)*45.) for i in range(300)]
        protocol = dict(version='orientation_v13_test300_frozen_forward', tasks=300, split='test',
                        architecture='RefinePlaceV13', run_kind='complete_test300',
                        original_entry_ids=list(range(300)), test_labels_read=False,
                        ground_truth_relationships_read=False, instance_annotation_inputs=False,
                        official_evaluator_modified=False, checkpoint_sha256='a'*64,
                        manifest_sha256='b'*64, training_protocol_sha256='c'*64,
                        v11_prediction_sha256='c511211848fd0d89ca6185b4789603679c7659643600e8dba1c97c535aa73c74',
                        v11_checkpoint_sha256='480bb902bbbeb174a36da395fdf531c1845386cc66d653478c2584081e41c774')
        evaluate.write(prediction, rows)
        evaluate.write(protocol_path, protocol)
        summary = dict(protocol, status='complete', official_test300_complete=True,
                       inference_complete=True, weights_unchanged=True,
                       center_unchanged=True, baseline_first_two_steps_unchanged=True,
                       frozen_v11_prediction_identical=True,
                       frozen_v11_prediction_sha256=protocol['v11_prediction_sha256'],
                       prediction_filename='predictions.json', prediction_sha256=evaluate.sha(prediction),
                       protocol_sha256=evaluate.sha(protocol_path))
        evaluate.write(summary_path, summary)
        return prediction, protocol_path, summary_path, summary

    def test_complete_export_admitted_but_smoke_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            prediction, protocol, summary, data = self.export(folder)
            evaluate.validate_inference(summary, prediction, protocol)
            data['official_test300_complete'] = False
            evaluate.write(summary, data)
            with self.assertRaises(ValueError):
                evaluate.validate_inference(summary, prediction, protocol)

    def test_duplicate_id_rejected_even_with_consistent_prediction_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            prediction, protocol, summary, data = self.export(folder)
            rows = evaluate.read(prediction)
            rows[-1]['entry_id'] = 0
            evaluate.write(prediction, rows)
            data['prediction_sha256'] = evaluate.sha(prediction)
            evaluate.write(summary, data)
            with self.assertRaises(ValueError):
                evaluate.validate_inference(summary, prediction, protocol)

    def test_corrupted_source_seal_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            prediction, protocol, summary, data = self.export(folder)
            data['checkpoint_sha256'] = 'd'*64
            evaluate.write(summary, data)
            with self.assertRaises(ValueError):
                evaluate.validate_inference(summary, prediction, protocol)

    def test_nan_pose_and_pipeline_failure_are_fail_closed(self):
        with self.assertRaises(ValueError):
            watch.strict_json('{"tx": NaN}')
        self.assertFalse(watch.check_pipeline(None))
        self.assertFalse(watch.check_pipeline({'stage': 'inference'}))
        self.assertFalse(watch.check_pipeline({'stage': 'predictions_ready', 'status': 'complete'}))
        self.assertTrue(watch.check_pipeline({'stage': 'predictions_ready', 'status': 'complete', 'official_test300_complete': True}))
        with self.assertRaises(ValueError):
            watch.check_pipeline({'stage': 'failed'})

    def test_wrong_official_resources_or_failed_checks_are_rejected(self):
        source = dict(prediction_sha256='a'*64, checkpoint_sha256='b'*64)
        summary = dict(status='complete', audited=True, tasks=300, final_metrics_available=True,
                       prediction_sha256='a'*64, checkpoint_sha256='b'*64,
                       model_architecture='RefinePlaceV13', workers=3)
        receipt = dict(status='passed', prediction_sha256='a'*64, checks={'no_oom': True})
        protocol = dict(version='orientation_v13_official_test300_v1', prediction_sha256='a'*64,
                        workers=3, memory_bytes=evaluate.MEMORY_BYTES,
                        memory_plus_swap_bytes=evaluate.MEMORY_SWAP_BYTES, image=evaluate.IMAGE)
        native = {'metrics_by_constraint_count': {'one': {'total_placements': 200}, 'two': {'total_placements': 100}}}
        watch.assert_official_receipt(summary, receipt, protocol, native, source)
        wrong = dict(protocol, workers=4)
        with self.assertRaises(ValueError):
            watch.assert_official_receipt(summary, receipt, wrong, native, source)
        receipt['checks']['no_oom'] = False
        with self.assertRaises(ValueError):
            watch.assert_official_receipt(summary, receipt, protocol, native, source)

    def test_different_local_artifact_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            incoming, output = Path(folder)/'incoming', Path(folder)/'output'
            incoming.write_text('new')
            output.write_text('original')
            with self.assertRaises(ValueError):
                watch.commit_local(incoming, output)
            self.assertEqual(output.read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
