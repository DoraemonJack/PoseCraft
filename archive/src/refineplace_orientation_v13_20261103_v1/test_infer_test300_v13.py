"""CPU-only export safeguards; no model, labels, CUDA or evaluator runs."""
import json
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import infer_test300_v13 as inference


class ExportSafeguards(unittest.TestCase):
    def refinement_fixture(self):
        baseline = dict(center=[1., 2., 3.], yaw=0., initial_center=[1., 2., 3.],
                        initial_yaw=0., initial_candidate=2, final_candidate=2,
                        path=[dict(mode='stop', center=[1., 2., 3.], yaw=0.)])
        saved = inference.snapshot_baseline(baseline)
        refined = copy.deepcopy(baseline)
        step = dict(mode='rotate', previous_center=[1., 2., 3.], center=[1., 2., 3.],
                    previous_yaw=0., yaw=math.pi/4)
        refined.update(yaw=math.pi/4, baseline_path=copy.deepcopy(baseline['path']),
                       path=baseline['path']+[step], refinement_path=[step])
        return saved, refined

    def test_yaw_refinement_keeps_center_and_original_path(self):
        saved, refined = self.refinement_fixture()
        inference.assert_baseline_preserved(saved, refined)
        refined['center'][0] += .01
        with self.assertRaises(ValueError):
            inference.assert_baseline_preserved(saved, refined)

    def test_bad_action_or_modified_baseline_trace_rejected(self):
        saved, refined = self.refinement_fixture()
        refined['refinement_path'][0]['yaw'] = math.pi/2
        refined['yaw'] = math.pi/2
        with self.assertRaises(ValueError):
            inference.assert_baseline_preserved(saved, refined)
        saved, refined = self.refinement_fixture()
        refined['path'][0]['mode'] = 'move'
        with self.assertRaises(ValueError):
            inference.assert_baseline_preserved(saved, refined)

    def test_historical_prediction_drift_is_rejected(self):
        original = dict(entry_id=1, dataset_idx=-1, tx=1., ty=2., tz=3., rotation=45.)
        inference.assert_v11_prediction_identity(dict(original), original)
        with self.assertRaises(ValueError):
            inference.assert_v11_prediction_identity(dict(original, rotation=90.), original)

    def test_smoke_is_separate_from_official300_export(self):
        self.assertEqual(inference.selected_entry_ids(8), tuple(range(8)))
        self.assertEqual(inference.prediction_filename(8), 'predictions_smoke_8.json')
        self.assertEqual(inference.selected_entry_ids(300)[-1], 299)
        self.assertEqual(inference.prediction_filename(300), 'predictions.json')
        for invalid in (0, 301, -1, True, 1.5):
            with self.assertRaises(ValueError):
                inference.selected_entry_ids(invalid)

    def test_masked_infinite_logits_are_recorded_and_serializable(self):
        record = {'path': [{'meta': {'mode_logits': [0., float('-inf'), 1.]}}],
                  'prediction': {'rotation': 90.}}
        self.assertEqual(inference.nonfinite_paths(record), ['path[0].meta.mode_logits[1]'])
        clean = inference.diagnostic_json(record)
        self.assertIsNone(clean['path'][0]['meta']['mode_logits'][1])
        self.assertEqual(clean['prediction']['rotation'], 90.)
        json.dumps(clean, allow_nan=False)
        # A source record remains unchanged; diagnosis does not mutate inference.
        self.assertEqual(record['path'][0]['meta']['mode_logits'][1], float('-inf'))

    def test_baseline_subset_keeps_original_identity_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'original.json'
            rows = [{'entry_id': i, 'rotation': float(i % 360)} for i in range(3500)]
            path.write_text(json.dumps(rows), encoding='utf-8')
            with patch.object(inference, 'sha', return_value=inference.BASELINE_SHA256):
                result = inference.check_baseline(path, (0, 4, 299))
                self.assertEqual(set(result), {0, 4, 299})
                self.assertEqual(result[299]['entry_id'], 299)
                rows[-1]['entry_id'] = 0
                path.write_text(json.dumps(rows), encoding='utf-8')
                with self.assertRaises(ValueError):
                    inference.check_baseline(path, (0,))


if __name__ == '__main__':
    unittest.main()
