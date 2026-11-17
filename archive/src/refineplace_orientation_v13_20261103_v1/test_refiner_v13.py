"""CPU integration of the real refiner/selector with recorded stub observations.

Uses no dataset labels, GPU, mesh, SSH, training or official evaluator. The
actual inference trace admission helper is exercised, not a reimplementation.
"""
import copy
import json
import math
import unittest

import numpy as np
import torch

import infer_test300_v13 as inference
import refiner_v13 as refiner


class ValueStub:
    local_state_dim = 160

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def action_values(self, state, centers, yaws, task_indices):
        index = len(self.calls)
        self.calls.append(dict(state=state.clone(), centers=centers.clone(),
                               yaws=yaws.clone(), task_indices=task_indices.clone()))
        return dict(task_q=state.new_tensor([self.scores[index]]),
                    collision_cost=state.new_tensor([[.1, .1, .1]]),
                    collision_known_logits=state.new_tensor([[10., 10., 10.]]),
                    action_delta=state.new_tensor([[-math.pi/4, 0., math.pi/4]]))


class GeometryStub:
    def __init__(self):
        self.requests = []

    def query_many(self, requests):
        self.requests.extend(copy.deepcopy(requests))
        return [(np.full(26, len(self.requests), dtype=np.float32),
                 dict(observed_yaw=q['yaw'], query_number=len(self.requests)), 1)
                for q in requests]


def fixture():
    center = np.array([1.25, -2., .75])
    yaw = math.radians(170.)
    path = [dict(step=1, mode='switch', center=center.tolist(), yaw=yaw),
            dict(step=2, mode='stop', center=center.tolist(), yaw=yaw,
                 meta=dict(mode_logits=[0., 1., float('-inf')]))]
    baseline = dict(center=center, yaw=yaw, initial_center=np.array([1., -2., .75]),
                    initial_yaw=0., initial_candidate=0, final_candidate=2,
                    fused=torch.arange(3*128, dtype=torch.float32).reshape(3, 128),
                    path=path, stopped=True, steps=2, switches=1,
                    geometry={'baseline_marker': True}, actual_geometry_queries=7)
    observation = dict(scene='synthetic_scene', split='test', asset_uuid='synthetic_asset',
                       dims=np.array([.5, .4, .8]), centers=np.zeros((3, 3)))
    return baseline, observation


def admit(before, after):
    before_keys = ('center', 'yaw', 'path', 'initial_center', 'initial_yaw', 'initial_candidate', 'final_candidate')
    after_keys = before_keys + ('baseline_path', 'refinement_path')
    saved = inference.snapshot_baseline(inference.diagnostic_json(refiner._json({k: before[k] for k in before_keys})))
    result = inference.diagnostic_json(refiner._json({k: after[k] for k in after_keys}))
    inference.assert_baseline_preserved(saved, result)
    json.dumps(result, allow_nan=False)


class RefinerContractTests(unittest.TestCase):
    def test_stop_keeps_pose_and_queries_only_current_state(self):
        baseline, observation = fixture()
        before = copy.deepcopy(baseline)
        model, service = ValueStub([[.1, .8, .2]]), GeometryStub()
        result = refiner.refine_after_baseline_v13(model, baseline, observation, service)
        admit(before, result)
        self.assertEqual(result['yaw'], before['yaw'])
        self.assertTrue(np.array_equal(result['center'], before['center']))
        self.assertEqual(result['refinement_modes'], {'stop': 1})
        self.assertEqual(result['refinement_path'][0]['meta']['delta_yaw_rad'], 0.)
        # One current-state observation is needed; STOP causes no pose-change query.
        self.assertEqual(len(service.requests), 1)
        self.assertEqual(service.requests[0]['yaw'], before['yaw'])
        self.assertEqual(result['refinement_geometry_queries'], 1)
        self.assertEqual(baseline['path'], before['path'])
        self.assertEqual(baseline['geometry'], before['geometry'])

    def test_rotate_then_stop_requeries_only_selected_pose_and_keeps_prefix(self):
        baseline, observation = fixture()
        before = copy.deepcopy(baseline)
        model, service = ValueStub([[.1, .3, .9], [.1, .9, .2]]), GeometryStub()
        result = refiner.refine_after_baseline_v13(model, baseline, observation, service)
        admit(before, result)
        self.assertEqual([p['mode'] for p in result['refinement_path']], ['rotate', 'stop'])
        self.assertEqual(result['path'][:2], before['path'])
        self.assertEqual(result['refinement_steps'], 2)
        self.assertEqual(result['steps'], 4)
        self.assertEqual(len(service.requests), 2)
        self.assertEqual(result['refinement_geometry_queries'], 2)
        self.assertEqual(result['actual_geometry_queries'], 9)
        self.assertEqual(service.requests[0]['yaw'], before['yaw'])
        self.assertEqual(service.requests[1]['yaw'], result['yaw'])
        for request in service.requests:
            self.assertEqual(request['center'], before['center'].tolist())
        delta = result['refinement_path'][0]['meta']['delta_yaw_rad']
        self.assertAlmostEqual(delta, math.pi/4, places=6)
        expected = (before['yaw']+delta+math.pi)%(2*math.pi)-math.pi
        self.assertAlmostEqual(result['yaw'], expected, places=12)
        self.assertLess(result['yaw'], 0., 'Exercise wrapping through +pi')
        self.assertFalse(result['refinement_path'][0]['meta']['safety_guarantee'])
        # The second decision observes the geometry returned at the new yaw,
        # while both decisions use the original selected candidate's features.
        for call in model.calls:
            self.assertTrue(torch.equal(call['state'][0, :128], before['fused'][2]))
            self.assertTrue(torch.equal(call['task_indices'], torch.tensor([0])))
        self.assertTrue(torch.equal(model.calls[1]['state'][0, 128:154], torch.full((26,), 2.)))
        self.assertEqual(baseline['path'], before['path'])
        self.assertTrue(np.array_equal(baseline['center'], before['center']))
        self.assertEqual(baseline['yaw'], before['yaw'])

    def test_zero_budget_preserves_entire_baseline_without_queries(self):
        baseline, observation = fixture()
        before = copy.deepcopy(baseline)
        model, service = ValueStub([]), GeometryStub()
        result = refiner.refine_after_baseline_v13(model, baseline, observation, service, max_steps=0)
        admit(before, result)
        self.assertEqual(result['refinement_path'], [])
        self.assertEqual(result['path'], before['path'])
        self.assertEqual(result['geometry'], before['geometry'])
        self.assertEqual(result['refinement_geometry_queries'], 0)
        self.assertEqual(service.requests, [])
        self.assertEqual(model.calls, [])


if __name__ == '__main__':
    unittest.main()
