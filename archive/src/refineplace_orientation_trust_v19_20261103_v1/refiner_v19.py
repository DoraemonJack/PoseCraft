"""Learned STOP-relative orientation refinement after immutable V11 placement.

Only the selected new pose is queried. No GT or exhaustive geometric search
enters inference. Learned risk estimates retain uncertainty; they are not a
physical safety certificate.
"""
import math
from collections import Counter
import numpy as np
import torch
from model_v19 import select_action


def _request(observation, center, yaw):
    return dict(scene=observation['scene'], split=observation['split'], asset_uuid=observation['asset_uuid'],
        dims=np.asarray(observation['dims']).tolist(), center=np.asarray(center).tolist(), yaw=float(yaw))


def _json(value):
    if torch.is_tensor(value):
        return _json(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    return value


def refine_after_baseline_v19(model, baseline_result, observation, geometry_service,
                            *, max_steps=2, deterministic=True, task_index=0):
    if not deterministic or not 0 <= max_steps <= 2:
        raise ValueError('V19 pilot uses deterministic refinement with at most two steps')
    center = np.asarray(baseline_result['center'], float).copy()
    yaw = float(baseline_result['yaw'])
    original_yaw = yaw
    path = []
    vector = None
    geometry = baseline_result['geometry']
    queries = 0
    fused = baseline_result['fused'][int(baseline_result['final_candidate'])]
    if fused.ndim != 1 or not torch.isfinite(fused).all():
        raise ValueError('Finite selected frozen candidate feature required')
    device = fused.device
    for step in range(max_steps):
        if vector is None:
            answers = geometry_service.query_many([_request(observation, center, yaw)])
            if len(answers) != 1:
                raise ValueError('Refinement geometry lost its current pose')
            vector, geometry, _ = answers[0]
            queries += 1
        vector = np.asarray(vector, np.float32)
        extra = np.r_[vector, np.zeros(3), math.sin(yaw), math.cos(yaw), 1.]
        state = torch.cat((fused, torch.as_tensor(extra, dtype=fused.dtype, device=device)))[None]
        if state.shape[-1] != model.local_state_dim:
            raise ValueError('Current-pose refinement features do not match the training schema')
        values = model.action_values(state, state.new_tensor(center)[None], state.new_tensor([yaw]),
                                    torch.tensor([task_index], device=device, dtype=torch.long))
        choice = select_action(values)
        delta = float(choice['action_delta'][0].detach())
        previous = yaw
        stopped = bool(choice['stopped'][0])
        if stopped:
            if abs(delta) > 1e-7:
                raise ValueError('STOP must keep the exact current pose')
            mode = 'stop'
        else:
            if not math.isclose(abs(delta), math.pi/4, abs_tol=1e-6):
                raise ValueError('V19 pilot rotation must be exactly plus/minus 45 degrees')
            yaw = float((yaw + delta + math.pi) % (2*math.pi) - math.pi)
            answer = geometry_service.query_many([_request(observation, center, yaw)])
            if len(answer) != 1:
                raise ValueError('Refinement geometry lost its selected action')
            vector, geometry, _ = answer[0]
            queries += 1
            mode = 'rotate'
        path.append(dict(step=len(baseline_result['path'])+step+1, refinement_step=step+1,
            mode=mode, candidate=int(baseline_result['final_candidate']), previous_center=center.tolist(),
            center=center.tolist(), previous_yaw=previous, yaw=yaw, geometry=_json(geometry),
            meta=dict(action_space='one_step_yaw_value_comparison', action_index=int(choice['action_index'][0]),
                delta_yaw_rad=delta, task_q=_json(values['task_q'][0]),
                predicted_collision_cost=_json(values['collision_cost'][0]),
                predicted_collision_evidence_probability=_json(values['collision_known_logits'][0].sigmoid()),
                task_gain=float(choice['task_gain'][0]), uncertain_collision_evidence=bool(choice['uncertain'][0]),
                risk_improvement_credit=float(choice['risk_improvement_credit'][0]),
                eligible_actions=_json(choice['eligible'][0]), safety_guarantee=False),
            cumulative_refinement_geometry_queries=queries))
        if stopped:
            break
    result = dict(baseline_result)
    result.update(center=center, yaw=yaw, geometry=geometry,
        baseline_center=center.copy(), baseline_yaw=original_yaw, baseline_path=list(baseline_result['path']),
        refinement_path=path, refinement_steps=len(path), refinement_modes=dict(Counter(p['mode'] for p in path)),
        refinement_geometry_queries=queries, path=list(baseline_result['path'])+path,
        stopped=bool(path and path[-1]['mode']=='stop') if path else bool(baseline_result['stopped']),
        steps=int(baseline_result['steps'])+len(path),
        actual_geometry_queries=int(baseline_result.get('actual_geometry_queries',0))+queries)
    if not np.array_equal(result['center'], np.asarray(baseline_result['center'])):
        raise RuntimeError('Orientation refinement changed the frozen placement position')
    return result
