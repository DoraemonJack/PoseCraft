"""Isolated two-step treatment rollout; original rollout remains untouched.

No annotation inputs. Same physical-query budget as control. Adds directional
collision observations and same-candidate/unvisited-yaw switch choices only.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F
from orientation_policy_v1 import DIRECTIONAL_DIM


def jsonable(value):
    if torch.is_tensor(value):return jsonable(value.detach().cpu().numpy())
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    return value


def directional_observation(stats):
    vector=np.asarray(stats.get('directional_collision_vector'),dtype=np.float32)
    if vector.shape!=(DIRECTIONAL_DIM,) or not np.isfinite(vector).all():
        raise ValueError('geometry must provide the fixed 10-value directional collision schema')
    target=float(stats.get('collision_cost',vector[0]))
    known=bool(stats.get('collision_cost_known',bool(vector[1]) and bool(vector[2])))
    if not np.isfinite(target) or not 0.<=target<=1.:
        raise ValueError('collision cost must be finite in [0,1]')
    return vector,target,known


def _choices(state,anti_cycle):
    if not anti_cycle:return [p for p in state['top'] if p!=state['pose']]
    choices=[]
    for pose in state['order']:
        # Yaw is part of the pose identity. Spatial deduplication here would
        # incorrectly suppress collision-avoiding rotations at the same point.
        if pose in state['visited_poses']:continue
        choices.append(pose)
        if len(choices)==64:break
    return choices


def _task(observation,center,yaw):
    return dict(scene=observation['scene'],split=observation['split'],asset_uuid=observation['asset_uuid'],
                dims=np.asarray(observation['dims']).tolist(),center=np.asarray(center).tolist(),yaw=float(yaw))


def rollout_batch(model,features,texts,assets,candidate_mask,observations,geometry_service,*,steps=3,
                  deterministic=False,apply_offsets=True,anti_cycle=True,offset_radius=.15, mode_epsilon=0.,
                  epsilon_act=None):
    """Return V6-compatible per-task trajectories without annotation inputs.

    Candidates may use arbitrary padding positions; observation centers follow
    the valid candidate order. Every task must contain at least one candidate.
    Only active tasks reach local_batch and subsequent geometry service calls.
    """
    if not 0 <= mode_epsilon <= 1: raise ValueError("mode_epsilon must be in [0,1]")
    if deterministic and mode_epsilon: raise ValueError("Validation cannot mix exploration")
    batch=len(observations)
    if batch!=features.shape[0] or steps<0:raise ValueError('Invalid rollout batch/steps')
    device=features.device
    scores,fused,scene_values=model.candidates_batch(features,texts,assets,candidate_mask)
    offsets=model.offsets_batch(fused,candidate_mask,radius=offset_radius)
    # One transfer per batch, never per-task GPU argsort/scores downloads.
    cpu_scores=scores.detach().cpu();cpu_offsets=offsets.detach().cpu().numpy() if apply_offsets else None
    cpu_mask=candidate_mask.detach().cpu().numpy()
    distribution=torch.distributions.Categorical(logits=scores.reshape(batch,-1))
    selected=scores.reshape(batch,-1).argmax(-1) if deterministic else distribution.sample()
    selected_cpu=selected.detach().cpu().tolist()
    candidate_logprobs=distribution.log_prob(selected);candidate_entropies=distribution.entropy()
    states=[]
    for b,observation in enumerate(observations):
        positions=np.flatnonzero(cpu_mask[b]);centers=np.asarray(observation['centers'],float)
        if not len(positions) or centers.shape!=(len(positions),3):raise ValueError('One center per valid candidate required')
        local_scores=cpu_scores[b,positions].reshape(-1)
        selected_candidate=int(np.flatnonzero(positions==selected_cpu[b]//8)[0]);pose=selected_candidate*8+selected_cpu[b]%8
        offset=cpu_offsets[b,positions] if cpu_offsets is not None else None
        center=centers[pose//8].copy()
        if offset is not None:center[:2]+=offset[pose//8,pose%8]
        yaw=(pose%8)*math.pi/4
        states.append(dict(centers=centers,positions=positions,pose=pose,j=pose//8,center=center,yaw=yaw,
            initial_center=center.copy(),initial_yaw=yaw,initial_j=pose//8,anchor_center=centers[pose//8].copy(),offsets=offset,
            order=torch.argsort(local_scores,descending=True).tolist(),top=torch.topk(local_scores,min(64,len(local_scores))).indices.tolist(),
            visited_candidates={pose//8},visited_poses={pose},visited_centers=[center.copy()],repeat_count=0,candidate_revisit_count=0,
            path=[],logprobs=[],values=[],entropies=[],costs=[],switches=0,stopped=False,actual_queries=0,
            collision_predictions=[],collision_targets=[],collision_target_masks=[],collision_target_poses=[]))
    initial=geometry_service.query_many([_task(o,s['center'],s['yaw']) for o,s in zip(observations,states)])
    for state,(vector,stats,count) in zip(states,initial):
        state.update(vector=vector,stats=stats,initial_stats=stats,queries=count,initial_queries=count,actual_queries=1)
    for step in range(steps):
        active=[b for b,s in enumerate(states) if not s['stopped']]
        if not active:break
        chosen_fused=torch.stack([fused[b,int(states[b]['positions'][states[b]['j']])] for b in active])
        extra=[];choices=[];directions=[];collision_targets=[];collision_masks=[]
        for b in active:
            s=states[b];dims=np.asarray(observations[b]['dims'],float)
            extra.append(np.r_[s['vector'],(s['center']-s['initial_center'])/dims,
                               math.sin(s['yaw']),math.cos(s['yaw']),(steps-step)/steps])
            choices.append(_choices(s,anti_cycle))
            direction,target,known=directional_observation(s['stats'])
            directions.append(direction);collision_targets.append(target);collision_masks.append(known)
        local_state=torch.cat((chosen_fused,torch.as_tensor(np.asarray(extra),dtype=features.dtype,device=device)),-1)
        if hasattr(model,'local_batch_with_reference'):
            mode_logits,means,local_values=model.local_batch_with_reference(
                local_state,local_state.new_tensor(np.asarray([states[b]['center'] for b in active])),
                local_state.new_tensor([states[b]['yaw'] for b in active]),
                torch.tensor(active,dtype=torch.long,device=device),
                directional_collision=local_state.new_tensor(np.asarray(directions)),
                object_dims=local_state.new_tensor(np.asarray([observations[b]['dims'] for b in active])))
        else:mode_logits,means,local_values=model.local_batch(local_state)
        if not hasattr(model,'last_collision_prediction') or model.last_collision_prediction is None:
            raise ValueError('treatment rollout requires the collision-conditioned policy wrapper')
        if model.last_collision_prediction.shape!=(len(active),):
            raise ValueError('one collision prediction required per active pre-action pose')
        for index,b in enumerate(active):
            s=states[b]
            s['collision_predictions'].append(model.last_collision_prediction[index])
            s['collision_targets'].append(collision_targets[index])
            s['collision_target_masks'].append(collision_masks[index])
            s['collision_target_poses'].append(dict(center=s['center'].tolist(),yaw=float(s['yaw'])))
        switch_allowed=torch.tensor([bool(c) for c in choices],device=device)
        switch_mask=torch.stack((torch.ones_like(switch_allowed),torch.ones_like(switch_allowed),switch_allowed),-1)
        mode_logits=mode_logits.masked_fill(~switch_mask,-torch.inf)
        raw_probs=torch.softmax(mode_logits,-1)
        uniform=switch_mask.to(raw_probs.dtype)/switch_mask.sum(-1,keepdim=True)
        mixture=(1-mode_epsilon)*raw_probs+mode_epsilon*uniform
        mode_dist=torch.distributions.Categorical(probs=mixture)
        modes=mode_logits.argmax(-1) if deterministic else mode_dist.sample()
        normal=torch.distributions.Normal(means,model.log_std.exp().clamp(.1,1.5))
        latent=means if deterministic else normal.sample()
        action=torch.tanh(latent)
        max_choices=max(1,max(map(len,choices)))
        switch_logits=[]
        for b,c in zip(active,choices):
            if c:
                original=torch.as_tensor(states[b]['positions'],device=device,dtype=torch.long)
                task_scores=scores[b,original].reshape(-1)
                selected_scores=task_scores[torch.as_tensor(c,device=device,dtype=torch.long)]
                switch_logits.append(F.pad(selected_scores,(0,max_choices-len(c)),value=-torch.inf))
            else:switch_logits.append(F.pad(scores.new_zeros(1),(0,max_choices-1),value=-torch.inf))
        switch_logits=torch.stack(switch_logits)
        switch_dist=torch.distributions.Categorical(logits=switch_logits)
        switch_indices=switch_logits.argmax(-1) if deterministic else switch_dist.sample()
        decisions=torch.cat((modes[:,None],action,switch_indices[:,None]),-1).detach().cpu().numpy()
        mode_logprob=mode_dist.log_prob(modes);mode_entropy=mode_dist.entropy()
        move_logprob=normal.log_prob(latent).sum(-1)
        switch_logprob=switch_dist.log_prob(switch_indices);switch_entropy=switch_dist.entropy()
        queries=[];query_tasks=[];step_records={}
        for index,b in enumerate(active):
            s=states[b];kind=int(decisions[index,0]);act=decisions[index,1:4]
            previous=s['center'].copy();previous_yaw=s['yaw'];meta={};cost=0.
            logprob=mode_logprob[index];entropy=mode_entropy[index]
            if kind==0:
                logprob=logprob+move_logprob[index]
                local_xy=act[:2]*.05;norm=np.linalg.norm(local_xy)
                if norm>.05:local_xy*=.05/norm
                c,sine=math.cos(s['yaw']),math.sin(s['yaw'])
                s['center']=s['center']+np.r_[np.array([[c,-sine],[sine,c]])@local_xy,0.]
                s['yaw']=float((s['yaw']+act[2]*math.pi/12+math.pi)%(2*math.pi)-math.pi)
                meta=dict(snap=dict(snapped=False,reason='actual_asset_bottom_frame_preserved',height_change=0.,query_count=0,spatial_queries=0),
                          guard=dict(accepted_fraction=1.,query_count=1,heuristic_guard=False,mesh_constraints='soft_reward'),latent_action=act.tolist())
                s['queries']+=1;cost=.002
            elif kind==2:
                logprob=logprob+switch_logprob[index];entropy=entropy+switch_entropy[index]
                s['pose']=choices[index][int(decisions[index,4])];s['j']=s['pose']//8
                s['repeat_count']+=int(s['pose'] in s['visited_poses'])
                s['candidate_revisit_count']+=int(s['j'] in s['visited_candidates'])
                s['visited_candidates'].add(s['j']);s['visited_poses'].add(s['pose'])
                s['center']=s['centers'][s['j']].copy()
                if s['offsets'] is not None:s['center'][:2]+=s['offsets'][s['j'],s['pose']%8]
                s['yaw']=(s['pose']%8)*math.pi/4;s['switches']+=1;cost=.01
                meta=dict(candidate_restart=s['j'],yaw_bin_restart=s['pose']%8,anchor_center=s['centers'][s['j']].tolist(),
                          applied_offset_xy_m=(s['center'][:2]-s['centers'][s['j'],:2]).tolist())
            else:s['stopped']=True
            if kind in (0,2):
                s['visited_centers'].append(s['center'].copy());queries.append(_task(observations[b],s['center'],s['yaw']));query_tasks.append(b)
            s['logprobs'].append(logprob);s['values'].append(local_values[index]);s['entropies'].append(entropy);s['costs'].append(cost)
            meta.update(mode_logits=mode_logits[index].detach().cpu().tolist(),mode_policy_probs=raw_probs[index].detach().cpu().tolist(),mode_behavior_probs=mixture[index].detach().cpu().tolist(),mode_epsilon=mode_epsilon)
            step_records[b]=(kind,previous,previous_yaw,meta,cost)
        updates=geometry_service.query_many(queries) if queries else []
        for b,(vector,stats,count) in zip(query_tasks,updates):
            states[b].update(vector=vector,stats=stats);states[b]['queries']+=count;states[b]['actual_queries']+=1
        for index,b in enumerate(active):
            s=states[b];kind,previous,previous_yaw,meta,cost=step_records[b]
            meta['stop_reason'] = 'policy_stop' if kind == 1 else None
            if epsilon_act is not None and kind != 1:
                dims = np.asarray(observations[b]['dims'], float)
                extra_next = np.r_[s['vector'], (s['center']-s['initial_center'])/dims,
                                   math.sin(s['yaw']), math.cos(s['yaw']), (steps-step-1)/steps]
                next_fused = fused[b, int(s['positions'][s['j']])]
                next_state = torch.cat((next_fused, torch.as_tensor(extra_next, dtype=features.dtype, device=device)))[None]
                next_direction, _, _ = directional_observation(s['stats'])
                if hasattr(model, 'local_batch_with_reference'):
                    _, _, next_values = model.local_batch_with_reference(
                        next_state, next_state.new_tensor(s['center'])[None],
                        next_state.new_tensor([s['yaw']]), torch.tensor([b], dtype=torch.long, device=device),
                        directional_collision=next_state.new_tensor(next_direction)[None],
                        object_dims=next_state.new_tensor(dims)[None])
                else:
                    _, _, next_values = model.local_batch(next_state)
                gain = float((next_values[0, 0] - local_values[index, 0]).detach())
                meta['value_gain'] = gain
                if gain < epsilon_act:
                    s['stopped'] = True
                    meta['stop_reason'] = 'value_gain'
                elif step + 1 == steps:
                    meta['stop_reason'] = 'step_cap'
            s['path'].append(dict(step=step+1,mode=('move','stop','switch')[kind],candidate=s['j'],previous_center=previous.tolist(),
                previous_yaw=previous_yaw,center=s['center'].tolist(),yaw=s['yaw'],geometry=jsonable(s['stats']),meta=jsonable(meta),
                step_cost=cost,cumulative_queries=s['queries'],cumulative_actual_geometry_queries=s['actual_queries']))
    results=[]
    for b,s in enumerate(states):
        positions=torch.as_tensor(s['positions'],device=device,dtype=torch.long)
        results.append(dict(scores=scores[b,positions],fused=fused[b,positions],supervised_offsets=offsets[b,positions],
            candidate_logprob=candidate_logprobs[b],candidate_entropy=candidate_entropies[b],scene_values=scene_values[b],
            residual_logprobs=s['logprobs'],residual_values=s['values'],entropies=s['entropies'],step_costs=s['costs'],
            initial_candidate=s['initial_j'],final_candidate=s['j'],anchor_center=s['anchor_center'],
            predicted_offsets=offsets[b,positions] if apply_offsets else None,repeat_count=s['repeat_count'],
            repeat_count_scope='exact_candidate_yaw_pose',candidate_revisit_count=s['candidate_revisit_count'],
            visited_candidates=sorted(s['visited_candidates']),visited_poses=sorted(s['visited_poses']),visited_centers=[c.tolist() for c in s['visited_centers']],
            apply_offsets=apply_offsets,anti_cycle=anti_cycle,initial_center=s['initial_center'],initial_yaw=s['initial_yaw'],
            center=s['center'],yaw=s['yaw'],initial_geometry=s['initial_stats'],geometry=s['stats'],path=s['path'],steps=len(s['path']),
            switches=s['switches'],stopped=s['stopped'],queries=s['queries'],initial_queries=s['initial_queries'],snap_spatial_queries=0,
            actual_geometry_queries=s['actual_queries'],query_semantics='V6 logical pose evaluations; actual_geometry_queries counts service poses',
            collision_predictions=s['collision_predictions'],collision_targets=s['collision_targets'],
            collision_target_masks=s['collision_target_masks'],collision_target_poses=s['collision_target_poses'],
            random_draw_order='batched candidate/mode/latent/switch draws; differs from per-task V6'))
    return results


def batched_rl_losses(results,qualities,geometry_weight=.5,rl_weight=.05):
    """Per-task V6 RL term only; exact/distance/offset supervision is separate.

    Geometry and quality originate on CPU. Padding reductions retain V6's sum
    of local score-function terms and per-task mean local value/entropy terms.
    """
    from train_joint_pose_v6 import costs_with_reliability,discounted_returns
    batch=len(results)
    if not batch:raise ValueError('Nonempty result batch required')
    qualities=np.asarray(qualities,dtype=float)
    if qualities.shape!=(batch,):raise ValueError('Expected one CPU quality per task')
    lengths=[len(result['step_costs']) for result in results]
    if min(lengths)<1:raise ValueError('RL requires at least one rollout step per task')
    maximum=max(lengths);cost_rows=[];utility_rows=[];return_rows=[]
    for result,quality in zip(results,qualities):
        costs,reliability=costs_with_reliability(result['geometry'])
        geometric=float(np.dot(costs,reliability))*geometry_weight
        sequence=[-cost for cost in result['step_costs'][:-1]]+[quality-geometric-result['step_costs'][-1]]
        returns=discounted_returns(sequence)
        cost_rows.append(costs);return_rows.append(returns+[0.]*(maximum-len(returns)))
        utility_rows.append(quality-geometric-sum(result['step_costs']))
    scene=torch.stack([result['scene_values'] for result in results])
    returns=scene.new_tensor(return_rows)
    counts=torch.tensor(lengths,device=scene.device)
    mask=torch.arange(maximum,device=scene.device)[None,:]<counts[:,None]
    local_values=torch.stack([F.pad(torch.stack(result['residual_values'])[:,0],(0,maximum-length)) for result,length in zip(results,lengths)])
    logprobs=torch.stack([F.pad(torch.stack(result['residual_logprobs']),(0,maximum-length)) for result,length in zip(results,lengths)])
    entropies=torch.stack([F.pad(torch.stack(result['entropies']),(0,maximum-length)) for result,length in zip(results,lengths)])
    candidate_logprobs=torch.stack([result['candidate_logprob'] for result in results])
    candidate_entropies=torch.stack([result['candidate_entropy'] for result in results])
    candidate_advantage=returns[:,0]-scene[:,0]
    local_advantage=returns-local_values
    rl=-candidate_logprobs*candidate_advantage.detach()+.5*candidate_advantage.square()
    rl=rl-(logprobs*local_advantage.detach()*mask).sum(-1)+.5*(local_advantage.square()*mask).sum(-1)/counts
    entropy=candidate_entropies+(entropies*mask).sum(-1)/counts
    return dict(rl_loss=rl_weight*(rl-.005*entropy),costs=np.asarray(cost_rows),utilities=np.asarray(utility_rows))
