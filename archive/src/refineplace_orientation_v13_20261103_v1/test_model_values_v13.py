"""CPU checks for frozen placement, 160-D action values, paired losses and risk masks."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'refineplace_orientation_v11_20261103_v1')]
from model_v13 import RefinePlaceV13, select_actions
from model_v11 import RefinePlaceV11
from test_model_v11 import DummyInner,inputs
from paired_value_losses import paired_supervision_loss,balanced_mode_loss


def baseline160():
    inner=DummyInner()
    inner.base.actor_body=nn.Sequential(nn.Linear(160,48))
    inner.local=nn.Linear(160,9)
    return RefinePlaceV11(inner,5,2)


def observed_context():
    z=inputs()
    ctx={k:v for k,v in z.items() if k not in ('fused','assets','positions','yaws')}
    ctx['candidate_xyz']=z['positions']
    return z,ctx,torch.randn(2,3,5),torch.randn(2,768),torch.randn(2,3,8,5)


def dummy_outputs(b=1):
    n=b*8
    return dict(task_q=torch.full((n,3),.5,requires_grad=True),
        task_q_logits=torch.zeros(n,3,requires_grad=True),
        collision_cost=torch.full((n,3),.4,requires_grad=True),
        collision_known_logits=torch.zeros(n,3,requires_grad=True),
        mode_logits=torch.zeros(n,2,requires_grad=True),normalized_delta=torch.zeros(n,requires_grad=True),
        action_delta=torch.tensor([-math.pi/4,0.,math.pi/4])[None].expand(n,-1))


class V13Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(31)

    def test_frozen_legacy_outputs_and_160d_new_heads_backward(self):
        baseline=baseline160().eval()
        original={k:v.clone() for k,v in baseline.state_dict().items()}
        z,ctx,features,texts,regions=observed_context()
        state=torch.randn(2,160);centers=torch.randn(2,3);yaws=torch.zeros(2);idx=torch.arange(2)
        kwargs=dict(directional_collision=torch.rand(2,10),object_dims=torch.ones(2,3))
        with baseline.reference_batch(**ctx),baseline.region_batch(regions):
            prior_scores=baseline.candidates_batch(features,texts,z['assets'],ctx['candidate_mask'])[0]
            prior_local=baseline.local_batch_with_reference(state,centers,yaws,idx,**kwargs)
        model=RefinePlaceV13(baseline,2).train()
        self.assertFalse(model.baseline.training)
        self.assertEqual(model.local_state_dim,160)
        self.assertTrue(all(not p.requires_grad for p in model.baseline.parameters()))
        with model.reference_batch(**ctx),model.region_batch(regions):
            scores=model.baseline.candidates_batch(features,texts,z['assets'],ctx['candidate_mask'])[0]
            local=model.baseline.local_batch_with_reference(state,centers,yaws,idx,**kwargs)
            outputs=model.action_values(state.repeat_interleave(8,0),centers.repeat_interleave(8,0),
                torch.arange(8).repeat(2)*math.pi/4,idx.repeat_interleave(8))
        torch.testing.assert_close(scores,prior_scores,atol=0,rtol=0)
        for got,want in zip(local,prior_local):
            torch.testing.assert_close(got,want,atol=0,rtol=0)
        self.assertEqual(outputs['task_q'].shape,(16,3))
        actions,report=select_actions(outputs)
        self.assertTrue(bool((actions==1).all()))
        json.dumps(report,allow_nan=False)
        valid=torch.zeros(2,8,dtype=torch.bool);valid[0,0]=True;valid[1,3]=True
        loss=paired_supervision_loss(outputs,valid,torch.zeros(2,8),torch.ones(2,8,dtype=torch.bool))
        loss['loss'].backward()
        self.assertGreater(float(model.task_value[-1].weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.collision_value[-1].weight.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in model.baseline.parameters()))
        for key,value in original.items():
            torch.testing.assert_close(model.baseline.state_dict()[key],value,atol=0,rtol=0)
        clone=RefinePlaceV13(baseline160(),2)
        clone.load_state_dict(model.state_dict(),strict=True)

    def test_mode_group_balance_does_not_learn_seven_to_one_rotate_prior(self):
        shared=nn.Parameter(torch.zeros(2))
        logits=shared[None].expand(8,-1)
        stop=torch.tensor([True]+[False]*7)
        loss=balanced_mode_loss(logits,stop,torch.zeros(8,dtype=torch.long))
        loss.backward()
        torch.testing.assert_close(shared.grad,torch.zeros(2),atol=1e-7,rtol=0)

    def test_endpoint_modulo_and_far_orientation_have_progress(self):
        outputs=dummy_outputs()
        valid=torch.tensor([[True]+[False]*7])
        losses=paired_supervision_loss(outputs,valid,torch.zeros(1,8),torch.ones(1,8,dtype=torch.bool))
        reward=losses['task_reward_targets']
        self.assertGreater(float(reward[4,0]),float(reward[4,1])) #180 ->135 >180 STOP
        self.assertEqual(float(reward[7,2]),1.) #315 +45 ->0
        self.assertEqual(float(reward[1,0]),1.) #45 -45 ->0
        self.assertEqual(float(reward[0,1]),1.)
        torch.testing.assert_close(reward[7,2],reward[0,1],atol=0,rtol=0)

    def test_unknown_nan_collision_has_no_cost_gradient(self):
        outputs=dummy_outputs()
        valid=torch.tensor([[True]+[False]*7])
        costs=torch.zeros(1,8);costs[0,0]=torch.nan
        known=torch.ones(1,8,dtype=torch.bool);known[0,0]=False
        result=paired_supervision_loss(outputs,valid,costs,known)
        self.assertTrue(bool(torch.isfinite(result['loss'])))
        result['loss'].backward()
        ends=result['endpoint_bin_indices']
        self.assertTrue(bool((outputs['collision_cost'].grad[ends==0]==0).all()))
        self.assertGreater(float(outputs['collision_cost'].grad[ends!=0].abs().sum()),0.)
        self.assertEqual(result['unknown_collision_actions'],3)

    def test_opposite_heading_does_not_force_arbitrary_left_direction(self):
        outputs=dummy_outputs()
        valid=torch.tensor([[True]+[False]*7])
        result=paired_supervision_loss(outputs,valid,torch.zeros(1,8),torch.ones(1,8,dtype=torch.bool))
        reward=result['task_reward_targets']
        torch.testing.assert_close(reward[4,0],reward[4,2],atol=1e-6,rtol=0)
        self.assertFalse(bool(result['target_stop_mask'][4]))
        self.assertFalse(bool(result['target_direction_mask'][4]))
        self.assertEqual(float(result['target_normalized_delta'][4]),0.)
        result['loss'].backward()
        self.assertEqual(float(outputs['normalized_delta'].grad[4]),0.)
        # 45 degrees from 0 has a strictly better left endpoint and still learns.
        self.assertTrue(bool(result['target_direction_mask'][1]))
        self.assertEqual(float(result['target_normalized_delta'][1]),-1.)
        self.assertGreater(float(outputs['normalized_delta'].grad[1].abs()),0.)

    def test_two_symmetric_valid_headings_preserve_equal_action_values(self):
        outputs=dummy_outputs()
        valid=torch.zeros(1,8,dtype=torch.bool);valid[0,1]=True;valid[0,7]=True
        result=paired_supervision_loss(outputs,valid,torch.zeros(1,8),torch.ones(1,8,dtype=torch.bool))
        reward=result['task_reward_targets']
        self.assertEqual(float(reward[0,0]),1.)
        self.assertEqual(float(reward[0,2]),1.)
        self.assertFalse(bool(result['target_stop_mask'][0]))
        self.assertFalse(bool(result['target_direction_mask'][0]))
        self.assertFalse(bool(result['target_direction_mask'][4]))
        result['loss'].backward()
        self.assertEqual(float(outputs['normalized_delta'].grad[0]),0.)
        self.assertEqual(float(outputs['normalized_delta'].grad[4]),0.)

    def test_uncertain_cost_not_zero_or_unconditional_stop_and_known_risk_veto(self):
        values=dict(task_q=torch.tensor([[.8,.5,.5],[.8,.5,.55],[.54,.5,.5],[.8,.8,.8]]),
            collision_cost=torch.tensor([[.95,.3,.8],[.9,.1,.1],[.9,.3,.8],[.1,.1,.1]]),
            collision_known_logits=torch.tensor([[-8.,-8.,-8.],[8.,8.,8.],[-8.,-8.,-8.],[8.,8.,8.]]),
            action_delta=torch.tensor([-math.pi/4,0.,math.pi/4])[None].expand(4,-1))
        indices,report=select_actions(values)
        self.assertEqual(indices.tolist(),[0,2,1,1])
        self.assertTrue(report['uncertain'][0])
        self.assertAlmostEqual(report['selected_collision_cost'][0],.95,places=6)
        self.assertEqual(report['risk_improvement_credit'][0],0.)
        self.assertFalse(report['safety_guarantee'])
        json.dumps(report,allow_nan=False)


if __name__=='__main__':
    unittest.main(verbosity=2)
