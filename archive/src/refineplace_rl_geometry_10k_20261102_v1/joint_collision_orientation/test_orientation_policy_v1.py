import contextlib
import math
import unittest
import numpy as np
import torch
from torch import nn
from paired_orientation_v1 import (AXIS_CONVENTION, numeric_heading, make_positive_pose_pairs,
                                    paired_pose_quality, paired_quality_from_annotations)
from orientation_policy_v1 import (DirectionalReferencePosePolicy, collision_auxiliary_loss,
                                   wrap_checkpoint_policy)
from joint_rollout_v1 import _choices, rollout_batch, directional_observation


class PairedQualityTests(unittest.TestCase):
    def test_angular_and_spatial_witness_must_be_same_pair(self):
        pairs=np.array([[0.,0.,0.,math.pi],[.3,0.,0.,0.]])
        result=paired_pose_quality([0,0,0],0.,pairs)
        self.assertAlmostEqual(result['quality'],math.exp(-2))
        self.assertEqual(result['witness_index'],0)
        self.assertAlmostEqual(result['joint_angle_rad'],math.pi)

    def test_wrap_continuity_and_positive_angular_attraction(self):
        pairs=np.array([[0.,0.,0.,math.pi]])
        a=paired_pose_quality([0,0,0],math.pi-.001,pairs)
        b=paired_pose_quality([0,0,0],-math.pi+.001,pairs)
        self.assertAlmostEqual(a['quality'],b['quality'])
        self.assertGreater(a['quality'],paired_pose_quality([0,0,0],math.pi-.2,pairs)['quality'])

    def test_positive_masks_empty_and_duplicate_union(self):
        xyz=np.array([[0.,0.,0.],[.2,0.,0.]])
        mask=np.zeros((3,8),bool);mask[0,0]=True;mask[1,0]=True;mask[2,4]=True
        pairs=make_positive_pose_pairs(xyz,np.array([0,0,1]),mask)
        self.assertEqual(pairs.shape,(2,4))
        self.assertAlmostEqual(paired_quality_from_annotations([.2,0,0],math.pi,xyz,np.array([0,0,1]),mask)['quality'],1.)
        empty=paired_pose_quality([0,0,0],0.,np.empty((0,4)))
        self.assertTrue(empty['no_positive']);self.assertFalse(empty['has_positive'])
        self.assertIsNone(empty['joint_angle_rad']);self.assertEqual(empty['quality'],0.)
        with self.assertRaises(ValueError):make_positive_pose_pairs(xyz,np.array([0]),np.array([1],np.uint8))

    def test_axis_is_numeric_and_explicitly_not_semantic_front(self):
        np.testing.assert_allclose(numeric_heading(math.pi/2),[0,1,0],atol=1e-14)
        self.assertIn('semantic asset front unverified',AXIS_CONVENTION)


class TinyInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Linear(12,9)
        self.log_std=nn.Parameter(torch.zeros(3))
        self.yaw_reference=nn.Linear(1,1)
        self.local_reference=nn.Linear(1,1)
        self.reference_head=nn.Linear(1,1)
    @property
    def base(self):return self
    @contextlib.contextmanager
    def reference_batch(self,**kwargs):yield self
    def local_batch(self,state):return tuple(self.layer(state).split(3,-1))
    def local_batch_with_reference(self,state,centers,yaws,task_indices):return self.local_batch(state)
    def candidates_batch(self,*args,**kwargs):return args
    def offsets_batch(self,*args,**kwargs):return args


class WrapperTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.inner=TinyInner()
        self.policy=DirectionalReferencePosePolicy(self.inner,local_state_dim=12)
        self.state=torch.randn(2,12)
        self.centers=torch.randn(2,3);self.yaws=torch.tensor([0.,math.pi/2])
        self.indices=torch.arange(2)
        self.direction=torch.zeros(2,10);self.direction[:,1:3]=1.
        self.dims=torch.tensor([[.2,.3,.4],[1.,1.,1.]])
    def call(self):
        return self.policy.local_batch_with_reference(self.state,self.centers,self.yaws,self.indices,
            directional_collision=self.direction,object_dims=self.dims)
    def test_old_checkpoint_strict_load_and_initial_policy_exactly_preserved(self):
        state={k:v.clone() for k,v in self.inner.state_dict().items()}
        loaded=wrap_checkpoint_policy(TinyInner(),state,local_state_dim=12)
        for k,v in state.items():torch.testing.assert_close(loaded.inner.state_dict()[k],v)
        for got,want in zip(self.call(),self.inner.local_batch(self.state)):
            torch.testing.assert_close(got,want,rtol=0,atol=0)
        torch.testing.assert_close(self.policy.last_collision_prediction,torch.full((2,),.5))
        self.assertIs(self.policy.log_std,self.inner.log_std)
        with self.assertRaises(RuntimeError):wrap_checkpoint_policy(TinyInner(),{},local_state_dim=12)
    def test_collision_auxiliary_mask_and_action_gradients(self):
        logits,means,_=self.call();prediction=self.policy.last_collision_prediction
        rows=[dict(collision_predictions=[prediction[0]],collision_targets=[1.],collision_target_masks=[True]),
              dict(collision_predictions=[prediction[1]],collision_targets=[0.],collision_target_masks=[False])]
        aux=collision_auxiliary_loss(rows)
        self.assertGreater(float(aux[0]),0.);self.assertEqual(float(aux[1]),0.)
        (aux.sum()+logits.sum()+means.sum()).backward()
        self.assertGreater(float(self.policy.collision_head.weight.grad.abs().sum()),0.)
        self.assertGreater(float(self.policy.action_residual.weight.grad.abs().sum()),0.)
    def test_unobserved_direction_cannot_change_actions_even_after_learning(self):
        with torch.no_grad():self.policy.action_residual.bias.fill_(2.)
        self.direction[:,1]=0.
        for got,want in zip(self.call(),self.inner.local_batch(self.state)):
            torch.testing.assert_close(got,want,rtol=0,atol=0)
    def test_unknown_open_asset_excluded_from_cost_target(self):
        vector=np.zeros(10,np.float32);vector[0]=.8;vector[1]=1.
        _,target,known=directional_observation({'directional_collision_vector':vector})
        self.assertEqual(target,float(vector[0]));self.assertFalse(known)
    def test_cost_prediction_cannot_copy_its_scalar_target_input(self):
        with torch.no_grad():self.policy.collision_head.weight.normal_()
        self.direction[:,0]=0.
        self.call();before=self.policy.last_collision_prediction.clone()
        self.direction[:,0]=1.
        self.call()
        torch.testing.assert_close(self.policy.last_collision_prediction,before,rtol=0,atol=0)
    def test_empty_auxiliary_is_finite_with_masked_score_padding(self):
        scores=torch.tensor([1.,-torch.inf],requires_grad=True)
        loss=collision_auxiliary_loss([dict(scores=scores)])
        self.assertEqual(float(loss[0]),0.)
        loss.sum().backward()
        torch.testing.assert_close(scores.grad,torch.zeros(2))


class ScriptedPolicy(nn.Module):
    def __init__(self):
        super().__init__();self.scores=nn.Parameter(torch.tensor([10.,1.,2.,3.,9.,4.,5.,6.]))
        self.log_std=nn.Parameter(torch.zeros(3));self.calls=0;self.last_collision_prediction=None
    def candidates_batch(self,features,texts,assets,candidate_mask):
        return self.scores[None,None],torch.zeros(1,1,4),torch.zeros(1,3)
    def offsets_batch(self,fused,candidate_mask,radius=.15):return torch.zeros(1,1,8,2)
    def local_batch_with_reference(self,state,centers,yaws,task_indices,**kwargs):
        self.calls+=1;self.last_collision_prediction=self.scores.new_full((len(state),),.5,requires_grad=True)
        logits=self.scores.new_tensor([[-20.,-20.,20.]] if self.calls==1 else [[-20.,20.,-20.]])
        return logits,torch.zeros(len(state),3),torch.zeros(len(state),3)


class GeometryService:
    def __init__(self):self.tasks=[]
    def query_many(self,tasks):
        output=[]
        for task in tasks:
            self.tasks.append(dict(task));cost=(1.-math.cos(task['yaw']))/2
            vector=np.zeros(10,np.float32);vector[:3]=[cost,1.,1.]
            stats=dict(directional_collision_vector=vector,collision_cost=cost,collision_cost_known=True)
            output.append((np.zeros(26,np.float32),stats,1))
        return output


class RolloutTests(unittest.TestCase):
    def test_same_candidate_new_yaw_not_deduplicated(self):
        state=dict(top=[0,1,2],pose=0,order=[0,1,2],visited_poses={0},visited_candidates={0},
                   centers=np.zeros((1,3)),offsets=None,visited_centers=[np.zeros(3)])
        self.assertEqual(_choices(state,True),[1,2])
    def test_pre_action_target_alignment_and_fixed_query_budget_with_stop(self):
        service=GeometryService();model=ScriptedPolicy()
        observation=dict(scene='synthetic',split='train',asset_uuid='asset',dims=[1.,1.,1.],centers=[[0.,0.,0.]])
        rows=rollout_batch(model,torch.zeros(1,1,2),torch.zeros(1,3),torch.zeros(1,3),
                           torch.ones(1,1,dtype=torch.bool),[observation],service,steps=2,deterministic=True)
        row=rows[0]
        self.assertEqual([x['mode'] for x in row['path']],['switch','stop'])
        self.assertAlmostEqual(row['yaw'],math.pi)
        self.assertEqual(row['collision_targets'],[0.,1.])
        self.assertEqual(row['collision_target_masks'],[True,True])
        self.assertEqual([x['yaw'] for x in row['collision_target_poses']],[0.,math.pi])
        self.assertEqual(len(row['collision_predictions']),2)
        self.assertEqual(row['actual_geometry_queries'],2)
        self.assertEqual(len(service.tasks),2)
        self.assertEqual(row['repeat_count'],0)
        self.assertEqual(row['candidate_revisit_count'],1)
        self.assertLessEqual(len(service.tasks),3)


if __name__=='__main__':unittest.main()
