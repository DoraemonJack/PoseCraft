"""CPU checks for changed relation-binding paths; no scene or GT files read."""
import copy
from contextlib import contextmanager
import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_v11 import RelationBoundPoseResidual, RefinePlaceV11, RefinePlaceV10


def inputs(batch=2, candidates=3):
    torch.manual_seed(13)
    reference_xyz = torch.tensor([[1., 2., .4], [-2., .2, .8]])[None].repeat(batch, 1, 1)
    return dict(fused=torch.randn(batch, candidates, 128), assets=torch.randn(batch, 768),
                positions=torch.randn(batch, candidates, 3),
                yaws=torch.arange(8)[None, None].expand(batch, candidates, 8) * (math.pi / 4),
                reference_xyz=reference_xyz, reference_extents=torch.ones(batch, 2, 3),
                reference_logits=torch.tensor([[6., -6., -8.], [-6., 6., -8.]])[None].repeat(batch, 1, 1),
                reference_mask=torch.ones(batch, 2, dtype=torch.bool),
                relation_types=torch.tensor([[0, 1]]).repeat(batch, 1),
                relation_mask=torch.ones(batch, 2, dtype=torch.bool),
                candidate_mask=torch.ones(batch, candidates, dtype=torch.bool))


def activate(head):
    with torch.no_grad():
        head.output[-1].weight.normal_(std=.2)
        head.output[-1].bias.fill_(.7)


class DummyReferenceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.inferred_mask = None
        self.inferred_weight = .3


class DummyInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        self.base.actor_body = nn.Sequential(nn.Linear(24, 48))
        self.reference_head = DummyReferenceHead()
        self.yaw_reference = nn.Identity()
        self.local_reference = nn.Identity()
        self.log_std = nn.Parameter(torch.zeros(3))
        self.fused = nn.Linear(5, 128)
        self.score = nn.Linear(128, 8)
        self.value = nn.Linear(128, 3)
        self.local = nn.Linear(24, 9)
        self._context = None

    @contextmanager
    def reference_batch(self, **kwargs):
        self._context = kwargs
        try:
            yield self
        finally:
            self._context = None

    def candidates_batch(self, features, texts, assets, candidate_mask):
        f = self.fused(features)
        return self.score(f).masked_fill(~candidate_mask[..., None], -torch.inf), f, self.value(f.mean(1))

    def local_batch_with_reference(self, state, centers, yaws, task_indices):
        value = self.local(state)
        return value[:, :3], value[:, 3:6], value[:, 6:]


class BoundRelationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_relation_bindings_and_reference_permutations(self):
        z = inputs()
        head = RelationBoundPoseResidual(2, candidate_chunk=2).eval()
        activate(head)
        expected = head(**z)
        permuted = dict(z)
        for key in ('reference_xyz', 'reference_extents', 'reference_mask'):
            permuted[key] = z[key][:, [1, 0]]
        permuted['reference_logits'] = z['reference_logits'][..., [1, 0, 2]]
        torch.testing.assert_close(head(**permuted), expected, atol=1e-6, rtol=1e-5)
        swapped = dict(z, reference_logits=z['reference_logits'][:, [1, 0]])
        self.assertGreater(float((head(**swapped) - expected).abs().max()), 1e-5)
        # Query order is irrelevant; its type and reference hypothesis stay bound.
        permuted = dict(z)
        for key in ('reference_logits', 'relation_types', 'relation_mask'):
            permuted[key] = z[key][:, [1, 0]]
        torch.testing.assert_close(head(**permuted), expected, atol=1e-6, rtol=1e-5)

    def test_missing_reference_padding_and_periodicity(self):
        z = inputs()
        head = RelationBoundPoseResidual(2).eval()
        activate(head)
        z['reference_mask'][0] = False
        z['candidate_mask'][1, -1] = False
        result = head(**z)
        self.assertTrue(torch.equal(result[0], torch.zeros_like(result[0])))
        self.assertTrue(torch.equal(result[1, -1], torch.zeros_like(result[1, -1])))
        torch.testing.assert_close(head(**dict(z, yaws=z['yaws'] + 2 * math.pi)), result, atol=2e-6, rtol=1e-5)
        empty = dict(z, reference_xyz=z['reference_xyz'][:, :0], reference_extents=z['reference_extents'][:, :0],
                     reference_mask=z['reference_mask'][:, :0], reference_logits=z['reference_logits'][..., -1:])
        self.assertTrue(torch.equal(head(**empty), torch.zeros_like(result)))

    def test_backward_finite_and_asset_conditioned_frame(self):
        z = inputs()
        z['assets'] = z['assets'].requires_grad_()
        z['positions'] = z['positions'].requires_grad_()
        head = RelationBoundPoseResidual(2)
        activate(head)
        head(**z).square().mean().backward()
        self.assertTrue(torch.isfinite(z['assets'].grad).all())
        self.assertGreater(float(head.numeric_frame.weight.grad.abs().sum()), 0.)
        self.assertTrue(torch.isfinite(z['positions'].grad).all())
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_zero_initialization_old_scores_and_local_interface(self):
        torch.manual_seed(9)
        old = RefinePlaceV10(DummyInner(), 5).eval()
        new = RefinePlaceV11(copy.deepcopy(old.inner), 5, 2).eval()
        absent, extra = new.load_state_dict(old.state_dict(), strict=False)
        self.assertFalse(extra)
        self.assertTrue(all(k.startswith(('bound_pose.', 'bound_local.')) for k in absent))
        z = inputs()
        context = {k: v for k, v in z.items() if k not in ('fused', 'assets', 'positions', 'yaws')}
        context['candidate_xyz'] = z['positions']
        features = torch.randn(2, 3, 5)
        text = torch.randn(2, 768)
        regions = torch.randn(2, 3, 8, 5)
        local = torch.randn(2, 24)
        direction = torch.rand(2, 10)
        args = (local, torch.randn(2, 3), torch.randn(2), torch.tensor([1, 0]))
        extra_args = dict(directional_collision=direction, object_dims=torch.ones(2, 3))
        with old.reference_batch(**context), old.region_batch(regions):
            expected = old.candidates_batch(features, text, z['assets'], z['candidate_mask'])
            expected_local = old.local_batch_with_reference(*args, **extra_args)
        with new.reference_batch(**context), new.region_batch(regions):
            actual = new.candidates_batch(features, text, z['assets'], z['candidate_mask'])
            actual_local = new.local_batch_with_reference(*args, **extra_args)
        for a, b in zip(actual + actual_local, expected + expected_local):
            torch.testing.assert_close(a, b, atol=0., rtol=0.)
        self.assertIsNone(new._bound_context)
        self.assertIsNone(new._bound_assets)

    def test_inferred_downweight_and_local_head_gradient(self):
        new = RefinePlaceV11(DummyInner(), 5, 2)
        activate(new.bound_pose)
        activate(new.bound_local)
        z = inputs()
        ctx = {k: v for k, v in z.items() if k not in ('fused', 'assets', 'positions', 'yaws')}
        explicit = new._bound_forward(new.bound_pose, z['fused'], z['assets'], z['positions'], z['yaws'], ctx)
        new.reference_head.inferred_mask = ctx['relation_mask'].clone()
        inferred_ctx = dict(ctx, relation_mask=torch.zeros_like(ctx['relation_mask']))
        inferred = new._bound_forward(new.bound_pose, z['fused'], z['assets'], z['positions'], z['yaws'], inferred_ctx)
        torch.testing.assert_close(inferred, .3 * explicit, atol=1e-6, rtol=1e-5)
        new.reference_head.inferred_mask = None
        ctx['candidate_xyz'] = z['positions']
        with new.reference_batch(**ctx), new.region_batch(torch.randn(2, 3, 8, 5)):
            new.candidates_batch(torch.randn(2, 3, 5), torch.randn(2, 768), z['assets'], z['candidate_mask'])
            logits, means, _ = new.local_batch_with_reference(torch.randn(2, 24), torch.randn(2, 3),
                         torch.randn(2), torch.tensor([1, 0]), directional_collision=torch.rand(2, 10), object_dims=torch.ones(2, 3))
            self.assertEqual(means.shape, (2, 3))
            (means.square().mean() + logits.square().mean()).backward()
        self.assertGreater(float(new.bound_local.output[-1].weight.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
