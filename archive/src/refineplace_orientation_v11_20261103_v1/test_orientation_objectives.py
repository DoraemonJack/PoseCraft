"""Checks for the changed supervision/reward boundaries; no dataset is read."""
import math
import unittest
import numpy as np
import torch
from orientation_objectives import supervised_full8, dense_paired_pose_quality


def example_label(n=1, yaws=(0,)):
    positive = np.zeros((n, 8), bool)
    positive[0, list(yaws)] = True
    rank = positive.astype(float)
    if rank.sum():
        rank /= rank.sum()
    return dict(known=np.array([True] + [False] * 7), positive=positive,
                rank=rank, offset_xy=np.zeros((n, 8, 2)), offset_mask=np.zeros((n, 8), bool))


class SupervisionTests(unittest.TestCase):
    def test_previously_unknown_yaw_receives_gradient(self):
        scores = torch.zeros(1, 1, 8, requires_grad=True)
        offsets = torch.zeros(1, 1, 8, 2, requires_grad=True)
        loss, report = supervised_full8(scores, offsets, {0: example_label()}, [0])
        loss.sum().backward()
        self.assertLess(scores.grad[0, 0, 0].item(), 0)
        self.assertTrue(bool((scores.grad[0, 0, 1:] > 0).all()))
        self.assertEqual(report['positive_count'].item(), 1)
        self.assertEqual(report['negative_count'].item(), 7)
        self.assertEqual(report['old_known_yaw_count'].item(), 1)

    def test_multiple_valid_yaws_are_not_mutually_exclusive(self):
        scores = torch.zeros(1, 1, 8, requires_grad=True)
        loss, _ = supervised_full8(scores, torch.zeros(1, 1, 8, 2),
                                  {0: example_label(yaws=(0, 4))}, [0])
        loss.sum().backward()
        self.assertLess(scores.grad[0, 0, 0].item(), 0)
        self.assertLess(scores.grad[0, 0, 4].item(), 0)
        self.assertGreater(scores.grad[0, 0, 2].item(), 0)

    def test_padding_is_finite_invariant_and_has_no_gradient(self):
        label = example_label()
        a = torch.randn(1, 1, 8)
        ref, _ = supervised_full8(a, torch.zeros(1, 1, 8, 2), {0: label}, [0])
        padded = torch.cat((a, torch.full_like(a, -torch.inf)), dim=1).requires_grad_()
        offsets = torch.zeros(1, 2, 8, 2, requires_grad=True)
        loss, _ = supervised_full8(padded, offsets, {0: label}, [0],
                                  torch.tensor([[True, False]]))
        self.assertTrue(torch.isfinite(loss).all())
        torch.testing.assert_close(ref, loss)
        loss.sum().backward()
        self.assertTrue((padded.grad[:, 1] == 0).all())
        self.assertTrue((offsets.grad[:, 1] == 0).all())

    def test_empty_active_rows_zero_loss(self):
        scores = torch.full((1, 2, 8), -torch.inf, requires_grad=True)
        offsets = torch.zeros(1, 2, 8, 2, requires_grad=True)
        loss, report = supervised_full8(scores, offsets, {0: example_label(2)}, [0],
                                       torch.zeros(1, 2, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(report['valid_candidate_count'].item(), 0)
        loss.sum().backward()
        self.assertTrue((scores.grad == 0).all())

    def test_balancing_prevents_negative_count_dominating(self):
        for n in (1, 128):
            scores = torch.zeros(1, n, 8, requires_grad=True)
            loss, _ = supervised_full8(scores, torch.zeros(1, n, 8, 2),
                                      {0: example_label(n)}, [0],
                                      positive_mass_weight=0., rank_weight=0., offset_weight=0.)
            loss.sum().backward()
            self.assertAlmostEqual(scores.grad[0, 0, 0].item(), -.25)
            self.assertAlmostEqual(scores.grad.sum().item(), 0., places=6)

    def test_offsets_kept_and_unlabelled_candidate_rejected(self):
        label = example_label()
        label['offset_mask'][0, 0] = True
        label['offset_xy'][0, 0] = [.15, 0]
        offsets = torch.zeros(1, 1, 8, 2, requires_grad=True)
        loss, report = supervised_full8(torch.zeros(1, 1, 8), offsets, {0: label}, [0])
        self.assertGreater(report['offset_loss'].item(), 0.)
        loss.sum().backward()
        self.assertLess(offsets.grad[0, 0, 0, 0].item(), 0.)
        with self.assertRaisesRegex(ValueError, 'no label row'):
            supervised_full8(torch.zeros(1, 2, 8), torch.zeros(1, 2, 8, 2),
                             {0: label}, [0], torch.ones(1, 2, dtype=torch.bool))


class RewardTests(unittest.TestCase):
    def test_orientation_signal_survives_thirty_centimetres(self):
        pairs = np.array([[0., 0., 0., 0.]])
        right = dense_paired_pose_quality([.3, 0., 0.], 0., pairs, facing_active=True)
        wrong = dense_paired_pose_quality([.3, 0., 0.], math.pi, pairs, facing_active=True)
        self.assertLess(right['paired_quality'], .000124)
        self.assertEqual(right['angular_quality'], 1.)
        self.assertEqual(wrong['angular_quality'], 0.)
        self.assertGreater(right['quality'] - wrong['quality'], .249)

    def test_no_facing_no_evidence_or_too_far_disables_extra(self):
        pairs = np.array([[0., 0., 0., 0.]])
        for center, data, facing in [([.3, 0, 0], pairs, False),
                                    ([0, 0, 0], np.empty((0, 4)), True),
                                    ([.6, 0, 0], pairs, True)]:
            result = dense_paired_pose_quality(center, 0., data, facing_active=facing)
            self.assertFalse(result['enabled'])
            self.assertEqual(result['angular_quality'], 0.)
            self.assertEqual(result['quality'], result['paired_quality'])

    def test_far_witness_does_not_supply_a_better_yaw(self):
        pairs = np.array([[0., 0., 0., 0.], [2., 0., 0., math.pi]])
        result = dense_paired_pose_quality([.3, 0., 0.], math.pi, pairs, facing_active=True)
        self.assertEqual(result['nearest_position'], [0., 0., 0.])
        self.assertEqual(result['angular_quality'], 0.)

    def test_same_position_multimodal_and_circular(self):
        pairs = np.array([[0., 0., 0., 0.], [0., 0., 0., math.pi]])
        right = dense_paired_pose_quality([0., 0., 0.], -math.pi, pairs, facing_active=True)
        self.assertEqual(right['angular_quality'], 1.)
        self.assertEqual(right['local_yaw_count'], 2)

    def test_equal_distance_tie_is_row_order_invariant(self):
        pairs = np.array([[-.1, 0., 0., 0.], [.1, 0., 0., math.pi]])
        a = dense_paired_pose_quality([0., 0., 0.], 0., pairs, facing_active=True)
        b = dense_paired_pose_quality([0., 0., 0.], 0., pairs[::-1], facing_active=True)
        self.assertEqual(a['nearest_position'], b['nearest_position'])
        self.assertEqual(a['angular_quality'], b['angular_quality'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
