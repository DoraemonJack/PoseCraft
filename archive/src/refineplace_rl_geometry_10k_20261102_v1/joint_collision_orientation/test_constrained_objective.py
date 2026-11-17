import unittest
import torch
from constrained_objective import constrained_rl_losses, update_multiplier, collision_target


def trajectory(known):
    stats = dict(scene_geometry_available=True, support_known_fraction=1., support_contact_fraction=1.,
                 support_gap_fraction=0., collision_noncontact_pairs=0, collision_cost=.8,
                 collision_cost_known=known)
    return dict(geometry=stats, residual_values=[torch.zeros(3, requires_grad=True)],
                residual_logprobs=[torch.tensor(-1., requires_grad=True)],
                candidate_logprob=torch.tensor(-1., requires_grad=True),
                candidate_entropy=torch.tensor(0.), entropies=[torch.tensor(0.)],
                scene_values=torch.zeros(3, requires_grad=True), step_costs=[0.])


class ObjectiveTests(unittest.TestCase):
    def test_collision_increases_policy_penalty(self):
        free = trajectory(True); costly = trajectory(True)
        free['geometry']['collision_cost'] = 0.
        a = constrained_rl_losses([free], [1.], collision_multiplier=2.)
        b = constrained_rl_losses([costly], [1.], collision_multiplier=2.)
        a['rl_loss'].sum().backward(); b['rl_loss'].sum().backward()
        self.assertGreater(float(costly['candidate_logprob'].grad), float(free['candidate_logprob'].grad))

    def test_unknown_cost_has_no_cost_critic_gradient(self):
        result = trajectory(False)
        loss = constrained_rl_losses([result], [.4], collision_multiplier=2.)
        loss['rl_loss'].sum().backward()
        self.assertEqual(float(result['scene_values'].grad[1]), 0.)
        self.assertEqual(float(result['residual_values'][0].grad[1]), 0.)

    def test_multiplier_uses_only_observed_cost(self):
        self.assertEqual(update_multiplier(1., [1.], [False]), 1.)
        self.assertGreater(update_multiplier(1., [.8], [True]), 1.)
        self.assertLess(update_multiplier(1., [0.], [True]), 1.)

    def test_complete_nonintersection_is_explicit_observation_proxy(self):
        s = trajectory(False)['geometry']
        s['collision_complete'] = True
        value, known, mode = collision_target(s)
        self.assertEqual(value, 0.)
        self.assertTrue(known)
        self.assertEqual(mode, 'observed_surface_nonintersection_proxy')
        s['collision_complete'] = False
        self.assertFalse(collision_target(s)[1])


if __name__ == '__main__':
    unittest.main()
