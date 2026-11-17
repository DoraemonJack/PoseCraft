import unittest,math
import torch
from projected_loss import project_gradient,repair_replay

class RepairProtectionTests(unittest.TestCase):
    def test_conflicting_gradient_is_projected_orthogonal(self):
        result,r=project_gradient([torch.tensor([-2.,3.])],[torch.tensor([1.,0.])])
        torch.testing.assert_close(result[0],torch.tensor([0.,3.]))
        self.assertTrue(r['conflict']);self.assertGreaterEqual(r['dot_after'],-1e-7)
    def test_nonconflicting_gradient_and_zero_constraint_are_unchanged(self):
        g=torch.tensor([2.,3.])
        for constraint in [torch.tensor([1.,0.]),torch.zeros(2)]:
            result,r=project_gradient([g],[constraint]);self.assertTrue(torch.equal(result[0],g));self.assertFalse(r['conflict'])
    def test_replay_contains_only_nodes_of_successful_frozen_repairs(self):
        q=torch.full((8,3),.1);q[:,1]=.8;q[1:3,0]=.95;q[1:3,1]=.1
        v=dict(task_q=q,collision_cost=torch.zeros(8,3),collision_known_logits=torch.ones(8,3)*10,
            action_delta=torch.tensor([[-math.pi/4,0.,math.pi/4]]).expand(8,-1))
        valid=torch.zeros(1,8,dtype=torch.bool);valid[0,0]=True
        actions,mask=repair_replay(v,valid)
        self.assertEqual(mask.tolist(),[True,True,True,False,False,False,False,False])
        self.assertEqual(actions[:3].tolist(),[1,0,0])

if __name__=='__main__':unittest.main()
