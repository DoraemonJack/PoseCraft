import unittest
import torch
from conservative_loss import targets,loss_fn

class ConservativeLossTests(unittest.TestCase):
    def test_valid_state_teaches_exact_stop_and_180_tie_is_symmetric(self):
        valid=torch.zeros(1,8,dtype=torch.bool);valid[0,0]=True
        _,optimal,correct=targets(valid)
        self.assertEqual(optimal[0].tolist(),[False,True,False])
        self.assertEqual(optimal[4].tolist(),[True,False,True])
    def test_gradient_preserves_valid_pose_and_repairs_invalid_pose(self):
        valid=torch.zeros(1,8,dtype=torch.bool);valid[0,0]=True
        logits=torch.zeros(8,3,requires_grad=True)
        output=dict(task_q_logits=logits,task_q=logits.sigmoid(),original_task_logits=torch.zeros(8,3),mode_logits=torch.zeros(8,2,requires_grad=True))
        loss,_=loss_fn(output,valid);loss.backward()
        self.assertLess(float(logits.grad[0,1]),0.)
        self.assertGreater(float(logits.grad[0,0]),0.)
        self.assertLess(float(logits.grad[1,0]),0.)
        self.assertTrue(torch.isfinite(logits.grad).all())
    def test_all_valid_headings_does_not_invent_rotation_targets(self):
        valid=torch.ones(2,8,dtype=torch.bool);_,optimal,_=targets(valid)
        self.assertTrue(optimal[:,1].all());self.assertFalse(optimal[:,[0,2]].any())
        logits=torch.zeros(16,3,requires_grad=True)
        loss,_=loss_fn(dict(task_q_logits=logits,task_q=logits.sigmoid(),original_task_logits=logits.detach(),mode_logits=torch.zeros(16,2,requires_grad=True)),valid)
        self.assertTrue(torch.isfinite(loss));loss.backward()

if __name__=='__main__':unittest.main()
