"""Position head with an independently weighted inferred-reference residual.

The caller supplies inferred query mask for each batch, and clears it after use.
The normal relation mask contains explicit relations only, so other pose/action
branches do not mistakenly treat a hypothesis as a hard language condition.
"""
from reference_conditioned_position_head_v2 import ReferenceConditionedPositionHead


class InferredReferenceHead(ReferenceConditionedPositionHead):
    def __init__(self, *args, inferred_weight=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0. <= inferred_weight <= 1.:
            raise ValueError('inferred weight out of range')
        self.inferred_weight = inferred_weight
        self.inferred_mask = None

    def forward(self, fused, candidate_xyz, reference_xyz, reference_extents,
                reference_logits, reference_mask, relation_types, relation_mask,
                candidate_mask):
        args = (fused, candidate_xyz, reference_xyz, reference_extents,
                reference_logits, reference_mask, relation_types)
        base = super().forward(*args, relation_mask, candidate_mask)
        if self.inferred_mask is None or self.inferred_weight == 0.:
            return base
        mask = self.inferred_mask
        if mask.shape != relation_mask.shape or mask.dtype != relation_mask.dtype:
            raise ValueError('inferred query mask mismatch')
        if (mask & relation_mask).any():
            raise ValueError('inferred and explicit queries must be disjoint')
        if not mask.any():
            return base
        extra = super().forward(*args, mask, candidate_mask)
        return base + self.inferred_weight * extra
