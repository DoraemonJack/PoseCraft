"""Relation-bound numeric-yaw learning on top of the frozen-interface V1.0.

Reference coordinates are never averaged before pose reasoning. Each relation
keeps a distribution over reference instances, then interacts with the other
relation tokens. The asset-dependent heading frame is learned from numeric yaw
supervision: it is NOT a verified semantic-front annotation. No ground-truth
instance IDs, task labels or evaluator outputs enter the forward interface.
"""
from contextlib import contextmanager
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
V10 = HERE.parent / 'refineplace_v1_0_20261103_v1'
if not V10.is_dir():
    V10 = Path('/path/to/research300/refineplace_v1_0_20261103_v1')
sys.path.insert(0, str(V10))
from model_v10 import RefinePlaceV10, model_from_checkpoint as make_v10


class RelationBoundPoseResidual(nn.Module):
    """Per-instance pose reasoning followed by per-relation attention.

    Candidate chunks bound peak activation allocation. Null-reference mass is
    applied after all biases, so an absent reference cannot introduce a score.
    Output has shape [B, C, Y, outputs]. Candidate masks may contain padding.
    """
    def __init__(self, relation_count, fused_dim=128, outputs=1, asset_dim=768,
                 hidden=48, top_k=4, candidate_chunk=32):
        super().__init__()
        if relation_count < 1 or hidden % 4 or top_k < 1 or candidate_chunk < 1:
            raise ValueError('positive relation/top-k/chunk counts and hidden divisible by four required')
        self.relation_count = int(relation_count)
        self.top_k, self.candidate_chunk = int(top_k), int(candidate_chunk)
        self.outputs, self.asset_dim = int(outputs), int(asset_dim)
        self.fused = nn.Linear(fused_dim, hidden)
        self.asset = nn.Linear(asset_dim, hidden)
        self.relation = nn.Embedding(relation_count, hidden)
        self.numeric_frame = nn.Linear(asset_dim, 2)
        nn.init.zeros_(self.numeric_frame.weight)
        with torch.no_grad():
            self.numeric_frame.bias.copy_(torch.tensor([1., 0.]))
        # fwd/left + unit bearing + dz/log-distance + extent xyz + p/mass.
        self.geometry = nn.Sequential(nn.Linear(11, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.instance_body = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.instance_compatibility = nn.Linear(hidden, 1)
        self.relation_attention = nn.MultiheadAttention(hidden, 4, dropout=0., batch_first=True)
        self.relation_norm = nn.LayerNorm(hidden)
        self.relation_body = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.query_compatibility = nn.Linear(hidden, 1)
        self.output = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, outputs))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, fused, assets, positions, yaws, reference_xyz,
                reference_extents, reference_logits, reference_mask,
                relation_types, relation_mask, candidate_mask):
        if fused.ndim != 3:
            raise ValueError('fused must be [B,C,D]')
        b, c, _ = fused.shape
        if positions.shape != (b, c, 3) or yaws.ndim != 3 or yaws.shape[:2] != (b, c):
            raise ValueError('pose shape mismatch')
        r, q, y = reference_xyz.shape[1], relation_types.shape[1], yaws.shape[-1]
        if (assets.shape != (b, self.asset_dim) or reference_xyz.shape != (b, r, 3)
            or reference_extents.shape != (b, r, 3) or reference_logits.shape != (b, q, r + 1)
            or reference_mask.shape != (b, r) or relation_mask.shape != (b, q)
            or candidate_mask.shape != (b, c)):
            raise ValueError('asset/reference shape mismatch')
        if any(m.dtype != torch.bool for m in (reference_mask, relation_mask, candidate_mask)):
            raise ValueError('boolean masks required')
        if relation_types.dtype != torch.long:
            raise ValueError('long relation codes required')
        if any(not torch.isfinite(t).all() for t in
               (fused, assets, positions, yaws, reference_xyz, reference_extents, reference_logits)):
            raise ValueError('finite inputs required')
        if (reference_extents < 0).any() or ((relation_types[relation_mask] < 0) |
                (relation_types[relation_mask] >= self.relation_count)).any():
            raise ValueError('invalid extents or relation code')
        if not r or not q or not c:
            return fused.new_zeros((b, c, y, self.outputs))
        available = torch.cat((reference_mask, torch.ones((b, 1), device=fused.device, dtype=torch.bool)), -1)
        prior = reference_logits.masked_fill(~available[:, None, :], -torch.inf).softmax(-1)[..., :r]
        weight, ids = prior.topk(min(r, self.top_k), -1)
        k = ids.shape[-1]
        gather = ids[..., None].expand(b, q, k, 3)
        xyz = reference_xyz[:, None].expand(b, q, r, 3).gather(2, gather)
        ext = reference_extents[:, None].expand(b, q, r, 3).gather(2, gather)
        mass = weight.sum(-1)
        active = relation_mask & (mass > 1e-8)
        # Padded relation codes are never embedded with an invalid index.
        rel = self.relation(relation_types.masked_fill(~relation_mask, 0))
        a = self.asset(assets)
        frame = F.normalize(self.numeric_frame(assets), dim=-1, eps=1e-6)
        f = self.fused(fused)
        results = []
        for start in range(0, c, self.candidate_chunk):
            end = min(c, start + self.candidate_chunk)
            n = end - start
            shape = (b, n, y, q, k)
            delta = xyz[:, None, None] - positions[:, start:end, None, None, None, :]
            cos = yaws[:, start:end].cos()[:, :, :, None, None]
            sin = yaws[:, start:end].sin()[:, :, :, None, None]
            forward = delta[..., 0] * cos + delta[..., 1] * sin
            left = -delta[..., 0] * sin + delta[..., 1] * cos
            fx, fy = frame[:, 0, None, None, None, None], frame[:, 1, None, None, None, None]
            forward, left = forward * fx + left * fy, -forward * fy + left * fx
            horizontal = delta[..., :2].square().sum(-1).sqrt().clamp_min(.01)
            distance = torch.log1p(delta.square().sum(-1).sqrt()).expand(shape)
            z = delta[..., 2].expand(shape)
            log_ext = torch.log1p(ext)[:, None, None].expand(*shape, 3)
            p = weight[:, None, None].expand(shape)
            m = mass[:, None, None, :, None].expand(shape)
            geom = torch.stack((forward.clamp(-20, 20), left.clamp(-20, 20),
                                forward / horizontal, left / horizontal, z.clamp(-20, 20), distance,
                                log_ext[..., 0], log_ext[..., 1], log_ext[..., 2], p, m), -1)
            token = self.geometry(geom)
            token = token + f[:, start:end, None, None, None, :] + a[:, None, None, None, None, :]
            token = self.instance_body(token + rel[:, None, None, :, None, :])
            logits = self.instance_compatibility(token).squeeze(-1) + weight.clamp_min(1e-12).log()[:, None, None]
            # All-null rows remain finite and are masked exactly at the output.
            attention = logits.softmax(-1)
            token = (token * attention[..., None]).sum(-2)
            query_mass = (mass * active)[:, None, None, :].expand(b, n, y, q)
            token = token * query_mass[..., None]
            flat = token.reshape(b * n * y, q, -1)
            mask = active[:, None, None].expand(b, n, y, q).reshape(-1, q)
            safe_mask = mask.clone()
            safe_mask[:, 0] |= ~safe_mask.any(-1)
            interaction, _ = self.relation_attention(flat, flat, flat, key_padding_mask=~safe_mask,
                                                     need_weights=False)
            mixed = self.relation_norm(flat + interaction)
            mixed = mixed + self.relation_body(mixed)
            mixed = mixed.reshape(b, n, y, q, -1)
            query_logits = self.query_compatibility(mixed).squeeze(-1)
            query_logits = query_logits.masked_fill(~safe_mask.reshape(b, n, y, q), -torch.inf)
            query_attention = query_logits.softmax(-1) * active[:, None, None]
            average = (mixed * query_attention[..., None]).sum(-2)
            active5 = active[:, None, None, :, None]
            minimum = mixed.masked_fill(~active5, torch.inf).amin(-2)
            maximum = mixed.masked_fill(~active5, -torch.inf).amax(-2)
            exists = active.any(-1)[:, None, None, None]
            minimum = torch.where(exists, minimum, torch.zeros_like(minimum))
            maximum = torch.where(exists, maximum, torch.zeros_like(maximum))
            output = self.output(torch.cat((average, minimum, maximum), -1))
            confidence = (query_mass * query_attention).sum(-1)
            output = output * confidence[..., None]
            output = output.masked_fill(~candidate_mask[:, start:end, None, None], 0.)
            results.append(output)
        return torch.cat(results, 1)


class RefinePlaceV11(RefinePlaceV10):
    """V1.0-compatible backbone with new relation-bound pose/action residuals."""
    def __init__(self, inner, region_dim, relation_count):
        super().__init__(inner, region_dim)
        self.bound_pose = RelationBoundPoseResidual(relation_count)
        self.bound_local = RelationBoundPoseResidual(relation_count, self.local_state_dim, outputs=6)
        self._bound_context = None
        self._bound_assets = None

    @contextmanager
    def reference_batch(self, **kwargs):
        if self._bound_context is not None:
            raise RuntimeError('nested relation-bound context')
        self._bound_context = {key: value.detach().clone() for key, value in kwargs.items()}
        try:
            with super().reference_batch(**kwargs):
                yield self
        finally:
            self._bound_context = None
            self._bound_assets = None

    def _bound_forward(self, module, fused, assets, centers, yaws, context, task_indices=None):
        ctx = {key: value for key, value in context.items() if key != 'candidate_xyz'}
        delta = module(fused, assets, centers, yaws, **ctx)
        # Inferred queries remain hypotheses. Their complete residual is
        # downweighted after biases; they are never promoted to hard conditions.
        inferred = getattr(self.reference_head, 'inferred_mask', None)
        strength = float(getattr(self.reference_head, 'inferred_weight', .3))
        if inferred is not None and strength:
            if task_indices is not None:
                inferred = inferred.index_select(0, task_indices)
            if inferred.shape != ctx['relation_mask'].shape or inferred.dtype != torch.bool:
                raise ValueError('inferred query mask mismatch')
            if (inferred & ctx['relation_mask']).any():
                raise ValueError('explicit and inferred masks must be disjoint')
            if inferred.any():
                inferred_ctx = dict(ctx, relation_mask=inferred)
                delta = delta + strength * module(fused, assets, centers, yaws, **inferred_ctx)
        return delta

    def candidates_batch(self, features, texts, assets, candidate_mask):
        if self._bound_context is None:
            raise RuntimeError('explicit reference context required')
        scores, fused, values = super().candidates_batch(features, texts, assets, candidate_mask)
        self._bound_assets = assets.detach()
        yaws = (torch.arange(8, device=fused.device, dtype=fused.dtype) * (math.pi / 4))[None, None]
        yaws = yaws.expand(*fused.shape[:2], 8)
        ctx = dict(self._bound_context, candidate_mask=candidate_mask)
        delta = self._bound_forward(self.bound_pose, fused, assets.detach(), ctx['candidate_xyz'], yaws, ctx)
        return (scores + delta.squeeze(-1)).masked_fill(~candidate_mask[..., None], -torch.inf), fused, values

    def local_batch_with_reference(self, state, centers, yaws, task_indices, *,
                                   directional_collision, object_dims):
        if self._bound_context is None or self._bound_assets is None:
            raise RuntimeError('candidate forward must precede local policy within reference context')
        logits, means, values = super().local_batch_with_reference(
            state, centers, yaws, task_indices, directional_collision=directional_collision, object_dims=object_dims)
        ctx = {k: v.index_select(0, task_indices) for k, v in self._bound_context.items()
               if k not in ('candidate_xyz', 'candidate_mask')}
        ctx['candidate_mask'] = torch.ones((len(state), 1), dtype=torch.bool, device=state.device)
        delta = self._bound_forward(self.bound_local, state[:, None], self._bound_assets.index_select(0, task_indices),
                                    centers[:, None], yaws[:, None, None], ctx, task_indices)[:, 0, 0]
        return logits + delta[:, :3], means + delta[:, 3:], values


def model_from_checkpoint(d, types, region_dim, checkpoint=None):
    """Load every old parameter strictly before adding zero-output modules."""
    is_v11 = checkpoint is not None and checkpoint.get('architecture') == 'RefinePlaceV11'
    old = make_v10(d, types, region_dim, None if is_v11 else checkpoint)
    model = RefinePlaceV11(old.inner, region_dim, len(types))
    absent, extra = model.load_state_dict(old.state_dict(), strict=False)
    if extra or any(not k.startswith(('bound_pose.', 'bound_local.')) for k in absent):
        raise RuntimeError('V1.0 warm-start parameter mismatch')
    if is_v11:
        model.load_state_dict(checkpoint['joint_policy'], strict=True)
    return model
