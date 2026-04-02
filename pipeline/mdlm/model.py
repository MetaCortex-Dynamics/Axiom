"""
MDLM — Masked Diffusion Language Model for governed structures.

Architecture:
  - Small transformer encoder (4 layers, 128 dim, 4 heads)
  - Absorbing-state masking: tokens → <MASK> at rate alpha(t)
  - Denoising: predict original token from masked sequence
  - Loss: cross-entropy on masked positions (reweighted MLM)

Masking schedules:
  A: hierarchical hierarchical (Tier 1 → Tier 2 → Tier 3+readiness)
  B: flat hierarchical (operators only, no readiness staging)
  C: Uniform random
  D: inverted inverted

Per PLAN-GHA-002 §4.4: A > B > C > D predicted.
"""

from __future__ import annotations

import math
import random
from enum import Enum

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from pipeline.mdlm.tokenizer import (
    VOCAB_SIZE, MASK, PAD, NEVER_MASKED,
    TIER_1_TOKENS, TIER_2_TOKENS, TIER_3_TOKENS,
    get_tier, pad_sequence,
)


class MaskingSchedule(str, Enum):
    """Masking schedule variants for the hierarchical hypothesis test."""
    HIERARCHICAL = "A"  # hierarchical: Tier 1 → Tier 2 → CL+PreAttest
    FLAT = "B"  # flat: operators only, uniform within tiers
    UNIFORM = "C"           # uniform random over all maskable tokens
    INVERTED = "D"      # inverted: CL first, Tier 1 last


if HAS_TORCH:

    class StructureModel(nn.Module):
        """Small transformer for governed structure denoising."""

        def __init__(
            self,
            vocab_size: int = VOCAB_SIZE,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 4,
            max_len: int = 40,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.d_model = d_model
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
            self.pos_embedding = nn.Embedding(max_len, d_model)
            self.timestep_embedding = nn.Embedding(1000, d_model)  # diffusion timestep

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_proj = nn.Linear(d_model, vocab_size)

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """
            x: (batch, seq_len) — token ids with some positions masked
            t: (batch,) — diffusion timestep (0 = clean, T = fully masked)
            Returns: (batch, seq_len, vocab_size) — logits for each position
            """
            B, L = x.shape
            positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)

            h = self.embedding(x) + self.pos_embedding(positions)
            h = h + self.timestep_embedding(t).unsqueeze(1)

            pad_mask = (x == PAD)
            h = self.transformer(h, src_key_padding_mask=pad_mask)
            return self.output_proj(h)


    def apply_mask(
        tokens: torch.Tensor,
        mask_rate: float,
        schedule: MaskingSchedule,
        timestep: int = 0,
        total_timesteps: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply masking schedule to a batch of token sequences.

        Returns:
            masked_tokens: tokens with some positions replaced by MASK
            mask_positions: boolean tensor (True = was masked)
        """
        B, L = tokens.shape
        masked = tokens.clone()
        mask_positions = torch.zeros(B, L, dtype=torch.bool, device=tokens.device)

        for b in range(B):
            for i in range(L):
                tok = tokens[b, i].item()
                if tok in NEVER_MASKED:
                    continue

                tier = get_tier(tok)
                if tier == 0:
                    continue

                # Compute per-tier mask probability based on schedule
                p = _tier_mask_prob(tier, mask_rate, schedule, timestep, total_timesteps)

                if random.random() < p:
                    masked[b, i] = MASK
                    mask_positions[b, i] = True

        return masked, mask_positions


    def _tier_mask_prob(
        tier: int,
        base_rate: float,
        schedule: MaskingSchedule,
        timestep: int,
        total_timesteps: int,
    ) -> float:
        """Compute mask probability for a token based on its tier and the schedule."""
        t_frac = timestep / max(total_timesteps, 1)  # 0 = clean, 1 = fully masked

        if schedule == MaskingSchedule.UNIFORM:
            return base_rate

        if schedule == MaskingSchedule.HIERARCHICAL:
            # Tier 1 (Tier 1): masked last, unmasked first
            # Tier 3 (CL+PreAttest): masked first, unmasked last
            if tier == 1:
                return base_rate * max(0.0, (t_frac - 0.66) / 0.34) if t_frac > 0.66 else 0.0
            elif tier == 2:
                return base_rate * max(0.0, (t_frac - 0.33) / 0.34) if t_frac > 0.33 else 0.0
            else:  # tier 3
                return base_rate * min(1.0, t_frac / 0.33)

        if schedule == MaskingSchedule.FLAT:
            # Same as 369 but witness tokens are tier 2 priority
            if tier == 1:
                return base_rate * max(0.0, (t_frac - 0.66) / 0.34) if t_frac > 0.66 else 0.0
            elif tier == 2:
                return base_rate * max(0.0, (t_frac - 0.33) / 0.34) if t_frac > 0.33 else 0.0
            else:
                return base_rate * min(1.0, t_frac / 0.33)

        if schedule == MaskingSchedule.INVERTED:
            # Inverted: Tier 1 masked first
            if tier == 1:
                return base_rate * min(1.0, t_frac / 0.33)
            elif tier == 2:
                return base_rate * max(0.0, (t_frac - 0.33) / 0.34) if t_frac > 0.33 else 0.0
            else:
                return base_rate * max(0.0, (t_frac - 0.66) / 0.34) if t_frac > 0.66 else 0.0

        return base_rate


    def compute_loss(
        model: StructureModel,
        batch: torch.Tensor,
        schedule: MaskingSchedule,
        timestep: int,
        total_timesteps: int = 100,
        mask_rate: float = 0.5,
    ) -> torch.Tensor:
        """Compute MDLM denoising loss on a batch.

        Loss = cross-entropy on masked positions only.
        Returns zero loss if no positions were masked (avoids NaN).
        """
        device = next(model.parameters()).device
        batch = batch.to(device)
        t_tensor = torch.full((batch.size(0),), timestep, dtype=torch.long, device=device)

        masked, mask_pos = apply_mask(batch, mask_rate, schedule, timestep, total_timesteps)

        # If nothing was masked, return zero loss
        if not mask_pos.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        logits = model(masked, t_tensor)

        # Loss only on masked positions
        loss = F.cross_entropy(
            logits[mask_pos],
            batch[mask_pos],
            ignore_index=PAD,
        )
        return loss


    def generate(
        model: StructureModel,
        num_samples: int,
        seq_len: int,
        schedule: MaskingSchedule,
        total_timesteps: int = 50,
        g_slots: int = 3,
        s_slots: int = 4,
        f_slots: int = 3,
    ) -> torch.Tensor:
        """Generate governed structures by template-guided iterative unmasking.

        The channel_b frame is IMPOSED (governance), not learned:
          <BOS> <G> [MASK slots] </G> <S> [MASK slots] </S> <F> [MASK slots] </F>
                [witness MASK slots] <EOS>

        The model fills in operator tokens and witness attestation status.
        This respects PROPOSE ≠ PROMOTE: the frame is governance,
        the content is what the kernel crystallizes.

        g_slots, s_slots, f_slots: number of operator MASK slots per modality.
        Should match the corpus distribution.
        """
        device = next(model.parameters()).device
        from pipeline.mdlm.tokenizer import (
            BOS, EOS, G_OPEN, G_CLOSE, S_OPEN, S_CLOSE, F_OPEN, F_CLOSE,
            WIT_OFFSET, ATTESTED,
        )

        # Build template with configurable slot counts
        template = [BOS, G_OPEN] + [MASK] * g_slots + [G_CLOSE,
                    S_OPEN] + [MASK] * s_slots + [S_CLOSE,
                    F_OPEN] + [MASK] * f_slots + [F_CLOSE]
        # 7 witness pairs: WIT_TOKEN MASK
        for w in range(7):
            template.extend([WIT_OFFSET + w, MASK])
        template.append(EOS)

        # Pad to seq_len
        while len(template) < seq_len:
            template.append(PAD)
        template = template[:seq_len]

        samples = torch.tensor([template] * num_samples, dtype=torch.long, device=device)

        model.eval()
        with torch.no_grad():
            for step in range(total_timesteps, -1, -1):
                t_tensor = torch.full((num_samples,), step, dtype=torch.long, device=device)
                logits = model(samples, t_tensor)
                probs = F.softmax(logits, dim=-1)

                t_frac = step / total_timesteps

                for b in range(num_samples):
                    for i in range(seq_len):
                        if samples[b, i].item() != MASK:
                            continue

                        pred = torch.multinomial(probs[b, i], 1).item()
                        tier = get_tier(pred)

                        # Tier-based unmasking schedule
                        should_unmask = False
                        if schedule == MaskingSchedule.HIERARCHICAL:
                            should_unmask = (tier == 1 and t_frac < 0.33) or \
                                          (tier == 2 and 0.33 <= t_frac < 0.66) or \
                                          (tier == 3 and t_frac >= 0.66) or \
                                          (step == 0)  # unmask everything at final step
                        else:
                            should_unmask = True

                        if should_unmask:
                            samples[b, i] = pred

            # Final pass: force-unmask any remaining MASK tokens
            remaining = (samples == MASK)
            if remaining.any():
                t_tensor = torch.zeros((num_samples,), dtype=torch.long, device=device)
                logits = model(samples, t_tensor)
                for b in range(num_samples):
                    for i in range(seq_len):
                        if samples[b, i].item() == MASK:
                            samples[b, i] = logits[b, i].argmax().item()

        return samples
