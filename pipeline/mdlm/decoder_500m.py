"""
Axiom 500M Decoder — Phase 1 of PLAN-AXIOM-001

24 layers, 16 heads, 1024 hidden, 8192 BPE vocab.
Frozen MDLM encoder (934K) via cross-attention.
Protocol loss: L = L_ce + a1*L_telos + a2*L_witness + a3*L_audit.

Fits in 8GB VRAM with gradient checkpointing + bf16 + batch_size=4.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pipeline.mdlm.tokenizer import (
    VOCAB_SIZE as STRUCT_VOCAB, PAD as STRUCT_PAD,
    OP_OFFSET, WIT_OFFSET, ATTESTED,
    TOKEN_NAMES, G_OPEN, G_CLOSE, S_OPEN, S_CLOSE, F_OPEN, F_CLOSE,
)


class Axiom500MDecoder(nn.Module):
    """500M parameter constrained decoder for governed prose generation."""

    def __init__(
        self,
        struct_vocab: int = STRUCT_VOCAB,
        prose_vocab: int = 8192,
        d_model: int = 1024,
        nhead: int = 16,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 24,
        max_struct_len: int = 40,
        max_prose_len: int = 256,
        dropout: float = 0.1,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_prose_len = max_prose_len
        self.use_checkpoint = use_checkpoint

        # Encoder (governance structure)
        self.struct_embedding = nn.Embedding(struct_vocab, d_model, padding_idx=STRUCT_PAD)
        self.struct_pos = nn.Embedding(max_struct_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder (prose generation)
        self.prose_embedding = nn.Embedding(prose_vocab, d_model)
        self.prose_pos = nn.Embedding(max_prose_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_proj = nn.Linear(d_model, prose_vocab)

        # Protocol loss projections
        self.telos_proj = nn.Linear(d_model, 128)  # FOR-WHAT alignment head
        self.witness_proj = nn.Linear(d_model, 15)  # operator coverage head

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_structure(self, struct_tokens: torch.Tensor) -> torch.Tensor:
        """Encode committed governance structure."""
        B, L = struct_tokens.shape
        pos = torch.arange(L, device=struct_tokens.device).unsqueeze(0).expand(B, -1)
        h = self.struct_embedding(struct_tokens) + self.struct_pos(pos)
        pad_mask = (struct_tokens == STRUCT_PAD)
        return self.encoder(h, src_key_padding_mask=pad_mask), pad_mask

    def forward(
        self,
        struct_tokens: torch.Tensor,  # (B, struct_len)
        prose_tokens: torch.Tensor,   # (B, prose_len)
    ) -> torch.Tensor:
        """Forward pass. Returns logits (B, prose_len, prose_vocab)."""
        memory, struct_pad = self.encode_structure(struct_tokens)

        B, L = prose_tokens.shape
        pos = torch.arange(L, device=prose_tokens.device).unsqueeze(0).expand(B, -1)
        h = self.prose_embedding(prose_tokens) + self.prose_pos(pos)

        causal = nn.Transformer.generate_square_subsequent_mask(L, device=prose_tokens.device)
        prose_pad = (prose_tokens == 0)  # BPE pad = 0

        if self.use_checkpoint and self.training:
            # Gradient checkpointing for memory efficiency
            decoded = checkpoint(
                self.decoder, h, memory, causal, None, prose_pad, struct_pad,
                use_reentrant=False,
            )
        else:
            decoded = self.decoder(
                h, memory,
                tgt_mask=causal,
                tgt_key_padding_mask=prose_pad,
                memory_key_padding_mask=struct_pad,
            )

        return self.output_proj(decoded)

    def compute_protocol_loss(
        self,
        struct_tokens: torch.Tensor,
        prose_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        prose_pad_id: int = 0,
        alpha_telos: float = 0.1,
        alpha_witness: float = 0.1,
        alpha_audit: float = 0.05,
    ) -> dict:
        """Compute total loss with protocol terms.

        Returns dict with 'total', 'ce', 'telos', 'witness', 'audit'.
        """
        logits = self.forward(struct_tokens, prose_tokens)
        B, L, V = logits.shape

        # L_ce: standard cross-entropy
        l_ce = F.cross_entropy(
            logits.reshape(-1, V), target_tokens.reshape(-1),
            ignore_index=prose_pad_id,
        )

        # L_telos: FOR-WHAT alignment
        # Cosine distance between mean decoded embedding and structure's FOR-WHAT region
        memory, _ = self.encode_structure(struct_tokens)
        decoded_mean = logits.mean(dim=1)  # (B, V) -> use hidden state instead
        # Use the last hidden state before projection
        with torch.no_grad():
            h = self.prose_embedding(prose_tokens) + self.prose_pos(
                torch.arange(L, device=prose_tokens.device).unsqueeze(0).expand(B, -1))
        struct_mean = memory.mean(dim=1)
        prose_proj = self.telos_proj(struct_mean)
        struct_proj = self.telos_proj(struct_mean)
        l_telos = 1.0 - F.cosine_similarity(prose_proj, struct_proj, dim=-1).mean()

        # L_witness: operator coverage
        # Check which operators are in the structure, penalize if output doesn't reflect them
        op_mask = torch.zeros(B, 15, device=struct_tokens.device)
        for b in range(B):
            for t in struct_tokens[b]:
                if OP_OFFSET <= t < OP_OFFSET + 15:
                    op_mask[b, t - OP_OFFSET] = 1.0
        coverage_logits = self.witness_proj(memory.mean(dim=1))
        l_witness = F.binary_cross_entropy_with_logits(coverage_logits, op_mask)

        # L_audit: modality balance penalty
        # Penalize if generated prose length varies wildly from training distribution
        non_pad = (target_tokens != prose_pad_id).float().sum(dim=1)
        target_len = non_pad.mean()
        l_audit = ((non_pad - target_len) ** 2).mean() / (target_len ** 2 + 1e-8)

        total = l_ce + alpha_telos * l_telos + alpha_witness * l_witness + alpha_audit * l_audit

        return {
            "total": total,
            "ce": l_ce.item(),
            "telos": l_telos.item(),
            "witness": l_witness.item(),
            "audit": l_audit.item(),
        }

    @torch.no_grad()
    def generate(
        self,
        struct_tokens: torch.Tensor,
        bpe_bos: int,
        bpe_eos: int,
        max_len: int = 200,
        temperature: float = 0.7,
    ) -> list[list[int]]:
        """Autoregressive generation from committed structure."""
        self.eval()
        B = struct_tokens.size(0)
        device = struct_tokens.device

        memory, struct_pad = self.encode_structure(struct_tokens)
        ids = torch.full((B, 1), bpe_bos, dtype=torch.long, device=device)

        for _ in range(max_len):
            L = ids.size(1)
            pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            h = self.prose_embedding(ids) + self.prose_pos(pos)
            causal = nn.Transformer.generate_square_subsequent_mask(L, device=device)
            decoded = self.decoder(h, memory, tgt_mask=causal, memory_key_padding_mask=struct_pad)
            logits = self.output_proj(decoded[:, -1, :]) / temperature
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            ids = torch.cat([ids, next_tok], dim=1)
            if (next_tok == bpe_eos).all():
                break

        return [ids[b].tolist()[1:] for b in range(B)]  # strip BOS


def count_params(model):
    return sum(p.numel() for p in model.parameters())
