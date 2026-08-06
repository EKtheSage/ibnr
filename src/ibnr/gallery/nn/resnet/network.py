"""The residual convolutional triangle network: 2-D conv encoder over the
(origin x dev) grid with a mixture density head. This module imports torch -
only import it from inside the entry's fit/predict paths (``ibnr.gallery``
must import without the [nn] extra; see model.py and CLAUDE.md's "torch must
never be imported at module level" rule).

The encoder-body comparison beside the transformer (global attention) and the
per-cell MLP: residual 3x3 convolutions see LOCAL (origin, dev) neighborhoods
and grow their receptive field with depth. The head and its loss/sampler are
IMPORTED from the transformer's network module, not copied - the three bodies
share one MDN so a leaderboard difference is attributable to the encoder.

Leakage discipline, spelled out because a conv body makes it easy to get
wrong: convolutions see the WHOLE grid - unlike an attention mask there is no
architectural way to hide a cell from the receptive field. What protects
against conditioning on the future is the INPUT construction: each value
channel is zeroed outside ITS OWN context mask, and the F masks ride along as
explicit input channels so the network can tell "zero because
padding/future/unobserved" from "zero because a zero increment was observed"
(the contract's zeros are padding, never data - every consumer gates on
``x_obs``). The mask is per channel and not per cell because a feature can be
missing where the target is observed, and a single flag would present the
contract's padding as an observed zero. The no-leak tests in
tests/test_resnet.py poison a beyond-cutoff cell and a masked-off feature
channel and assert bit-identical outputs; they are the load-bearing tests for
this body.

Model writeup: card.md. Data contract feeding it: kernels/nn_contract.py."""

from __future__ import annotations

import torch
from torch import nn

from ibnr.gallery.nn.resnet.config import ResNetConfig

# ONE mixture head across the NN family: the loss and the sampler are the
# transformer's, imported so the three encoder bodies cannot drift apart.
from ibnr.gallery.nn.transformer.network import mdn_nll, mdn_sample

__all__ = ["ResidualBlock", "TriangleResNet", "mdn_nll", "mdn_sample"]


class ResidualBlock(nn.Module):
    """Pre-activation residual block: GN -> GELU -> 3x3 conv, twice, + skip.

    GroupNorm, never BatchNorm: under calendar-cutoff augmentation every
    cohort in a batch is conditioned at its OWN cutoff, so batch statistics
    would couple activations across different conditioning states (train-time
    leakage between augmentation tasks) and eval-time running stats would be
    an average over cutoffs that matches none of them. GroupNorm normalizes
    within a single sample, so nothing crosses the batch axis."""

    def __init__(self, channels: int, n_groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(n_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y = self.conv1(nn.functional.gelu(self.norm1(h)))
        y = self.conv2(nn.functional.gelu(self.norm2(y)))
        return h + y


class TriangleResNet(nn.Module):
    """Residual conv encoder over the full (origin x dev) grid of one cohort.

    Input is an image-like stack of per-cell channels:

    - the F standardized value channels, each ZEROED outside ITS OWN context
      mask (the model must predict non-context cells, not read them);
    - those F masks as explicit 0/1 channels - the only way the network can
      distinguish an observed zero from padding, since the contract's zeros
      are never data. One per value channel, because observedness is per
      channel (``nn_contract``'s ``x_obs``): a feature missing at a cell whose
      target is observed must not be read as a feature of zero;
    - the RELATIVE calendar channel: distance past the conditioning cutoff,
      clamped to [0, n_d] exactly like the transformer's dist_emb, then
      scaled by 1/n_d. Relative, never absolute: forecast diagonals lie past
      the training window where an absolute calendar encoding never received
      a gradient (the documented v1/v2 transformer defect);
    - broadcast cohort conditioning: LOB embedding + normalized log premium,
      constant over the grid.

    A 3x3 conv stem lifts this stack to ``channels``, ``n_blocks`` residual
    blocks mix local (origin, dev) neighborhoods - the receptive field grows
    by 2 cells per conv, so depth controls how far development-year and
    calendar structure can propagate - and a 1x1 conv head reads a K-Gaussian
    MDN over the normalized incremental loss ratio of every cell. Forward
    signature and output shapes match ``TriangleTransformer`` exactly, so the
    entry, the shared training loop and the rollout drive either body
    unchanged."""

    def __init__(
        self, cfg: ResNetConfig, *, n_lob: int, n_features: int, n_w: int, n_d: int
    ) -> None:
        super().__init__()
        self.cfg = cfg
        # n_w = number of origin periods (accident years), n_d = number of dev
        # lags; the grid is the conv's (H, W) plane, one cell = one pixel.
        self.n_w, self.n_d = n_w, n_d
        # cohort conditioning: LOB identity + size (normalized log premium),
        # broadcast onto every cell. No company embedding on purpose (~600
        # companies x ~55 cells would just memorize; card.md).
        self.lob_emb = nn.Embedding(n_lob, cfg.lob_embedding_dim)
        # input stack: F values + F context flags (one per value channel) +
        # relative-calendar channel + lob embedding + log premium, all as
        # per-cell channels. At F = 1 this is the width the per-cell flag gave.
        in_channels = 2 * n_features + 1 + cfg.lob_embedding_dim + 1
        self.stem = nn.Conv2d(in_channels, cfg.channels, kernel_size=3, padding=1)
        # channel dropout (whole feature maps), the conv analogue of the
        # transformer's token dropout
        self.drop = nn.Dropout2d(cfg.dropout)
        self.blocks = nn.ModuleList(
            [ResidualBlock(cfg.channels, cfg.n_groups) for _ in range(cfg.n_blocks)]
        )
        self.out_norm = nn.GroupNorm(cfg.n_groups, cfg.channels)
        # MDN head: per cell, 3 params (mixture logit, mu, raw sigma) x K
        # components, read by a 1x1 conv (a per-cell linear layer).
        self.head = nn.Conv2d(cfg.channels, 3 * cfg.n_components, kernel_size=1)

        # 1-based calendar diagonal per cell, matching nn_contract's cal_idx
        # convention. A buffer: follows .to(device), never trains, and is a
        # pure function of (n_w, n_d) so it is not checkpointed.
        w_idx, d_idx = torch.meshgrid(torch.arange(n_w), torch.arange(n_d), indexing="ij")
        self.register_buffer("cal_idx", w_idx + d_idx + 1, persistent=False)  # (W, D)

    def forward(
        self,
        x: torch.Tensor,  # (B, F, W, D) normalized values (junk allowed off-context)
        context_mask: torch.Tensor,  # (B, F, W, D) bool - per channel
        lob_idx: torch.Tensor,  # (B,) long
        log_premium: torch.Tensor,  # (B,) normalized
        cutoff: torch.Tensor,  # (B,) long - 1-based conditioning diagonal
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode one batch of cohort grids and read an MDN over every cell.

        B = cohorts in the batch, F = channels, W = origins, D = dev lags,
        K = mixture components. Returns log_pi, mu, sigma each (B, W, D, K) -
        a K-Gaussian mixture over the normalized incremental loss ratio of
        every cell. Only ``context_mask``-true cells contribute their values,
        CHANNEL BY CHANNEL; the rest are zeroed BEFORE the first convolution,
        so no receptive field ever touches a value the mask does not declare
        observed (the model must predict those cells, not read them)."""
        b = x.shape[0]
        flags = context_mask.to(x.dtype)  # (B, F, W, D)
        # zero out non-context values: the ONLY thing standing between a conv
        # body and reading the future (see the module docstring). The flag
        # channels ride along so "zeroed" is distinguishable from "zero", one
        # per value channel so an absent feature is not an observed zero.
        vals = x * flags  # (B, F, W, D)
        # relative calendar position of each cell: diagonals past the cutoff,
        # clamped to [0, n_d] (the transformer's dist_emb clamp), scaled to
        # [0, 1] so the channel is on the same order as the standardized values.
        dist = (self.cal_idx[None] - cutoff[:, None, None]).clamp(0, self.n_d)  # (B, W, D)
        dist_ch = dist.unsqueeze(1).to(x.dtype) / self.n_d  # (B, 1, W, D)
        # per-cohort conditioning (LOB + size), broadcast onto every cell
        cond = torch.cat([self.lob_emb(lob_idx), log_premium.unsqueeze(-1)], dim=-1)  # (B, E+1)
        cond_ch = cond[:, :, None, None].expand(-1, -1, self.n_w, self.n_d)
        inp = torch.cat([vals, flags, dist_ch, cond_ch], dim=1)  # (B, 2F+E+2, W, D)
        h = self.drop(self.stem(inp))  # (B, channels, W, D)
        for block in self.blocks:
            h = block(h)  # residual local mixing; +2 cells of receptive field per conv
        # head -> (B, 3K, W, D), rearranged to per-cell (B, W, D, 3, K): the 3
        # slot splits into mixture logits / means / raw scales.
        out = self.head(nn.functional.gelu(self.out_norm(h)))
        out = out.permute(0, 2, 3, 1).reshape(b, self.n_w, self.n_d, 3, self.cfg.n_components)
        log_pi = out[..., 0, :].log_softmax(dim=-1)  # (B, W, D, K) normalized log weights
        mu = out[..., 1, :]  # (B, W, D, K) component means (normalized ratio scale)
        # softplus keeps sigma > 0; the 1e-3 floor prevents a collapsing
        # component from driving the NLL to -inf.
        sigma = nn.functional.softplus(out[..., 2, :]) + 1e-3  # (B, W, D, K)
        return log_pi, mu, sigma
