import torch
from torch.nn import TransformerEncoderLayer
from torch import Tensor
from typing import Optional
import torch.nn.functional as F


class InterpretableTransformerEncoder(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.3, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, device, dtype)
        self.attention_weights: Optional[Tensor] = None
        # When True, replace learned attention with uniform averaging:
        # every query attends 1/n to each token (out = W_o(mean_j V_j)).
        self.uniform_attention = False
        # Optional additive attention-logit bias (bz*nhead, L, L), set by the
        # caller before forward (PageRank centrality bias).
        self.attn_bias: Optional[Tensor] = None
        # Optional fixed mixing prior (bz, L): replaces attention entirely
        # with a rank-1 weighted average over values (every query uses the
        # same key weights). Uniform prior == uniform_attention.
        self.attn_prior: Optional[Tensor] = None
        print(f"nhead = {nhead}")

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                  is_causal: bool = False) -> Tensor:
        if self.uniform_attention:
            d = x.shape[-1]
            w_v = self.self_attn.in_proj_weight[2 * d:]
            b_v = (self.self_attn.in_proj_bias[2 * d:]
                   if self.self_attn.in_proj_bias is not None else None)
            v = F.linear(x, w_v, b_v)
            v = v.mean(dim=1, keepdim=True).expand_as(v)
            out = self.self_attn.out_proj(v)
            n = x.shape[1]
            self.attention_weights = torch.full(
                (x.shape[0], self.self_attn.num_heads, n, n), 1.0 / n,
                device=x.device, dtype=x.dtype)
            return self.dropout1(out)
        if self.attn_prior is not None:
            d = x.shape[-1]
            w_v = self.self_attn.in_proj_weight[2 * d:]
            b_v = (self.self_attn.in_proj_bias[2 * d:]
                   if self.self_attn.in_proj_bias is not None else None)
            v = F.linear(x, w_v, b_v)                       # (bz, L, D)
            mixed = torch.bmm(self.attn_prior.unsqueeze(1), v)  # (bz, 1, D)
            out = self.self_attn.out_proj(mixed.expand_as(v))
            n = x.shape[1]
            self.attention_weights = self.attn_prior.detach()[:, None, None, :] \
                .expand(x.shape[0], self.self_attn.num_heads, n, n)
            return self.dropout1(out)
        if attn_mask is None and self.attn_bias is not None:
            attn_mask = self.attn_bias
        x, weights = self.self_attn(x, x, x,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True,
                                    average_attn_weights=False)
        self.attention_weights = weights
        return self.dropout1(x)

    def get_attention_weights(self) -> Optional[Tensor]:
        return self.attention_weights
