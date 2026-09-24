import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .ptdec import DEC
from typing import List
from .components import InterpretableTransformerEncoder
from omegaconf import DictConfig
from ..base import BaseModel
import pickle


def batch_pagerank(adj: torch.Tensor, damping: float = 0.85, iters: int = 50) -> torch.Tensor:
    """
    Batched PageRank via power iteration.
    adj: (bz, n, n) non-negative edge weights. Returns (bz, n) scores summing to 1.
    Scores are treated as input statistics, not a differentiable module.
    """
    with torch.no_grad():
        bz, n, _ = adj.shape
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype)
        adj = adj * (1 - eye)
        col_sum = adj.sum(dim=1, keepdim=True)  # (bz, 1, n)
        # column-stochastic transition matrix; dangling columns fall back to uniform
        trans = torch.where(col_sum > 0, adj / col_sum.clamp(min=1e-12),
                            torch.full_like(adj, 1.0 / n))
        p = adj.new_full((bz, n, 1), 1.0 / n)
        for _ in range(iters):
            p = (1 - damping) / n + damping * torch.bmm(trans, p)
        return p.squeeze(-1)


class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True, freeze_center=False, project_assignment=True, nHead=4, local_transformer=False, num_cls_tokens=1, num_communities=8, encoder_hidden_size=32, token_device='cuda'):
        super().__init__()
        self.num_cls_tokens = num_cls_tokens
        self.num_communities = num_communities
        self.transformer = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=nHead,
                                                           dim_feedforward=hidden_size,
                                                           batch_first=True)

        self.local_transformer = local_transformer
        if local_transformer:
            self.pooling = False
        else:
            self.pooling = pooling
        if self.pooling:
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

        if local_transformer:
            self.class_token = nn.ParameterList()
            for _ in range(num_communities):
                self.class_token.append(nn.Parameter(torch.Tensor(1,input_feature_size), requires_grad = True).to(token_device))
        self.reset_parameters(local_transformer)

        self.mlp = nn.Sequential(
            nn.Linear(8*input_feature_size, 1024),
            nn.Linear(1024, input_feature_size),
            nn.ReLU()
        )       

    def reset_parameters(self, local_transformer=False):
        if local_transformer:
            for i in range(len(self.class_token)):
                self.class_token[i] = nn.init.xavier_normal_(self.class_token[i])
        

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, 
            x: torch.tensor, cluster_num=-1, prompt_embedding=None):
        bz, node_num, dim = x.shape
        if self.local_transformer:
            class_token = self.class_token[cluster_num]
            if prompt_embedding is not None:
                class_token = class_token + prompt_embedding
            class_token = class_token.repeat(bz,1,1)
            x = torch.cat((class_token, x), dim=1)
        x = self.transformer(x)
        if self.local_transformer:
            cls_token = x[:, 0, :]
            x = x[:, 1:, :]
            return x, None, cls_token.reshape(x.shape[0], 1, -1)
        else:
            cls_token = x[:, :self.num_cls_tokens, :]
            x = x[:, self.num_cls_tokens:, :]
            if self.pooling:
                x, assignment = self.dec(x)
                return x, assignment, cls_token
            else:
                return x, None, cls_token

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


class ComBrainTF(BaseModel):

    def __init__(self, config: DictConfig):

        super().__init__()

        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz

        self.pos_encoding = config.model.pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(
                config.dataset.node_sz, config.model.pos_embed_dim), requires_grad=True)
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        self.num_MHSA = config.model.num_MHSA
        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling

        self.node_sz = config.dataset.node_sz

        # Community assignment ({original ROI index -> community id}, keys
        # grouped by community). Default reproduces the hardcoded CC200 setup:
        # 8 communities, boundaries [41, 70, 91, 110, 130, 137, 158, 200].
        with open(config.model.get('community_map', 'node_clus_map.pickle'), 'rb') as handle:
            self.node_clus_map = pickle.load(handle)
        comm_ids = list(self.node_clus_map.values())
        assert comm_ids == sorted(comm_ids), "community map keys must be grouped by community"
        self.n_communities = len(set(comm_ids))
        self.node_rearranged_len = []
        for c in sorted(set(comm_ids)):
            self.node_rearranged_len.append(
                (self.node_rearranged_len[-1] if self.node_rearranged_len else 0)
                + comm_ids.count(c))
        assert self.node_rearranged_len[-1] == self.node_sz
        self.rearranged_indices = list(self.node_clus_map.keys())
        self.pagerank_node = config.model.get('pagerank_node', False)
        self.pagerank_community = config.model.get('pagerank_community', False)
        # design-space knobs (defaults reproduce the original PageRank setup)
        self.pr_damping = float(config.model.get('pagerank_damping', 0.85))
        self.pr_topk = int(config.model.get('pagerank_topk', 0))
        self.pr_signed = config.model.get('pagerank_signed', 'abs')
        self.pr_norm = config.model.get('pagerank_norm', 'zscore')
        self.pr_where = config.model.get('pagerank_where', 'local')
        self.pr_init = config.model.get('pagerank_init', 'zero')
        self.pr_lr_mult = float(config.model.get('pagerank_lr_mult', 1.0))
        self.pr_comm_mode = config.model.get('pagerank_comm_mode', 'gate')
        self.pagerank_attn_bias = config.model.get('pagerank_attn_bias', False)
        # replace global attention with a PageRank-softmax mixing prior
        # (rank-1 attention; beta=0 == uniform_global_attn, so this nests
        # baseline-uniform and PR-weighted mixing in one parameterization)
        self.pr_global_attn = config.model.get('pr_global_attn', False)
        # global-stage variants: what sits between the local transformers and
        # the classification head (default reproduces the released model)
        self.global_stage = config.model.get('global_stage', 'transformer')
        assert self.global_stage in ('transformer', 'transformer_cls',
                                     'dec_only', 'dec_cls', 'cls_only',
                                     'pr_pool')
        # DEC knobs (cluster count comes from sizes[1])
        self.dec_hidden = int(config.model.get('dec_hidden', 32))
        self.dec_kl_weight = float(config.model.get('dec_kl_weight', 0.0))
        assert self.pr_signed in ('abs', 'pos')
        assert self.pr_norm in ('zscore', 'rank', 'logz')
        assert self.pr_where in ('local', 'global', 'both')
        assert self.pr_init in ('zero', 'small', 'one')
        assert self.pr_comm_mode in ('gate', 'affine', 'softmax')
        if self.pagerank_community:
            assert config.model.num_MHSA == 1, \
                "pagerank_community only supports num_MHSA == 1"
        if self.global_stage != 'transformer':
            assert config.model.num_MHSA == 1, \
                "non-default global_stage only supports num_MHSA == 1"

        def _init_pr_linear(lin: nn.Linear):
            if self.pr_init == 'zero':
                nn.init.zeros_(lin.weight)
            else:  # 'small'/'one': break the zero saddle ('one' is meaningless
                   # for a Linear embedding, treated as 'small')
                nn.init.normal_(lin.weight, std=1e-2)
            nn.init.zeros_(lin.bias)

        def _pr_scalar():
            v = {'zero': 0.0, 'small': 1e-2, 'one': 1.0}[self.pr_init]
            return nn.Parameter(torch.full((1,), v))

        if self.pagerank_node:
            if self.pr_where in ('local', 'both'):
                self.pr_embed = nn.Linear(1, forward_dim)
                _init_pr_linear(self.pr_embed)
            if self.pr_where in ('global', 'both'):
                self.pr_embed_global = nn.Linear(1, forward_dim)
                _init_pr_linear(self.pr_embed_global)
        if self.pagerank_community:
            if self.pr_comm_mode == 'gate':
                # zero-init gate: tokens start unweighted
                self.pr_gate = _pr_scalar()
            elif self.pr_comm_mode == 'affine':
                v = 0.0 if self.pr_init == 'zero' else 1e-2
                self.pr_comm_a = nn.Parameter(
                    torch.full((self.n_communities,), v))
            else:  # softmax with learnable inverse temperature
                self.pr_comm_beta = _pr_scalar()
        if self.pagerank_attn_bias:
            self.pr_attn_gamma = _pr_scalar()
        if self.global_stage == 'pr_pool':
            # inverse temperature of the per-community PageRank-softmax
            # pooling weights; init 0 = uniform mean pooling, 1 = PR-weighted
            self.pr_pool_beta = _pr_scalar()
        if self.pr_global_attn:
            assert self.global_stage in ('transformer', 'transformer_cls'), \
                "pr_global_attn needs a global transformer"
            assert not config.model.get('uniform_global_attn', False), \
                "pr_global_attn already includes the uniform case (beta=0)"
            # inverse temperature of the PageRank mixing prior
            self.pr_gattn_beta = _pr_scalar()

        self.local_transformer = TransPoolingEncoder(input_feature_size=forward_dim,
                                                     input_node_num=in_sizes[1],
                                                     hidden_size=1024,
                                                     output_node_num=sizes[1],
                                                     pooling=False,
                                                     orthogonal=config.model.orthogonal,
                                                     freeze_center=config.model.freeze_center,
                                                     project_assignment=config.model.project_assignment,
                                                     nHead=config.model.nhead,
                                                     local_transformer=True,
                                                     num_communities=self.n_communities,
                                                     token_device=config.model.get('token_device', 'cuda'))

        if self.global_stage in ('dec_only', 'dec_cls'):
            # DEC pooling directly on the local-transformer outputs, no global
            # transformer. Mirrors the encoder/DEC construction inside
            # TransPoolingEncoder (same shapes: N nodes x forward_dim).
            self.gs_encoder = nn.Sequential(
                nn.Linear(forward_dim * in_sizes[1], self.dec_hidden),
                nn.LeakyReLU(),
                nn.Linear(self.dec_hidden, self.dec_hidden),
                nn.LeakyReLU(),
                nn.Linear(self.dec_hidden, forward_dim * in_sizes[1]),
            )
            self.gs_dec = DEC(cluster_number=sizes[1],
                              hidden_dimension=forward_dim,
                              encoder=self.gs_encoder,
                              orthogonal=config.model.orthogonal,
                              freeze_center=config.model.freeze_center,
                              project_assignment=config.model.project_assignment)
        elif self.global_stage in ('cls_only', 'pr_pool'):
            pass  # head consumes the K community tokens directly
        elif config.model.num_MHSA == 1:
                self.attention_list.append(
                    TransPoolingEncoder(input_feature_size=forward_dim,
                                        input_node_num=in_sizes[1],
                                        hidden_size=1024,
                                        output_node_num=sizes[1],
                                        pooling=do_pooling[1],
                                        orthogonal=config.model.orthogonal,
                                        freeze_center=config.model.freeze_center,
                                        project_assignment=config.model.project_assignment,
                                        nHead=config.model.nhead,
                                        local_transformer=False,
                                        num_cls_tokens=self.n_communities if self.pagerank_community else 1,
                                        encoder_hidden_size=self.dec_hidden))
        else:
            for index, size in enumerate(sizes):
                self.attention_list.append(
                    TransPoolingEncoder(input_feature_size=forward_dim,
                                        input_node_num=in_sizes[index],
                                        hidden_size=1024,
                                        output_node_num=size,
                                        pooling=do_pooling[index],
                                        orthogonal=config.model.orthogonal,
                                        freeze_center=config.model.freeze_center,
                                        project_assignment=config.model.project_assignment,
                                        nHead=config.model.nhead,
                                        local_transformer=False,
                                        encoder_hidden_size=self.dec_hidden))

        # number of tokens the classification head consumes
        if self.global_stage in ('transformer', 'dec_only'):
            head_tokens = sizes[-1]
        elif self.global_stage == 'transformer_cls':
            head_tokens = sizes[-1] + (self.n_communities
                                       if self.pagerank_community else 1)
        elif self.global_stage == 'dec_cls':
            head_tokens = sizes[-1] + self.n_communities
        else:  # cls_only / pr_pool: one token per community
            head_tokens = self.n_communities

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * head_tokens, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

        self.assignMat = None
        if self.global_stage in ('transformer', 'transformer_cls'):
            self.mlp = nn.Sequential(
                nn.Linear(self.n_communities * forward_dim, 512),
                nn.LeakyReLU(),
                nn.Linear(512, forward_dim),
                nn.LeakyReLU()
            )

        if config.model.get('uniform_local_attn', False):
            self.local_transformer.transformer.uniform_attention = True
        if config.model.get('uniform_global_attn', False):
            for atten in self.attention_list:
                atten.transformer.uniform_attention = True

        if self.pagerank_community:
            bounds = [0] + self.node_rearranged_len
            comm_mat = torch.zeros(len(self.node_rearranged_len), self.node_sz)
            for c in range(len(self.node_rearranged_len)):
                comm_mat[c, bounds[c]:bounds[c + 1]] = 1.0
            self.register_buffer('comm_mat', comm_mat)

        # Construct text adapters only after the original backbone. Independent
        # CPU RNG streams preserve backbone initialization and training RNG state;
        # each adapter is identical in its single-prompt and combined variants.
        self.semantic_prompt = config.model.get('semantic_prompt', 'none')
        self.community_prompt_scale = float(config.model.get('community_prompt_scale', 1.0))
        self.community_prompt_center = bool(config.model.get('community_prompt_center', False))
        if not 0 <= self.community_prompt_scale < float('inf'):
            raise ValueError('community_prompt_scale must be finite and nonnegative')
        if self.semantic_prompt not in ('none', 'roi', 'community', 'both'):
            raise ValueError('semantic_prompt must be none, roi, community, or both')
        if self.semantic_prompt != 'none' and self.pos_encoding != 'none':
            raise ValueError('semantic prompts currently require pos_encoding=none')
        for kind, enabled, count, offset in (
                ('roi', self.semantic_prompt in ('roi', 'both'), self.node_sz, 1000003),
                ('community', self.semantic_prompt in ('community', 'both'), self.n_communities, 2000003)):
            if not enabled:
                continue
            path = config.model.get(f'{kind}_embedding_path')
            if not path:
                raise ValueError(f'{kind}_embedding_path is required')
            embedding = torch.load(path, map_location='cpu', weights_only=True)
            if isinstance(embedding, (list, tuple)):
                embedding = torch.stack(embedding)
            if not isinstance(embedding, torch.Tensor) or embedding.shape != (count, 2048):
                raise ValueError(f'{kind} embedding must have shape ({count}, 2048)')
            embedding = embedding.detach().float().contiguous()
            if not torch.isfinite(embedding).all():
                raise ValueError(f'{kind} embedding contains nonfinite values')
            if kind == 'roi':
                embedding = embedding[self.rearranged_indices]
            self.register_buffer(f'semantic_{kind}_embedding', embedding)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(
                    (torch.initial_seed() + offset) % (2 ** 63 - 1))
                projection = nn.Linear(2048, forward_dim)
            setattr(self, f'semantic_{kind}_projection', projection)

    def rearrange_node_feature(self, node_feature_rearranged, node_feature, rearranged_indices):
        # Rearrange according to node_clus_map which is a dictionary {0:1, 1:3, .... 199:7}
        node_feature_rearranged = node_feature[:, rearranged_indices, :]
        node_feature_rearranged = node_feature_rearranged[:, :, rearranged_indices]
        return node_feature_rearranged

    def _prep_pr_graph(self, fc_signed: torch.Tensor) -> torch.Tensor:
        """Signed rearranged FC -> non-negative PageRank graph.

        pagerank_signed: 'abs' uses |FC|, 'pos' keeps positive correlations only.
        pagerank_topk > 0 keeps each node's k strongest edges (symmetrized by
        max), which decorrelates PageRank from plain node strength on the
        otherwise near-complete FC graph.
        """
        adj = fc_signed.abs() if self.pr_signed == 'abs' else fc_signed.clamp(min=0)
        if self.pr_topk > 0:
            n = adj.shape[-1]
            eye = torch.eye(n, device=adj.device, dtype=adj.dtype)
            adj = adj * (1 - eye)
            k = min(self.pr_topk, n - 1)
            thresh = adj.topk(k, dim=-1).values[..., -1:]
            mask = (adj >= thresh).to(adj.dtype)
            mask = torch.maximum(mask, mask.transpose(-1, -2))
            adj = adj * mask
        return adj

    def _norm_pr(self, pr: torch.Tensor) -> torch.Tensor:
        """Per-subject normalization of PageRank scores (bz, n) -> (bz, n)."""
        if self.pr_norm == 'zscore':
            return (pr - pr.mean(dim=1, keepdim=True)) \
                / (pr.std(dim=1, keepdim=True) + 1e-6)
        if self.pr_norm == 'logz':
            lp = pr.clamp(min=1e-12).log()
            return (lp - lp.mean(dim=1, keepdim=True)) \
                / (lp.std(dim=1, keepdim=True) + 1e-6)
        # 'rank': per-subject rank transform to [-1, 1]
        rank = pr.argsort(dim=1).argsort(dim=1).to(pr.dtype)
        return 2 * rank / (pr.shape[1] - 1) - 1

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor,
                prior_override=None):

        bz, _, _, = node_feature.shape

        # PageRank scores from the (rearranged, mixup-aware) FC graph; must be
        # computed on the fly since node_feature is a mixup convex combination.
        pr_node = None
        pr_comm = None
        need_node_pr = (self.pagerank_node or self.pagerank_attn_bias
                        or self.global_stage == 'pr_pool'
                        or self.pr_global_attn)
        if prior_override is not None:
            # Controlled prior interventions. Scores are in community-reordered
            # ROI order, before _norm_pr, and remain detached input statistics.
            # Omitting this argument preserves the original mixup-aware path.
            pr_node, pr_comm = prior_override
            for required, value, width in (
                    (need_node_pr, pr_node, self.node_sz),
                    (self.pagerank_community, pr_comm, self.n_communities)):
                if required and (value is None or value.shape != (bz, width)):
                    raise ValueError('prior_override has incompatible shape')
                if value is not None and value.device != node_feature.device:
                    raise ValueError('prior_override must be on the input device')
            pr_node = pr_node.detach() if pr_node is not None else None
            pr_comm = pr_comm.detach() if pr_comm is not None else None
        elif need_node_pr or self.pagerank_community:
            with torch.no_grad():
                idx = self.rearranged_indices
                fc_re = node_feature[:, :, :self.node_sz]
                fc_re = self._prep_pr_graph(fc_re[:, idx][:, :, idx])
                if need_node_pr:
                    pr_node = batch_pagerank(fc_re, damping=self.pr_damping)
                if self.pagerank_community:
                    comm_adj = torch.einsum('cn,bnm,dm->bcd',
                                            self.comm_mat, fc_re, self.comm_mat)
                    pr_comm = batch_pagerank(comm_adj, damping=self.pr_damping)

        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        assignments = []
        attn_weights = []

        node_feature_rearranged = self.rearrange_node_feature(None, node_feature, self.rearranged_indices)

        pr_z = self._norm_pr(pr_node) if pr_node is not None else None
        if self.pagerank_node and self.pr_where in ('local', 'both'):
            node_feature_rearranged = node_feature_rearranged \
                + self.pr_embed(pr_z.unsqueeze(-1))

        if self.semantic_prompt in ('roi', 'both'):
            node_feature_rearranged = node_feature_rearranged + \
                self.semantic_roi_projection(self.semantic_roi_embedding).unsqueeze(0)
        community_embedding = None
        if self.semantic_prompt in ('community', 'both'):
            embedding = self.semantic_community_embedding
            if self.community_prompt_center:
                # Remove the mean across communities, retaining the original
                # frozen buffer and the projection's trainable bias.
                embedding = embedding - embedding.mean(dim=0, keepdim=True)
            community_embedding = self.semantic_community_projection(embedding)
            if self.community_prompt_scale != 1.0:
                community_embedding = community_embedding * self.community_prompt_scale

        bounds = [0] + self.node_rearranged_len
        local_class_tokens = []
        local_tf = self.local_transformer.transformer
        for c in range(self.n_communities):
            a, b = bounds[c], bounds[c + 1]
            if self.pagerank_attn_bias:
                # bias local attention logits toward high-PageRank keys
                # (column 0 is the community cls token: no bias)
                L = (b - a) + 1
                bias = pr_z.new_zeros(bz, L, L)
                bias[:, :, 1:] = (self.pr_attn_gamma * pr_z[:, a:b]).unsqueeze(1)
                nhead = local_tf.self_attn.num_heads
                local_tf.attn_bias = bias.unsqueeze(1).expand(
                    bz, nhead, L, L).reshape(bz * nhead, L, L)
            node_feature_rearranged[:, a:b, :], _, cls_c = self.local_transformer(
                node_feature_rearranged[:, a:b, :], cluster_num=c,
                prompt_embedding=None if community_embedding is None else community_embedding[c:c + 1])
            local_class_tokens.append(cls_c)
        if self.pagerank_attn_bias:
            local_tf.attn_bias = None

        node_feature = node_feature_rearranged
        if self.pagerank_node and self.pr_where in ('global', 'both'):
            node_feature = node_feature + self.pr_embed_global(pr_z.unsqueeze(-1))
        class_token = torch.cat(local_class_tokens, dim=1)
        if self.pagerank_community:
            # keep all community tokens as prompt tokens, modulated by
            # community PageRank (K*pr has mean 1; all modes start unweighted
            # under zero init)
            n_comm = class_token.shape[1]
            if self.pr_comm_mode == 'gate':
                scale = 1 + self.pr_gate * (n_comm * pr_comm - 1)
            elif self.pr_comm_mode == 'affine':
                scale = 1 + self.pr_comm_a.unsqueeze(0) * (n_comm * pr_comm - 1)
            else:  # 'softmax'
                prz_c = (pr_comm - pr_comm.mean(dim=1, keepdim=True)) \
                    / (pr_comm.std(dim=1, keepdim=True) + 1e-6)
                scale = n_comm * torch.softmax(self.pr_comm_beta * prz_c, dim=1)
            class_token = class_token * scale.unsqueeze(-1)
        elif self.global_stage in ('transformer', 'transformer_cls'):
            class_token = class_token.reshape((bz, -1))
            class_token = self.mlp(class_token)
            class_token = class_token.reshape((bz, 1, -1))

        if self.global_stage in ('transformer', 'transformer_cls'):
            node_feature = torch.cat((class_token, node_feature), dim=1)
            if self.pr_global_attn:
                # PR-softmax mixing prior over [prompt tokens, nodes]; prompt
                # tokens get score 0 (= the z-scored average), beta=0 gives
                # exactly uniform mixing
                n_prompt = class_token.shape[1]
                s = torch.cat([pr_z.new_zeros(bz, n_prompt), pr_z], dim=1)
                self.attention_list[0].transformer.attn_prior = \
                    torch.softmax(self.pr_gattn_beta * s, dim=1)
            if self.num_MHSA == 1:
                node_feature, assign_mat, cls_token = self.attention_list[0](node_feature)
                assignments.append(assign_mat)
                attn_weights.append(self.attention_list[0].get_attention_weights())
            else:
                for atten in self.attention_list:
                    node_feature, _, cls_token = atten(node_feature)
                    attn_weights.append(atten.get_attention_weights())
            if self.global_stage == 'transformer_cls':
                # keep the prompt/cls token(s) for the head instead of
                # discarding them (the released code drops them here)
                node_feature = torch.cat((node_feature, cls_token), dim=1)
        elif self.global_stage in ('dec_only', 'dec_cls'):
            node_feature, assign_mat = self.gs_dec(node_feature)
            assignments.append(assign_mat)
            if self.global_stage == 'dec_cls':
                # community cls tokens go straight to the head alongside the
                # DEC-pooled features (no transformer in between)
                node_feature = torch.cat((node_feature, class_token), dim=1)
        elif self.global_stage == 'cls_only':
            node_feature = class_token
        else:  # 'pr_pool': per-community PageRank-softmax-weighted pooling
            pooled = []
            for c in range(self.n_communities):
                a, b = bounds[c], bounds[c + 1]
                w = torch.softmax(self.pr_pool_beta * pr_z[:, a:b], dim=1)
                pooled.append((node_feature[:, a:b, :]
                               * w.unsqueeze(-1)).sum(dim=1, keepdim=True))
            node_feature = torch.cat(pooled, dim=1)

        self.assignMat = assignments[0] if assignments else None

        # optional DEC KL self-training loss (present in the released code but
        # never wired into training; dec_kl_weight=0 keeps that behavior)
        loss_pool = None
        if self.dec_kl_weight > 0 and self.assignMat is not None:
            if self.global_stage in ('transformer', 'transformer_cls'):
                kl = self.attention_list[0].loss(self.assignMat)
            else:  # dec_only / dec_cls
                kl = self.gs_dec.loss(self.assignMat)
            loss_pool = self.dec_kl_weight * kl

        node_feature = self.dim_reduction(node_feature)

        node_feature = node_feature.reshape((bz, -1))

        return self.fc(node_feature), loss_pool

    def get_assign_mat(self):
        return self.assignMat

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_local_attention_weights(self):
        return self.local_transformer.get_attention_weights()

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all
