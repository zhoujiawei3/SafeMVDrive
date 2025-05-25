import torch
import torch.nn as nn

import diffusers
from diffusers.models.embeddings import get_3d_sincos_pos_embed
from diffusers.models.attention_processor import AttentionProcessor, JointAttnProcessor2_0
from einops import rearrange

from collections.abc import Iterable
from itertools import repeat
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

approx_gelu = lambda: nn.GELU(approximate="tanh")

def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift

def get_layernorm(hidden_size: torch.Tensor, eps: float, affine: bool, use_kernel: bool):
    if use_kernel:
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(hidden_size, elementwise_affine=affine, eps=eps)
        except ImportError:
            raise RuntimeError("FusedLayerNorm not available. Please install apex.")
    else:
        return nn.LayerNorm(hidden_size, eps, elementwise_affine=affine)

def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, "grad_checkpointing", False):
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, use_reentrant=False, **kwargs)
        gc_step = module[0].grad_checkpointing_step
        return checkpoint_sequential(module, gc_step, *args, use_reentrant=False, **kwargs)
    return module(*args, **kwargs)

def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class STDiT3Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        rope=None,
        qk_norm=False,
        temporal=False,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
    ):
        super().__init__()
        self.temporal = temporal
        self.hidden_size = hidden_size
        self.enable_flash_attn = enable_flash_attn
        self.enable_sequence_parallelism = enable_sequence_parallelism

        if self.enable_sequence_parallelism and not temporal:
            assert False

        if qk_norm:
            qk_norm = "rms_norm"
        else:
            qk_norm = None
        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = diffusers.models.attention_processor.Attention(
            query_dim=hidden_size,
            added_kv_proj_dim=hidden_size,
            qk_norm=qk_norm,
            dim_head=hidden_size//num_heads,
            heads=num_heads,
            out_dim=hidden_size,
            bias=True,
            processor=diffusers.models.attention_processor.AttnProcessor2_0(),
        )
        self.cross_attn = diffusers.models.attention_processor.Attention(
            query_dim=hidden_size,
            added_kv_proj_dim=hidden_size,
            qk_norm=qk_norm,
            dim_head=hidden_size//num_heads,
            heads=num_heads,
            out_dim=hidden_size,
            bias=True,
            processor=diffusers.models.attention_processor.AttnProcessor2_0(),
        )
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "b (t hw) c -> b t hw c", t=T, hw=S)
        masked_x = rearrange(masked_x, "b (t hw) c -> b t hw c", t=T, hw=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "b t hw c -> b (t hw) c")
        return x

    def forward(
        self,
        x,
        y,
        t,
        mask=None,  # text mask
        x_mask=None,  # temporal mask
        t0=None,  # t with timestamp=0
        T=None,  # number of frames
        S=None,  # number of pixel patches
        ids_keep=None,
        t_cond=None, t_wise=None            # vista
    ):
        # prepare modulate parameters
        B, N, C = x.shape
        _, V, T = y.shape[:3]
        scale_shift_table = self.scale_shift_table[None, None] + t.reshape(B, -1, 6, x.shape[-1])
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            map(lambda x: x.squeeze(2), scale_shift_table.repeat(1, N//T, 1, 1).chunk(6, dim=2))
        if x_mask is not None:
            raise NotImplementedError

        # modulate (attention)
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        if x_mask is not None:
            raise NotImplementedError

        # attention
        if self.temporal:
            x_m = rearrange(x_m, "b (t hw) c -> (b hw) t c", t=T, hw=S)
            x_m = self.attn(x_m)
            x_m = rearrange(x_m, "(b hw) t c -> b (t hw) c", t=T, hw=S)
        else:
            x_m = rearrange(x_m, "b (t hw) c -> (b t) hw c", t=T, hw=S)
            x_m = self.attn(x_m)
            x_m = rearrange(x_m, "(b t) hw c -> b (t hw) c", t=T, hw=S)

        # modulate (attention)
        x_m_s = gate_msa * x_m
        if x_mask is not None:
            raise NotImplementedError

        # residual
        x = x + self.drop_path(x_m_s)

        # cross attention
        y = rearrange(y, "b v t s c -> (b v t) s c")
        x = rearrange(x, "bv (t hw) c -> (bv t) hw c", t=T)
        x = x + self.cross_attn(x, y)
        x = rearrange(x, "(bv t) hw c -> bv (t hw) c", t=T)

        # modulate (MLP)
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        if x_mask is not None:
            raise NotImplementedError

        # MLP
        x_m = self.mlp(x_m)

        # modulate (MLP)
        x_m_s = gate_mlp * x_m
        if x_mask is not None:
            raise NotImplementedError

        # residual
        x = x + self.drop_path(x_m_s)

        return x


class MaskPatchEmbed(nn.Module):
    """
    TODO: check effectiveness: whether this should be frozen or learnable (add new parameters?)
    """
    def __init__(
        self,
        embed_dim,
        spatial_interpolation_scale=1.0,
        temporal_interpolation_scale=1.0,
        merge_type="add"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.spatial_interpolation_scale = spatial_interpolation_scale
        self.temporal_interpolation_scale = temporal_interpolation_scale
        self.merge_type = merge_type
        assert self.merge_type in ["add", "cat"]

        if self.merge_type == "cat":
            self.proj = nn.Linear(embed_dim*2, embed_dim, bias=False)           # TODO: bias=True
        else:
            self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, ori_shape, ids_drop, mask_token):
        """Forward function."""
        # padding
        batch_size, num_frames, height, width = ori_shape
        pos_embed = get_3d_sincos_pos_embed(
            embed_dim=self.embed_dim,
            spatial_size=(width, height),
            temporal_size=num_frames,
            spatial_interpolation_scale=self.spatial_interpolation_scale,
            temporal_interpolation_scale=self.temporal_interpolation_scale
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(mask_token.device).to(mask_token.dtype)
        pos_embed = pos_embed.repeat(batch_size, 1, 1, 1)           # n,t,l,d
        pos_embed = torch.gather(pos_embed.flatten(0, 1), dim=1, index=ids_drop.unsqueeze(-1).repeat(1, 1, self.embed_dim))
        if self.merge_type == "cat":
            return self.proj(torch.cat([mask_token, pos_embed], dim=-1))
        else:
            return mask_token + self.proj(pos_embed)


class MaskController(nn.Module):
    """
    Class for MR:

    Note 1: If Rope is used in your spatial transformer blocks, 
    you should adjust sequential positional embedding.
    Note 2: No adjustment is needed for inference
    """
    def __init__(
        self,
        # base
        num_heads=24,
        attention_head_dim: int = 64,
        mlp_ratio=4.0,
        # mae
        decode_layer=5,
        interpolater_layer=1,
        mask_ratio=0.25,
        mae_mask_type='constant',
        mae_mask_probs=None,
        forward_mix_interpolater=False,
        use_origin_mask_token=None,
        mask_token_type=None,
        temporal_pos_embed_size=None,
        decoder_pos_embed_recipe=None,
        mask_embed_config=None,
    ):
        super().__init__()

        hidden_size = num_heads * attention_head_dim

        self.decode_layer = decode_layer
        self.interpolater_layer = interpolater_layer
        self.mask_ratio = mask_ratio
        self.mae_mask_type = mae_mask_type
        self.mae_mask_probs = mae_mask_probs
        self.forward_mix_interpolater = forward_mix_interpolater
        self.use_origin_mask_token = use_origin_mask_token
        self.mask_token_type = mask_token_type
        self.temporal_pos_embed_size = temporal_pos_embed_size
        self.decoder_pos_embed_recipe = decoder_pos_embed_recipe
        self.mask_embed_config = mask_embed_config

        self.sideblocks = nn.ModuleList(
            [
                STDiT3Block(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qk_norm=True,
                    enable_flash_attn=True,
                    enable_layernorm_kernel=False,
                    enable_sequence_parallelism=False,
                )
                for i in range(self.interpolater_layer)
            ]
        )           # 1 spatial block
        if self.forward_mix_interpolater:
            self.temporal_sideblocks = nn.ModuleList(
                [
                    STDiT3Block(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qk_norm=True,
                        enable_flash_attn=True,
                        enable_layernorm_kernel=False,
                        enable_sequence_parallelism=False,
                        # temporal
                        temporal=True,
                    )
                    for i in range(self.interpolater_layer)
                ]
            )           # 1 spatial block

        if not isinstance(self.use_origin_mask_token, dict):
            assert self.use_origin_mask_token in ["noise", None]
        if self.use_origin_mask_token == "noise" or (
                isinstance(self.use_origin_mask_token, dict) and
                "noise" in self.use_origin_mask_token["target"]):
            if self.use_origin_mask_token.get("proj_field", "large") == "large":
                k, s, pad = 3, 2, 1
            else:
                assert self.use_origin_mask_token.get("proj_field", "large") == "small"
                k, s, pad = 1, 1, 0
            self.noise_proj = nn.Conv3d(16, hidden_size, (1, k, k), (1, s, s), padding=(0, pad, pad))
        else:
            self.noise_proj = None

        assert self.mask_token_type in [None, "temporal"]
        if mask_ratio is not None:
            if self.mask_token_type == "temporal":
                self.mask_token = nn.Parameter(torch.zeros(1, self.temporal_pos_embed_size, hidden_size), requires_grad=True)
            else:
                self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)
            self.mask_ratio = float(mask_ratio)
        else:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=False)
            self.mask_ratio = None

        # embed
        self.embed_from_mask = True
        if self.embed_from_mask:
            self.mask_path_embed = MaskPatchEmbed(hidden_size, merge_type="add")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def is_first_decoder_layer(self, index_block, num_blocks):
        return index_block == (num_blocks - self.decode_layer)

    def mask_reconstruction(self, x, mask_metas, ori_shape, y_t, y_lens, temb=None):
        # === fetch metas ===
        x_drop = mask_metas["x_drop"]
        ids_drop = mask_metas["ids_drop"]
        ids_restore = mask_metas["ids_restore"]
        rc_mask = mask_metas["rc_mask"]
        mask_mae = mask_metas["mask"]
        B, T, H, W = ori_shape
        x_mask, t0_mlp = None, None         # TODO: change to timestep embedding
        # === fetch metas ===

        t_mlp = self.t_block(temb)

        x_from_temporal_sideblock = None
        if self.decoder_pos_embed_recipe is not None:
            x = self.forward_side_interpolater(x, ids_restore.flatten(0, 1), 0, x_drop=x_drop, 
                ids_drop=ids_drop, ori_shape=ori_shape)          # no embed
            S = x.shape[-2]
        else:
            raise NotImplementedError
        S = x.shape[-2]
        # pass to the basic blocks
        x_before = rearrange(x, "(b t) hw c -> b t hw c", t=T, hw=S)
        x = rearrange(x, "(b t) hw c -> b (t hw) c", t=T, hw=S)
        for block_id, sideblock in enumerate(self.sideblocks):
            if self.forward_mix_interpolater:
                x_cur = x if x_from_temporal_sideblock is None else x_from_temporal_sideblock
                x_from_temporal_sideblock = auto_grad_checkpoint(self.temporal_sideblocks[block_id], 
                    x_cur, y_t, t_mlp, y_lens, x_mask, t0_mlp, T, S, ids_keep=None)
            x = auto_grad_checkpoint(sideblock, x, y_t, t_mlp, y_lens, x_mask, t0_mlp, T, S, ids_keep=None)
            if self.forward_mix_interpolater:
                x = rc_mask * x + (1-rc_mask) * x_from_temporal_sideblock

        x = rearrange(x, "b (t hw) c -> b t hw c", t=T, hw=S)

        # masked shortcut
        mask_mae = mask_mae.unsqueeze(dim=-1)
        x = x*mask_mae + (1-mask_mae)*x_before
        masked_stage = False
        x = rearrange(x, "b t hw c -> (b t) hw c", t=T, hw=S)
        return x

    def random_masking(self, x, noise_of_x=None, H=None, W=None, timestep=None, condition_residuals=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [(N, V), T, (H, W), C], sequence
        """
        if noise_of_x is not None:
            noise_of_x = self.noise_proj(noise_of_x)
            noise_of_x = rearrange(noise_of_x, "b c t h w-> b t (h w) c")
        N, T, L, D = x.shape  # batch, length, dim
        # import pdb; pdb.set_trace()         # # decoder_layer -> 2, all -> 28, mask_ratio -> 0.3/0.2(SD-DiT)
        x = x.flatten(0, 1)
        len_keep = int(L * (1 - self.mask_ratio))

        rc_mask = None
        if self.mae_mask_type == "rand_t":
            noise = torch.rand(N, T, L, device=x.device).flatten(0, 1)
        elif self.mae_mask_type == "mix_constant_row_t":
            assert self.forward_mix_interpolater
            assert L == H*W and len_keep%H == 0

            align_scale = self.mae_mask_probs.get("align_scale", 1)
            H *= align_scale; W //= align_scale
            noise_r = torch.rand(N, T, H, W, device=x.device)
            ids_shuffle = torch.argsort(noise_r, dim=-1)[..., len_keep//H:]
            noise_r = torch.scatter(noise_r, -1, ids_shuffle, 1, reduce='add').flatten(-2, -1)
            noise_c = torch.rand(N, L, device=x.device).unsqueeze(1).repeat(1, T, 1)
            rc_mask = (torch.rand(N, 1, 1, device=x.device) < self.mae_mask_probs["constant"]).to(x.dtype)
            noise = (rc_mask*noise_c + (1-rc_mask)*noise_r).flatten(0, 1)
            # !! Temporal shuffle -> shuffle on noise schedule
        elif self.mae_mask_type == "row_t_rc":
            assert self.forward_mix_interpolater
            assert L == H*W and len_keep%H == 0

            align_scale = self.mae_mask_probs.get("align_scale", 1)
            H *= align_scale; W //= align_scale
            noise_r = torch.rand(N, T, H, W, device=x.device)
            ids_shuffle_r = torch.argsort(noise_r, dim=-1)[..., len_keep//H:]
            noise_r = torch.scatter(noise_r, -1, ids_shuffle_r, 1, reduce='add').flatten(-2, -1)

            noise_c = torch.rand(N, H, W, device=x.device)
            ids_shuffle_c = torch.argsort(noise_c, dim=-1)[..., len_keep//H:]
            noise_c = torch.scatter(noise_c, -1, ids_shuffle_c, 1, reduce='add').flatten(-2, -1).unsqueeze(1).repeat(1, T, 1)
            rc_mask = (torch.rand(N, 1, 1, device=x.device) < self.mae_mask_probs["constant"]).to(x.dtype)
            noise = (rc_mask*noise_c + (1-rc_mask)*noise_r).flatten(0, 1)
        else:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            noise = noise.unsqueeze(1).repeat(1, T, 1).flatten(0, 1)

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)         # NxT, L

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]            # NxT, m_L
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))         # NxT, m_L, D

        if self.use_origin_mask_token is not None:
            ids_drop = ids_shuffle[:, len_keep:]
            if self.use_origin_mask_token == "noise":
                x_drop = torch.gather(noise_of_x.flatten(0, 1), dim=1, index=ids_drop.unsqueeze(-1).repeat(1, 1, D))
            elif isinstance(self.use_origin_mask_token, dict):
                assert self.use_origin_mask_token["type"] == "trans"        # must ids_drop and noise_of_x and t
                context = dict(
                    noise = torch.gather(noise_of_x.flatten(0, 1), dim=1, index=ids_drop.unsqueeze(-1).repeat(1, 1, D)) \
                        if "noise" in self.use_origin_mask_token["target"] else None,
                    learnable = self.reshape_mask_token(x.shape[0], ids_drop.shape[1], batch_size=N)
                )
                timestep = rearrange(timestep, "b t v-> (b v t)")
                timestep = timestep[:, None, None] / 1000            # TODO: change hardcode
                x_drop = context[self.use_origin_mask_token["target"][0]]*timestep + context[self.use_origin_mask_token["target"][1]]*(1-timestep)
        else:
            ids_drop = None
            x_drop = None

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N*T, L], device=x.device, dtype=x.dtype)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)         # NxT, L

        if condition_residuals is not None:
            new_condition_residuals = []
            for ft in condition_residuals:
                _, t, v, _, h, w = ft.shape
                ft = rearrange(ft, "b t v c h w -> (b v t) (h w) c")
                ft = torch.gather(ft, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
                ft = rearrange(ft, "(b v t) (h w) c-> b t v c h w", v=v, t=t, w=w)
                new_condition_residuals.append(ft)
            condition_residuals = new_condition_residuals
        
        x_masked = rearrange(x_masked, "(b t) hw c -> b t hw c", t=T)
        mask = rearrange(mask, "(b t) s -> b t s", t=T)
        ids_restore = rearrange(ids_restore, "(b t) s -> b t s", t=T)
        ids_keep = rearrange(ids_keep, "(b t) s -> b t s", t=T)

        mask_metas = dict(
            mask=mask, ids_restore=ids_restore, ids_keep=ids_keep,
            x_drop=x_drop, ids_drop=ids_drop, rc_mask=rc_mask
        )

        return x_masked, mask_metas, condition_residuals

    def forward_side_interpolater(self, x, ids_restore, decoder_pos_embed, 
        x_drop=None, ids_drop=None, ori_shape=None):
        if x_drop is None:
            # append mask tokens to sequence
            mask_tokens = self.reshape_mask_token(x.shape[0], ids_restore.shape[1] - x.shape[1])
        else:
            assert self.use_origin_mask_token is not None
            mask_tokens = x_drop
            # NJC: ssl_dual use extra pos_embed, but as only position information is target
            # here keep this embedding shared

        if self.embed_from_mask:
            assert decoder_pos_embed == 0
            mask_tokens = self.mask_path_embed(ori_shape, ids_drop, mask_tokens)

        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        
        # add pos embed
        x = x + decoder_pos_embed

        return x

    def reshape_mask_token(self, bt, s, batch_size=None):
        assert self.mask_token_type in [None, "temporal"]
        if self.mask_token_type == "temporal":
            t = bt // batch_size
            return self.mask_token[:, :t, None].repeat(batch_size, 1, s, 1).flatten(0, 1)
        else:
            return self.mask_token.repeat(bt, s, 1)