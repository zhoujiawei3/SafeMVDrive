import diffusers.models.attention
import diffusers.models.embeddings
import diffusers.models.resnet
import diffusers.models.transformers.transformer_temporal
import einops
import torch


class AlphaBlender(torch.nn.Module):

    strategies = ["fixed", "learned", "learned_with_images"]

    def __init__(
        self, alpha: float, merge_strategy: str = "learned_with_images"
    ):
        super().__init__()
        self.merge_strategy = merge_strategy

        if merge_strategy not in AlphaBlender.strategies:
            raise ValueError(
                "merge_strategy needs to be in {}"
                .format(AlphaBlender.strategies))

        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif "learned" in self.merge_strategy:
            self.register_parameter(
                "mix_factor", torch.nn.Parameter(torch.Tensor([alpha])))
        else:
            raise ValueError(
                "Unknown merge strategy {}".format(self.merge_strategy))

    def get_alpha(self, image_only_indicator=None):
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor
        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor)
        elif self.merge_strategy == "learned_with_images":
            if image_only_indicator is None:
                raise ValueError(
                    "Please provide image_only_indicator to use "
                    "learned_with_images merge strategy")

            alpha = torch.where(
                image_only_indicator,
                torch.ones((1,), device=image_only_indicator.device),
                torch.sigmoid(self.mix_factor))
        else:
            raise NotImplementedError

        return alpha

    def forward(
        self, a: torch.Tensor, b: torch.Tensor, image_only_indicator=None
    ):
        """Returns alpha * a + (1 - alpha) * b. For the True item in the
        image_only_indicator, the alpha is 1.

        Args:
            a (`torch.Tensor`): The first input tensor.
            b (`torch.Tensor`): The second input tensor,
            image_only_indicator
                (`torch.BoolTensor`, *optional*, default to None):
                The flags in the shape of `(batch_size,)` to switch the alpha
                to 1 if the item is True.
        """

        alpha = self.get_alpha(image_only_indicator).to(a.dtype)
        expected_alpha_shape = list(alpha.shape) + \
            [1 for _ in range(len(a.shape) - len(alpha.shape))]
        alpha = alpha.view(*expected_alpha_shape)
        return alpha * a + (1.0 - alpha) * b


class ResBlock(torch.nn.Module):
    """A crossview temporal Resnet block.

    Args:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None,
            same as `in_channels`.
        temb_channels (`int`, *optional*, default to `512`): the number of
            channels in timestep embedding.
        eps (`float`, *optional*, defaults to `1e-5`): The epsilon to use for
            the spatial resenet.
        enable_temporal (`bool`, default to `True`): The flag to define
            temporal resnet blocks.
        temporal_eps (`float`, *optional*, defaults to `eps`): The epsilon to
            use for the temporal resnet.
        merge_factor (`float`, *optional*, defaults to `0.5`): The merge factor
            to use for the temporal mixing.
        merge_strategy (`str`, *optional*, defaults to `learned_with_images`):
            The merge strategy to use for the temporal mixing.
    """

    def __init__(
        self, in_channels: int, out_channels=None, temb_channels: int = 512,
        eps: float = 1e-5, enable_temporal: bool = True, temporal_eps=None,
        merge_factor: float = 0.5, merge_strategy: str = "learned_with_images"
    ):
        super().__init__()

        self.spatial_res_block = diffusers.models.resnet.ResnetBlock2D(
            in_channels=in_channels, out_channels=out_channels,
            temb_channels=temb_channels, eps=eps)

        if enable_temporal:
            self.temporal_res_block = diffusers.models.resnet\
                .TemporalResnetBlock(
                    out_channels if out_channels is not None else in_channels,
                    out_channels if out_channels is not None else in_channels,
                    temb_channels=temb_channels,
                    eps=temporal_eps if temporal_eps is not None else eps)
            self.time_mixer = AlphaBlender(
                merge_factor, merge_strategy=merge_strategy)
        else:
            self.temporal_res_block = None

    def forward(
        self, hidden_states: torch.Tensor, temb=None, disable_temporal=None
    ):
        """The forward method.

        Args:
            hidden_states (`torch.Tensor`): The tensor with the following shape
                `(batch_size, sequence_length, view_count, channel, height,
                width)`.
            temb (`torch.Tensor`, *optional*, default to None): The tensor with
                the following shape `(batch_size, sequence_length, view_count,
                channel)`.
            disable_temporal (`torch.BoolTensor`, *optional*, default to None):
                The flags in the shape of `(batch_size, )` to disable the
                temporal attension result if the item is True.

        Returns:
            A `tuple` is returned where the first element is the sample tensor.
        """

        batch_size = hidden_states.shape[0]

        # spatio part
        hidden_states = self\
            .spatial_res_block(
                hidden_states.flatten(0, 2),
                temb.flatten(0, 2) if temb is not None else temb)\
            .unflatten(0, tuple(hidden_states.shape[:3]))

        # temporal part
        if self.temporal_res_block is not None:
            # hidden state: [B, T, V, C, H, W] <-> [B * V, C, T, H, W]
            # temb: [B, T, V, C] -> [B * V, T, C]
            temporal_hidden_states = self\
                .temporal_res_block(
                    hidden_states.permute(0, 2, 3, 1, 4, 5).flatten(0, 1),
                    temb.transpose(1, 2).flatten(0, 1) if temb is not None
                    else temb)\
                .unflatten(0, (batch_size, -1)).permute(0, 3, 1, 2, 4, 5)

            hidden_states = self.time_mixer(
                hidden_states, temporal_hidden_states,
                image_only_indicator=disable_temporal)

        return hidden_states


class TemporalBasicTransformerBlock(torch.nn.Module):
    """The basic Transformer block for multi-frame data.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        time_mix_inner_dim (`int`): The number of channels for temporal
            attention.
        num_attention_heads (`int`): The number of heads to use for multi-head
            attention.
        attention_head_dim (`int`): The number of channels in each head.
        cross_attention_dim (`int`, *optional*): The size of the
            encoder_hidden_states vector for cross attention.
    """

    def __init__(
        self, dim: int, time_mix_inner_dim: int, num_attention_heads: int,
        attention_head_dim: int, cross_attention_dim=None
    ):
        super().__init__()
        self.is_res = dim == time_mix_inner_dim

        self.norm_in = torch.nn.LayerNorm(dim)

        # Define 3 blocks. Each block has its own normalization layer.

        # 1. Self-Attn
        self.ff_in = diffusers.models.attention.FeedForward(
            dim, dim_out=time_mix_inner_dim, activation_fn="geglu")

        self.norm1 = torch.nn.LayerNorm(time_mix_inner_dim)
        self.attn1 = diffusers.models.attention_processor.Attention(
            query_dim=time_mix_inner_dim, heads=num_attention_heads,
            dim_head=attention_head_dim, cross_attention_dim=None)

        # 2. Cross-Attn
        if cross_attention_dim is not None:
            self.norm2 = torch.nn.LayerNorm(time_mix_inner_dim)
            self.attn2 = diffusers.models.attention_processor.Attention(
                query_dim=time_mix_inner_dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads, dim_head=attention_head_dim)
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        self.norm3 = torch.nn.LayerNorm(time_mix_inner_dim)
        self.ff = diffusers.models.attention.FeedForward(
            time_mix_inner_dim, activation_fn="geglu")

    def forward(
        self, hidden_states: torch.Tensor, num_frames: int,
        encoder_hidden_states=None, self_attention_mask=None
    ):
        # Notice that normalization is always applied before the real
        # computation in the following blocks.

        # 0. Self-Attention
        batch_frames, seq_length, _ = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states = hidden_states.unflatten(0, (batch_size, -1))\
            .transpose(1, 2).flatten(0, 1)

        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)
        hidden_states = self.ff_in(hidden_states)
        if self.is_res:
            hidden_states = hidden_states + residual

        if self_attention_mask is not None:
            self_attention_mask = self_attention_mask.repeat_interleave(
                seq_length, 0)

        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states, encoder_hidden_states=None,
            attention_mask=self_attention_mask)
        hidden_states = attn_output + hidden_states

        # 3. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states)
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        if self.is_res:
            hidden_states = ff_output + hidden_states
        else:
            hidden_states = ff_output

        hidden_states = hidden_states.unflatten(0, (batch_size, -1))\
            .transpose(1, 2).flatten(0, 1)

        return hidden_states


class TransformerModel(torch.nn.Module):

    def __init__(
        self, num_attention_heads: int = 16, attention_head_dim: int = 88,
        in_channels: int = 320, out_channels=None,
        enable_crossview: bool = True, enable_temporal: bool = True,
        enable_rowwise_crossview: bool = False,
        enable_rowwise_temporal: bool = False, num_layers: int = 1,
        cross_attention_dim=None, merge_factor: float = 0.5,
        merge_strategy="learned_with_images"
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.in_channels = in_channels

        # input layers
        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6)
        self.proj_in = torch.nn.Linear(in_channels, inner_dim)

        # transformers blocks
        self.transformer_blocks = torch.nn.ModuleList([
            diffusers.models.attention.BasicTransformerBlock(
                inner_dim, num_attention_heads, attention_head_dim,
                cross_attention_dim=cross_attention_dim)
            for _ in range(num_layers)
        ])

        self.time_proj = diffusers.models.embeddings.Timesteps(
            in_channels, True, 0)
        self.enable_rowwise_crossview = enable_rowwise_crossview
        if enable_crossview:
            self.view_pos_embed = diffusers.models.embeddings\
                .TimestepEmbedding(
                    in_channels, in_channels * 4, out_dim=in_channels)
            self.crossview_transformer_blocks = torch.nn.ModuleList([
                TemporalBasicTransformerBlock(
                    inner_dim, inner_dim, num_attention_heads,
                    attention_head_dim)
                for _ in range(num_layers)
            ])
            self.view_mixer = AlphaBlender(
                merge_factor, merge_strategy=merge_strategy)
        else:
            self.view_pos_embed = None
            self.crossview_transformer_blocks = [
                None for _ in range(num_layers)]

        self.enable_rowwise_temporal = enable_rowwise_temporal
        if enable_temporal:
            self.time_pos_embed = diffusers.models.embeddings\
                .TimestepEmbedding(
                    in_channels, in_channels * 4, out_dim=in_channels)
            self.temporal_transformer_blocks = torch.nn.ModuleList([
                TemporalBasicTransformerBlock(
                    inner_dim, inner_dim, num_attention_heads,
                    attention_head_dim)
                for _ in range(num_layers)
            ])
            self.time_mixer = AlphaBlender(
                merge_factor, merge_strategy=merge_strategy)
        else:
            self.time_pos_embed = None
            self.temporal_transformer_blocks = [
                None for _ in range(num_layers)]

        # output layers
        self.proj_out = torch.nn.Linear(inner_dim, in_channels)

        self.gradient_checkpointing = False

    def forward_crossview_block_and_mix_result(
        self, crossview_block: torch.nn.Module, hidden_states: torch.Tensor,
        view_emb: torch.Tensor, crossview_attention_mask, batch_size: int,
        view_count: int, width: int, disable_crossview: torch.BoolTensor
    ):
        crossview_hidden_states = hidden_states + view_emb
        if self.enable_rowwise_crossview:
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "btv (h w) c -> (btv w) h c", w=width)
            crossview_hidden_states = crossview_block(
                crossview_hidden_states, num_frames=view_count * width,
                self_attention_mask=crossview_attention_mask)
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(btv w) h c -> btv (h w) c", w=width)
        else:
            crossview_hidden_states = crossview_block(
                crossview_hidden_states, num_frames=view_count,
                self_attention_mask=crossview_attention_mask)

        return self.view_mixer(
            hidden_states.unflatten(0, (batch_size, -1)),
            crossview_hidden_states.unflatten(0, (batch_size, -1)),
            image_only_indicator=disable_crossview).flatten(0, 1)

    def forward_temporal_block_and_mix_result(
        self, temporal_block: torch.nn.Module, hidden_states: torch.Tensor,
        sequence_emb: torch.Tensor, batch_size: int, sequence_length: int,
        width: int, disable_temporal: torch.BoolTensor
    ):
        temporal_hidden_states = hidden_states + sequence_emb
        if self.enable_rowwise_temporal:
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b t v) (h w) c -> (b v t w) h c",
                b=batch_size, t=sequence_length, w=width)
            temporal_hidden_states = temporal_block(
                temporal_hidden_states, num_frames=sequence_length * width)
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b v t w) h c -> (b t v) (h w) c",
                b=batch_size, t=sequence_length, w=width)
        else:
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b t v) hw c -> (b v t) hw c",
                b=batch_size, t=sequence_length)
            temporal_hidden_states = temporal_block(
                temporal_hidden_states, num_frames=sequence_length)
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b v t) hw c -> (b t v) hw c",
                b=batch_size, t=sequence_length)

        return self.time_mixer(
            hidden_states.unflatten(0, (batch_size, -1)),
            temporal_hidden_states.unflatten(0, (batch_size, -1)),
            image_only_indicator=disable_temporal).flatten(0, 1)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states=None,
        disable_crossview=None, disable_temporal=None,
        crossview_attention_mask=None, return_dict: bool = True
    ):
        """The forward method.

        Args:
            hidden_states (`torch.Tensor`): The tensor with the following shape
                `(batch_size, sequence_length, view_count, channel, height,
                width)`.
            encoder_hidden_states (`torch.Tensor`): The tensor with the
                following shape `(batch_size, sequence_length, view_count,
                token_count, channel)`.
            disable_crossview (`torch.BoolTensor`, *optional*, default to None):
                The flags in the shape of `(batch_size,)` to disable the
                cross-view attension result if the item is True.
            disable_temporal (`torch.BoolTensor`, *optional*, default to None):
                The flags in the shape of `(batch_size,)` to disable the
                temporal attension result if the item is True.

        Returns:
            A `tuple` is returned where the first element is the sample tensor.
        """

        # 1. Input
        batch_size, sequence_length, view_count, _, height, width = \
            hidden_states.shape

        residual = hidden_states

        spatio_context = encoder_hidden_states.flatten(0, 2) \
            if encoder_hidden_states is not None else None
        hidden_states = self.norm(hidden_states.flatten(0, 2))

        hidden_states = self.proj_in(
            hidden_states.flatten(2).transpose(-2, -1))

        if self.view_pos_embed is not None:
            view_emb = torch\
                .arange(view_count, device=hidden_states.device)\
                .unsqueeze(0).unsqueeze(0)\
                .repeat(batch_size, sequence_length, 1)
            view_emb = self.time_proj(view_emb.flatten())\
                .to(dtype=hidden_states.dtype)
            view_emb = self.view_pos_embed(view_emb).unsqueeze(1)

        if self.time_pos_embed is not None:
            sequence_emb = torch\
                .arange(sequence_length, device=hidden_states.device)\
                .unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, view_count)
            sequence_emb = self.time_proj(sequence_emb.flatten())\
                .to(dtype=hidden_states.dtype)
            sequence_emb = self.time_pos_embed(sequence_emb).unsqueeze(1)

        if self.enable_rowwise_crossview and \
                crossview_attention_mask is not None:
            crossview_attention_mask = crossview_attention_mask\
                .repeat_interleave(width, 2)\
                .repeat_interleave(width, 1)\
                .repeat_interleave(sequence_length, 0)

        # 2. Blocks
        zipped_blocks = zip(
            self.transformer_blocks, self.crossview_transformer_blocks,
            self.temporal_transformer_blocks)
        for block, crossview_block, temporal_block in zipped_blocks:
            # spatio attention
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block, hidden_states, encoder_hidden_states=spatio_context,
                    use_reentrant=False)
            else:
                hidden_states = block(
                    hidden_states, encoder_hidden_states=spatio_context)

            # cross-view attention
            if crossview_block is not None:
                if self.training and self.gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        self.forward_crossview_block_and_mix_result,
                        crossview_block, hidden_states, view_emb,
                        crossview_attention_mask, batch_size, view_count,
                        width, disable_crossview, use_reentrant=False)
                else:
                    hidden_states =\
                        self.forward_crossview_block_and_mix_result(
                            crossview_block, hidden_states, view_emb,
                            crossview_attention_mask, batch_size, view_count,
                            width, disable_crossview)

            # temporal attention
            if temporal_block is not None:
                if self.training and self.gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        self.forward_temporal_block_and_mix_result,
                        temporal_block, hidden_states, sequence_emb,
                        batch_size, sequence_length, width, disable_temporal,
                        use_reentrant=False)
                else:
                    hidden_states = self.forward_temporal_block_and_mix_result(
                        temporal_block, hidden_states, sequence_emb,
                        batch_size, sequence_length, width, disable_temporal)

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)\
            .view(batch_size, sequence_length, view_count, -1, height, width)\
            .contiguous()

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return diffusers.models.transformers.transformer_temporal\
            .TransformerTemporalModelOutput(sample=output)


class Mixer(torch.nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.randn(1, channel) / channel**0.5)

    def forward(
            self, a: torch.Tensor, b: torch.Tensor,
            image_only_indicator: bool = False):    

        alpha = torch.where(
            image_only_indicator,
            torch.zeros((1,), device=image_only_indicator.device),
            torch.ones((1,), device=image_only_indicator.device)
        ).to(b).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return a + alpha * self.scale * b


class VTSelfAttentionBlock(torch.nn.Module):
    def __init__(
        self, dim: int, time_mix_inner_dim: int, 
        num_attention_heads: int, attention_head_dim: int, 
        qk_norm=None
    ):
        super().__init__()
        self.is_res = dim == time_mix_inner_dim

        self.norm_in = torch.nn.LayerNorm(dim)

        # 1. Self-Attn
        self.ff_in = diffusers.models.attention.FeedForward(
            dim, dim_out=time_mix_inner_dim, activation_fn="geglu")

        self.norm1 = torch.nn.LayerNorm(time_mix_inner_dim)
        self.attn1 = diffusers.models.attention_processor.Attention(
            query_dim=time_mix_inner_dim, heads=num_attention_heads,
            dim_head=attention_head_dim, cross_attention_dim=None,
            qk_norm=qk_norm)

        # 2. Feed-forward
        self.norm3 = torch.nn.LayerNorm(time_mix_inner_dim)
        self.ff = diffusers.models.attention.FeedForward(
            time_mix_inner_dim, activation_fn="geglu")

    def forward(
        self, hidden_states: torch.Tensor,
        self_attention_mask: torch.Tensor = None
    ):
        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)
        hidden_states = self.ff_in(hidden_states)
        hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states, encoder_hidden_states=None,
            attention_mask=self_attention_mask)
        hidden_states = attn_output + hidden_states

        # Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states
