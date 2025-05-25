import diffusers.models.adapter
import torch
from typing import Optional


class ImageAdapter(torch.nn.Module):
    def __init__(
        self, in_channels: int = 3,
        channels: list = [320, 320, 640, 1280, 1280],
        is_downblocks: list = [False, True, True, True, False],
        num_res_blocks: int = 2, downscale_factor: int = 8,
        use_zero_convs: bool = False, zero_gate_coef: Optional[float] = None,
        gradient_checkpointing: bool = True
    ):
        super().__init__()

        in_channels = in_channels * downscale_factor ** 2
        self.unshuffle = torch.nn.PixelUnshuffle(downscale_factor)
        self.body = torch.nn.ModuleList([
            diffusers.models.adapter.AdapterBlock(
                in_channels if i == 0 else channels[i - 1], channels[i],
                num_res_blocks, down=is_downblocks[i])
            for i in range(len(channels))
        ])
        self.gradient_checkpointing = gradient_checkpointing

        self.zero_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(channel, channel, 1)
            for channel in channels
        ]) if use_zero_convs else [None for _ in channels]
        for i in self.zero_convs:
            if i is not None:
                torch.nn.init.zeros_(i.weight)
                torch.nn.init.zeros_(i.bias)

        self.zero_gate_coef = zero_gate_coef
        self.zero_gates = torch.nn.Parameter(torch.zeros(len(channels))) \
            if zero_gate_coef else None

    def forward(self, x: torch.Tensor, return_features: bool = False):
        base_shape = x.shape[:-3]
        x = self.unshuffle(x.flatten(0, -4))
        features = []
        for i, (block, zero_conv) in enumerate(zip(self.body, self.zero_convs)):
            if self.training and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False)
            else:
                x = block(x)

            x_out = x
            if zero_conv is not None:
                x_out = zero_conv(x_out)

            if self.zero_gates is not None:
                x_out = x_out * torch.tanh(
                    self.zero_gate_coef * self.zero_gates[i])

            features.append(x_out.view(*base_shape, *x_out.shape[1:]))
        return features if not return_features else features[-1]
