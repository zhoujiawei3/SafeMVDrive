import torch


class ASPP(torch.nn.Module):

    def __init__(
        self, in_channels, out_channels: int = 256,
        dilations: list = [6, 12, 18], momentum: float = 0.1,
        drop: float = 0.5, device=None, dtype=None
    ):
        super().__init__()
        self.model = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels, out_channels, 1, bias=False,
                        device=device, dtype=dtype),
                    torch.nn.BatchNorm2d(
                        out_channels, momentum=momentum, device=device,
                        dtype=dtype),
                    torch.nn.ReLU(inplace=True))
            ] + [
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels, out_channels, 3, padding=i, dilation=i,
                        bias=False, device=device, dtype=dtype),
                    torch.nn.BatchNorm2d(
                        out_channels, momentum=momentum, device=device,
                        dtype=dtype),
                    torch.nn.ReLU(inplace=True))
                for i in dilations
            ] + [
                torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d((1, 1)),
                    torch.nn.Conv2d(
                        in_channels, out_channels, 1, stride=1, bias=False,
                        device=device, dtype=dtype),
                    torch.nn.BatchNorm2d(
                        out_channels, momentum=momentum, device=device,
                        dtype=dtype),
                    torch.nn.ReLU(inplace=True))
            ])
        self.project = torch.nn.Sequential(
            torch.nn.Conv2d(
                out_channels * 5, out_channels, 1, bias=False, device=device,
                dtype=dtype),
            torch.nn.BatchNorm2d(
                out_channels, momentum=momentum, device=device, dtype=dtype),
            torch.nn.ReLU(inplace=True))
        self.dropout = torch.nn.Dropout(drop)

    def forward(self, x):
        tensors = []
        for i, layer in enumerate(self.model):
            if i + 1 < len(self.model):
                tensors.append(layer(x))
            else:
                tensors.append(
                    torch.nn.functional.interpolate(
                        layer(x), size=x.shape[-2:], mode='bilinear'))

        x = torch.cat(tensors, dim=1)
        x = self.project(x)
        return self.dropout(x)


class CameraAware(torch.nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int,
        momentum: float = 0.1, gradient_checkpointing: bool = False,
        device=None, dtype=None
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        self.mlp = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, 1, bias=False, device=device,
                dtype=dtype),
            torch.nn.BatchNorm2d(
                out_channels, momentum=momentum, device=device, dtype=dtype),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                out_channels, out_channels, 1, bias=False, device=device,
                dtype=dtype),
            torch.nn.BatchNorm2d(
                out_channels, momentum=momentum, device=device, dtype=dtype),
            torch.nn.ReLU(inplace=True))
        self.excite = torch.nn.Sequential(
            torch.nn.Conv2d(
                out_channels, out_channels, 1, bias=True, device=device,
                dtype=dtype),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                out_channels, out_channels, 1, bias=True, device=device,
                dtype=dtype),
            torch.nn.Sigmoid())

    def forward(self, x, camera_features):
        if self.training and self.gradient_checkpointing:
            y = torch.utils.checkpoint.checkpoint(
                self.mlp, camera_features, use_reentrant=False)
            y = torch.utils.checkpoint.checkpoint(
                self.excite, y, use_reentrant=False)
        else:
            y = self.excite(self.mlp(camera_features))

        return x * y


class ResBlock(torch.nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, momentum: float = 0.1,
        device=None, dtype=None
    ):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, 3, padding=1, bias=False,
                device=device, dtype=dtype),
            torch.nn.BatchNorm2d(
                out_channels, momentum=momentum, device=device, dtype=dtype),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                out_channels, out_channels, 3, padding=1, bias=False,
                device=device, dtype=dtype),
            torch.nn.BatchNorm2d(
                out_channels, momentum=momentum, device=device, dtype=dtype))

    def forward(self, x):
        return torch.nn.functional.relu(x + self.model(x), inplace=True)


class DepthNet(torch.nn.Module):

    def __init__(
        self, in_channels: int, camera_parameter_channels: int,
        mid_channels: int, context_channels: int, depth_channels: int,
        momentum: float = 0.1, gradient_checkpointing: bool = False,
        bn_in_reduce_conv = False, device=None, dtype=None, upsample_scale=1
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        
        # add to sd3
        self.upsample_scale = upsample_scale 

        if bn_in_reduce_conv:
            self.reduce_conv = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels, mid_channels*upsample_scale*upsample_scale, 
                    3, padding=1, bias=False,
                    device=device, dtype=dtype),
                torch.nn.BatchNorm2d(
                    mid_channels*upsample_scale*upsample_scale, 
                    momentum=momentum, device=device, dtype=dtype),
                torch.nn.ReLU(inplace=True))
        else:
            self.reduce_conv = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels, mid_channels*upsample_scale*upsample_scale, 
                    3, padding=1, bias=False,
                    device=device, dtype=dtype),
                torch.nn.ReLU(inplace=True))

        # depth branch
        self.depth_camera_aware = CameraAware(
            camera_parameter_channels, mid_channels, momentum=momentum,
            gradient_checkpointing=gradient_checkpointing, device=device,
            dtype=dtype)
        self.depth_modules = torch.nn.ModuleList([
            ResBlock(
                mid_channels, mid_channels, momentum=momentum, device=device,
                dtype=dtype),
            ResBlock(
                mid_channels, mid_channels, momentum=momentum, device=device,
                dtype=dtype),
            ResBlock(
                mid_channels, mid_channels, momentum=momentum, device=device,
                dtype=dtype),
            ASPP(
                mid_channels, mid_channels, momentum=momentum, device=device,
                dtype=dtype),
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    mid_channels, mid_channels, 3, padding=1, bias=False,
                    device=device, dtype=dtype),
                torch.nn.BatchNorm2d(
                    mid_channels, momentum=momentum, device=device,
                    dtype=dtype),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(
                    mid_channels, depth_channels, 1, device=device,
                    dtype=dtype))
        ])

    def forward(self, x, camera_parameters):
        base_shape = x.shape[:-3]

        x = self.reduce_conv(x.flatten(0, -4))  
        if self.upsample_scale > 1:
            x = torch.nn.functional.pixel_shuffle(x, self.upsample_scale)

        depth = self.depth_camera_aware(
            x, camera_parameters.flatten(0, -4))
        for module in self.depth_modules:
            if self.training and self.gradient_checkpointing:
                depth = torch.utils.checkpoint.checkpoint(
                    module, depth, use_reentrant=False)
            else:
                depth = module(depth)

        depth = depth.view(*base_shape, *depth.shape[1:])
        
        return depth