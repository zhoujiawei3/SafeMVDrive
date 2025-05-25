import torch


def create_frustum(
    frustum_depth_range: list, frustum_height: int, frustum_width: int,
    device=None
):
    """Create the lifted frustum from the camera view.

    Args:
        frustum_depth_range (list): 3 float numbers (start, stop, step) of the
            depth range. The stop is exclusive as same as the common definition
            of range functions.
        frustum_height: The frustum height.
        frustum_width: The frustum width.
        device: The prefered device of the returned frustum tensor.

    Returns:
        The lifted frustum tensor in the shape of [4, D, frustum_height,
        frustum_width]. The D = (stop - start) / step of the
        frustum_depth_range. The 4 items in the first dimension are the (nx * z,
        ny * z, z, 1), where nx and ny are the normalized coordinate in the
        range of -1 ~ 1, z is in the unit of camera coordinate system.
    """

    depth_channels = int(
        (frustum_depth_range[1] - frustum_depth_range[0]) /
        frustum_depth_range[2])
    x = torch.arange(
        1 / frustum_width - 1, 1, 2 / frustum_width, device=device)
    y = torch.arange(
        1 / frustum_height - 1, 1, 2 / frustum_height, device=device)
    z = torch.arange(*frustum_depth_range, device=device)
    frustum = torch.stack([
        x.unsqueeze(0).unsqueeze(0).repeat(depth_channels, frustum_height, 1),
        y.unsqueeze(0).unsqueeze(-1).repeat(depth_channels, 1, frustum_width),
        z.unsqueeze(-1).unsqueeze(-1).repeat(1, frustum_height, frustum_width),
        torch.ones(
            (depth_channels, frustum_height, frustum_width), device=device)
    ])
    frustum[:2] *= frustum[2:3]
    return frustum


def make_homogeneous_matrix(a: torch.Tensor):
    right = torch.cat([
        torch.zeros(
            tuple(a.shape[:-1]) + (1,), dtype=a.dtype, device=a.device),
        torch.ones(
            tuple(a.shape[:-2]) + (1, 1), dtype=a.dtype, device=a.device)
    ], -2)
    return torch.cat([torch.nn.functional.pad(a, (0, 0, 0, 1)), right], -1)


def make_homogeneous_vector(a: torch.Tensor):
    return torch.cat([
        a,
        torch.ones(tuple(a.shape[:-1]) + (1,), dtype=a.dtype, device=a.device)
    ], -1)


def make_transform_to_frustum(
    image_size: torch.Tensor, frustum_width: int, frustum_height: int
):
    """Make the 4x4 transform from the image coordinates to the frustum
    coordinates.

    Args:
        image_size (torch.Tensor): The image size tensor in the shape of
            [..., 2], and the 2 numbers of the last dimension is
            (width, height).
        frustum_width (int): The width of the frustum.
        frustum_height (int): The height of the frustum.

    Returns:
        The transform matrix in the shape of [..., 4, 4].
    """

    base_shape = list(image_size.shape[:-1])
    frustum_size = torch.tensor(
        [frustum_width, frustum_height], dtype=image_size.dtype,
        device=image_size.device).view(*([1 for _ in base_shape] + [-1]))
    scale = frustum_size / image_size
    zeros = torch.zeros(
        base_shape + [4], dtype=scale.dtype, device=scale.device)
    ones = torch.ones(
        base_shape + [1], dtype=scale.dtype, device=scale.device)
    return torch.cat([
        scale[..., 0:1], zeros, scale[..., 1:2], zeros, ones, zeros, ones
    ], -1).unflatten(-1, (4, 4))


def normalize_intrinsic_transform(
    image_sizes: torch.Tensor, instrinsics: torch.Tensor
):
    """Make the normalized 3x3 intrinsic transform from the camera coordinates
    to the normalized coordinates (-1 ~ 1).

    Args:
        image_sizes (torch.Tensor): The image size tensor in the shape of
            [..., 2], and the 2 numbers of the last dimension is
            (width, height).
        instrinsics (torch.Tensor): The camera intrinsic transform from the
            camera coordinates (X-right, Y-down, Z-forward) to the image
            coordinates (X-right, Y-down).

    Returns:
        The transform matrix in the shape of [..., 3, 3].
    """
    base_shape = list(image_sizes.shape[:-1])
    scale = 2 / image_sizes
    translate = 1 / image_sizes - 1
    zeros = torch.zeros(
        base_shape + [2], dtype=scale.dtype, device=scale.device)
    ones = torch.ones(
        base_shape + [1], dtype=scale.dtype, device=scale.device)
    normalization_transform = torch.cat([
        scale[..., 0:1], zeros[..., :1], translate[..., 0:1], zeros[..., :1],
        scale[..., 1:2], translate[..., 1:2], zeros, ones
    ], -1).unflatten(-1, (3, 3))
    return normalization_transform @ instrinsics


def grid_sample_sequence(
    input: torch.Tensor, sequence: torch.Tensor, bundle_size: int = 128,
    mode: str = "bilinear", padding_mode: str = "border"
):
    count = sequence.shape[-2]
    bundle_count = (count + bundle_size - 1) // bundle_size
    assert bundle_count * bundle_size == count

    samples = torch.nn.functional.grid_sample(
        input.flatten(0, -4),
        sequence.unflatten(-2, (bundle_count, bundle_size)), mode=mode,
        padding_mode=padding_mode, align_corners=False)
    return samples.unflatten(0, input.shape[:-3]).flatten(-2)


def _sample_logistic(
    shape: torch.Size, out: torch.Tensor = None,
    generator: torch.Generator = None
):
    x = torch.rand(shape, generator=generator) if out is None else \
        out.resize_(shape).uniform_(generator=generator)
    return torch.log(x) - torch.log(1 - x)


def _sigmoid_sample(
    logits: torch.Tensor, tau: float = 1.0, generator: torch.Generator = None
):
    # Refer to Bernouilli reparametrization based on Maddison et al. 2017
    noise = _sample_logistic(logits.size(), None, generator)
    y = logits + noise
    return torch.sigmoid(y / tau)


def gumbel_sigmoid(
    logits, tau: float = 1.0, hard: bool = False,
    generator: torch.Generator = None
):
    # use CPU random generator
    y_soft = _sigmoid_sample(logits.cpu(), tau, generator).to(logits.device)
    if hard:
        y_hard = torch.where(
            y_soft > 0.5, torch.ones_like(y_soft), torch.zeros_like(y_soft))
        return y_hard.data - y_soft.data + y_soft

    else:
        return y_soft


def take_sequence_clip(item, start: int, stop: int):
    # print(item)
    if isinstance(item, (int, float, bool, str)):
        return item
    elif isinstance(item, torch.Tensor):
        return item if len(item.shape) <= 1 else item[:, start:stop]
    elif isinstance(item, list):
        assert len(item) > 0 and all([isinstance(i, list) for i in item])
        return [i[start:stop] for i in item]
    elif isinstance(item, dict):
        return item
    else:
        raise Exception("Unsupported type to take sequence clip.")


def memory_efficient_split_call(
    block: torch.nn.Module, tensor: torch.Tensor, func, split_size: int
):
    if split_size == -1:
        return func(block, tensor)
    else:
        return torch.cat([
            func(block, i)
            for i in tensor.split(split_size)
        ])
