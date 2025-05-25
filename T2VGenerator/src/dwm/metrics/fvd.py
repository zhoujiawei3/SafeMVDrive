import torch
import torch.distributed
import torchmetrics

# PYTHONPATH includes ${workspaceFolder}/externals/TATS/tats/fvd
import pytorch_i3d


def _compute_fid(
    mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor,
    sigma2: torch.Tensor
):
    # The same implementation as torchmetrics.image.fid._compute_fid()

    a = (mu1 - mu2).square().sum(dim=-1)
    b = sigma1.trace() + sigma2.trace()
    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)

    return a + b - 2 * c


class FrechetVideoDistance(torchmetrics.Metric):

    target_resolution = (224, 224)
    i3d_min = 10

    def __init__(
        self, inception_3d_checkpoint_path: str, sequence_count: int = -1,
        num_classes: int = 400, **kwargs
    ):
        super().__init__(**kwargs)

        self.inception = pytorch_i3d.InceptionI3d(num_classes)
        self.inception.eval()
        state_dict = torch.load(
            inception_3d_checkpoint_path, map_location="cpu",
            weights_only=True)
        self.inception.load_state_dict(state_dict)
        self.sequence_count = sequence_count

        mx_num_feats = (num_classes, num_classes)
        self.add_state(
            "real_features_sum", torch.zeros(num_classes).double(),
            dist_reduce_fx="sum")
        self.add_state(
            "real_features_cov_sum", torch.zeros(mx_num_feats).double(),
            dist_reduce_fx="sum")
        self.add_state(
            "real_features_num_samples", torch.tensor(0).long(),
            dist_reduce_fx="sum")
        self.add_state(
            "fake_features_sum", torch.zeros(num_classes).double(),
            dist_reduce_fx="sum")
        self.add_state(
            "fake_features_cov_sum", torch.zeros(mx_num_feats).double(),
            dist_reduce_fx="sum")
        self.add_state(
            "fake_features_num_samples", torch.tensor(0).long(),
            dist_reduce_fx="sum")

    def update(self, frames, real=True):
        """Update the state with extracted features.

        Args:
            frames: The video frame tensor to evaluate in the shape of
                `(batch_size, sequence_length, channels, height, width)`.
            real: Whether given frames are real or fake.
        """

        if self.sequence_count >= 0:
            frames = frames[:, :self.sequence_count]

        assert frames.shape[1] >= FrechetVideoDistance.i3d_min

        # normalize from [0, 1] to [-1, 1]
        frames = frames * 2 - 1

        frames = torch.nn.functional.interpolate(
            frames.flatten(0, 1), size=FrechetVideoDistance.target_resolution,
            mode="bilinear"
        ).unflatten(0, frames.shape[:2])
        features = self.inception(frames.transpose(1, 2))
        self.orig_dtype = features.dtype
        features = features.double()

        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += features.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += features.shape[0]

    def compute(self):
        if (
            self.real_features_num_samples < 2 or
            self.fake_features_num_samples < 2
        ):
            raise RuntimeError(
                "More than one sample is required for both the real and fake "
                "distributed to compute FVD")

        mean_real = (
            self.real_features_sum / self.real_features_num_samples
        ).unsqueeze(0)
        mean_fake = (
            self.fake_features_sum / self.fake_features_num_samples
        ).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - \
            self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - \
            self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(
            mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake
        ).to(self.orig_dtype)
