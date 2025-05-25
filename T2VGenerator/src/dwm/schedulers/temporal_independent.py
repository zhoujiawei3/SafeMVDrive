import diffusers.schedulers
import diffusers.utils.torch_utils
import torch


class DDPMScheduler(diffusers.schedulers.DDPMScheduler):

    def add_noise(
        self, original_samples: torch.Tensor, noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ):
        while len(timesteps.shape) < len(original_samples.shape):
            timesteps = timesteps.unsqueeze(-1)

        # Make sure alphas_cumprod and timestep have same device and dtype as
        # original_samples Move the self.alphas_cumprod to device to avoid
        # redundant CPU to GPU data movement for the subsequent add_noise calls
        self.alphas_cumprod = self.alphas_cumprod\
            .to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        noisy_samples = sqrt_alpha_prod * original_samples + \
            sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_velocity(
        self, sample: torch.Tensor, noise: torch.Tensor,
        timesteps: torch.IntTensor
    ):
        while len(timesteps.shape) < len(sample.shape):
            timesteps = timesteps.unsqueeze(-1)

        # Make sure alphas_cumprod and timestep have same device and dtype as
        # sample
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity


class DDIMScheduler(diffusers.schedulers.DDIMScheduler):

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = torch.where(
            prev_timestep >= 0,
            self.alphas_cumprod[prev_timestep],
            torch.ones(
                prev_timestep.shape, dtype=self.alphas_cumprod.dtype,
                device=self.alphas_cumprod.device) * self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * \
            (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def step(
        self, model_output: torch.Tensor, timestep: torch.IntTensor,
        sample: torch.Tensor, eta: float = 0.0,
        use_clipped_model_output: bool = False, generator=None,
        variance_noise=None, return_dict: bool = True,
    ):
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run "
                "'set_timesteps' after creating the scheduler")

        while len(timestep.shape) < len(sample.shape):
            timestep = timestep.unsqueeze(-1)

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // \
            self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep].to(device=sample.device)
        alpha_prod_t_prev = torch.where(
            prev_timestep >= 0,
            self.alphas_cumprod[prev_timestep],
            torch.ones(
                prev_timestep.shape, dtype=self.alphas_cumprod.dtype,
                device=self.alphas_cumprod.device) * self.final_alpha_cumprod
        ).to(device=sample.device)

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from
        # https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = \
                (sample - beta_prod_t ** (0.5) * model_output) / \
                alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = \
                (sample - alpha_prod_t ** (0.5) * pred_original_sample) / \
                beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - \
                (beta_prod_t ** 0.5) * model_output
            pred_epsilon = (alpha_prod_t ** 0.5) * model_output + \
                (beta_prod_t ** 0.5) * sample
        else:
            raise ValueError(
                "prediction_type given as {} must be one of `epsilon`, "
                "`sample`, or `v_prediction`"
                .format(self.config.prediction_type))

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)\
            .to(sample.device)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in
            # Glide
            pred_epsilon = \
                (sample - alpha_prod_t ** (0.5) * pred_original_sample) / \
                beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from
        # https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = \
            (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from
        # https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * \
            pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please "
                    "make sure that either `generator` or `variance_noise` "
                    "stays `None`.")

            if variance_noise is None:
                variance_noise = diffusers.utils.torch_utils.randn_tensor(
                    model_output.shape, generator=generator,
                    device=model_output.device, dtype=model_output.dtype)

            variance = std_dev_t * variance_noise
            prev_sample = prev_sample + variance

        if not return_dict:
            return (prev_sample, pred_original_sample)

        return diffusers.schedulers.scheduling_ddim.DDIMSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample)


class FlowMatchEulerDiscreteScheduler(
    diffusers.schedulers.FlowMatchEulerDiscreteScheduler
):
    def step_by_indices(
        self, model_output: torch.FloatTensor, timestep_indices,
        sample: torch.FloatTensor, return_dict: bool = True
    ):
        if isinstance(timestep_indices, torch.Tensor):
            while len(timestep_indices.shape) < model_output.ndim:
                timestep_indices = timestep_indices.unsqueeze(-1)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[timestep_indices]
        sigma_next = self.sigmas[timestep_indices + 1]
        prev_sample = sample + (sigma_next - sigma) * model_output

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)
        if not return_dict:
            return (prev_sample,)

        return diffusers.schedulers.scheduling_flow_match_euler_discrete.\
            FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
