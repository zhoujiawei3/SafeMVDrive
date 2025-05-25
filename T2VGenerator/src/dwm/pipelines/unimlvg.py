import av
import contextlib
import os
import time
import random
import re

import einops
import safetensors.torch
import torch
import torch.nn.functional as F
import torch.cuda.amp
import torch.utils.tensorboard
import torchvision
import transformers
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed.fsdp.sharded_grad_scaler
import diffusers
import diffusers.image_processor
import dwm.common
import dwm.functional


class Unimlvg:

    @staticmethod
    def load_state(path: str):
        if path.endswith(".safetensors"):
            state = safetensors.torch.load_file(path, device="cpu")
        else:
            state = torch.load(path, map_location="cpu")

        return state

    def get_autocast_context(self):
        if "autocast" in self.common_config:
            return torch.autocast(**self.common_config["autocast"])
        else:
            return contextlib.nullcontext()

    @staticmethod
    def compute_density_for_timestep_sampling(
        weighting_scheme: str,
        batch_size: int,
        logit_mean: float = None,
        logit_std: float = None,
        mode_scale: float = None,
    ):
        """Compute the density for sampling the timesteps when doing SD3 training.

        Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

        SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
        """
        if weighting_scheme == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            u = torch.normal(
                mean=logit_mean, std=logit_std, size=(
                    batch_size,), device="cpu"
            )
            u = torch.nn.functional.sigmoid(u)
        elif weighting_scheme == "mode":
            u = torch.rand(size=(batch_size,), device="cpu")
            u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            u = torch.rand(size=(batch_size,), device="cpu")
        return u

    @staticmethod
    def get_sigmas(noise_scheduler, timesteps, n_dim, device, dtype):
        sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item()
                        for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    @staticmethod
    def flatten_clip_text(
        clip_text: list | str, flattened_clip_text: list, parsed_shape: list,
        level: int = 0, text_condition_mask: list | bool = None,
        do_classifier_free_guidance: bool = False
    ):
        level_count = 0
        if isinstance(clip_text, list) and len(parsed_shape) <= level:
            parsed_shape.append(0)

        if do_classifier_free_guidance:
            if isinstance(clip_text, str):
                flattened_clip_text.append("")
                level_count += 1
            else:
                for i in clip_text:
                    Unimlvg.flatten_clip_text(
                        i, flattened_clip_text, parsed_shape, level + 1,
                        text_condition_mask, do_classifier_free_guidance)
                    level_count += 1

        if level == 0 or not do_classifier_free_guidance:
            if isinstance(clip_text, str):
                if text_condition_mask is None or (
                        isinstance(text_condition_mask, bool) and
                        text_condition_mask):
                    flattened_clip_text.append(clip_text)
                else:
                    flattened_clip_text.append("")

                level_count += 1

            else:
                for i_id, i in enumerate(clip_text):
                    Unimlvg.flatten_clip_text(
                        i, flattened_clip_text, parsed_shape, level + 1,
                        None if text_condition_mask is None else (
                            text_condition_mask[i_id] if
                            isinstance(text_condition_mask, list) else
                            text_condition_mask),
                        False)
                    level_count += 1

        if isinstance(clip_text, list):
            parsed_shape[level] = level_count

    @staticmethod
    def _encode_prompt_with_t5(
        text_encoder,
        tokenizer,
        common_config,
        max_sequence_length=77,
        prompt=None,
        num_images_per_prompt=1,
        device=None,
        joint_attention_dim=None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if text_encoder is None:
            return torch.zeros(
                (batch_size, 77, joint_attention_dim),
                device=device,
                dtype=torch.float16,
            )

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = dwm.functional.memory_efficient_split_call(
            text_encoder, text_input_ids.to(device),
            lambda block, tensor: block(tensor)[0],
            common_config.get("memory_efficient_batch", -1))
        prompt_embeds = text_encoder(text_input_ids.to(device))[0]

        dtype = text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        return prompt_embeds

    @staticmethod
    def _encode_prompt_with_clip(
        text_encoder,
        tokenizer,
        prompt: str,
        device,
        num_images_per_prompt: int = 1,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_input_ids.to(device), output_hidden_states=True
        )

        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds = prompt_embeds.to(
            dtype=text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        return prompt_embeds, pooled_prompt_embeds

    @staticmethod
    def get_DiT_conditions(
        text_encoders,
        tokenizers,
        layout_encoder,
        batch,
        device,
        dtype,
        text_condition_mask: torch.Tensor = None,
        _3dbox_condition_mask: torch.Tensor = None,
        hdmap_condition_mask: torch.Tensor = None,
        explicit_view_modeling_mask: torch.Tensor = None,
        drop_temporal_mask: torch.Tensor = None,
        do_classifier_free_guidance: bool = False,
        num_images_per_prompt: int = 1,
        joint_attention_dim: int = None,
        common_config: dict = None,
    ):
        batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
        if do_classifier_free_guidance:
            batch_size *= 2

        # text prompt
        if text_encoders is not None:
            flattened_clip_text = []
            parsed_shape = []
            Unimlvg.flatten_clip_text(
                batch["clip_text"], flattened_clip_text, parsed_shape,
                text_condition_mask=text_condition_mask,
                do_classifier_free_guidance=do_classifier_free_guidance)

            clip_tokenizers = tokenizers[:2]
            clip_text_encoders = text_encoders[:2]

            clip_prompt_embeds_list = []
            clip_pooled_prompt_embeds_list = []
            for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
                prompt_embeds, pooled_prompt_embeds = Unimlvg._encode_prompt_with_clip(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    prompt=flattened_clip_text,
                    device=text_encoder.device,
                    num_images_per_prompt=num_images_per_prompt,
                )
                clip_prompt_embeds_list.append(prompt_embeds)
                clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

            clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
            pooled_prompt_embeds = torch.cat(
                clip_pooled_prompt_embeds_list, dim=-1)

            t5_prompt_embed = Unimlvg._encode_prompt_with_t5(
                text_encoders[-1],
                tokenizers[-1],
                common_config,
                prompt=flattened_clip_text,
                num_images_per_prompt=num_images_per_prompt,
                device=device if device is not None else text_encoders[-1].device,
                joint_attention_dim=joint_attention_dim,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds,
                (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
            )
            prompt_embeds = torch.cat(
                [clip_prompt_embeds, t5_prompt_embed], dim=-2)

            if len(parsed_shape) == 1:
                # all times and views share the same text prompt
                prompt_embeds = prompt_embeds.unsqueeze(1).unsqueeze(1)\
                    .repeat(1, sequence_length, view_count, 1, 1)
                pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(1).unsqueeze(1)\
                    .repeat(1, sequence_length, view_count, 1)
            else:
                # all times and views use different text prompts
                prompt_embeds = prompt_embeds.view(
                    *parsed_shape, *prompt_embeds.shape[1:])
                pooled_prompt_embeds = pooled_prompt_embeds.view(
                    *parsed_shape, *pooled_prompt_embeds.shape[1:])

        condition_on_all_frames = \
            "condition_on_all_frames" in common_config and \
            common_config["condition_on_all_frames"]
        condition_image_list = []
        if "3dbox_images" in batch:
            if condition_on_all_frames or sequence_length == 1:
                _3dbox_images = batch["3dbox_images"].to(device)
            else:
                _3dbox_images = batch["3dbox_images"][:, :1].to(device)
                zero_padding = torch.zeros((batch_size, sequence_length-1, view_count) +
                                           _3dbox_images.shape[-3:], device=device)
                _3dbox_images = torch.cat((_3dbox_images, zero_padding), dim=1)

            if _3dbox_condition_mask is not None:
                _3dbox_images[
                    _3dbox_condition_mask.logical_not().to(device)] = 0

            if do_classifier_free_guidance:
                _3dbox_images = torch.cat(
                    [torch.zeros_like(_3dbox_images), _3dbox_images])

            condition_image_list.append(_3dbox_images)

        if "hdmap_images" in batch:
            if condition_on_all_frames or sequence_length == 1:
                hdmap_images = batch["hdmap_images"].to(device)
            else:
                hdmap_images = batch["hdmap_images"][:, :1].to(device)
                zero_padding = torch.zeros((batch_size, sequence_length-1, view_count) +
                                           hdmap_images.shape[-3:], device=device)
                hdmap_images = torch.cat((hdmap_images, zero_padding), dim=1)

            if hdmap_condition_mask is not None:
                hdmap_images[
                    hdmap_condition_mask.logical_not().to(device)] = 0

            if do_classifier_free_guidance:
                hdmap_images = torch.cat(
                    [torch.zeros_like(hdmap_images), hdmap_images])

            condition_image_list.append(hdmap_images)

        if len(condition_image_list) > 0:
            condition_image_tensor = torch.cat(condition_image_list, -3)
        else:
            condition_image_tensor = None

        if "added_time_ids" in common_config:
            if common_config["added_time_ids"] == "fps_speed_steering" and\
                    "fps" in batch and "ego_speed" in batch and \
                    "ego_steering" in batch:
                added_time_ids = torch.stack([
                    batch["fps"].unsqueeze(-1).unsqueeze(-1)
                    .repeat(1, sequence_length, view_count),
                    batch["ego_speed"].unsqueeze(-1).repeat(1, 1, view_count),
                    batch["ego_steering"].unsqueeze(-1)
                    .repeat(1, 1, view_count),
                ], -1)
                if do_classifier_free_guidance:
                    uncondition_value = -1000.0 * \
                        torch.ones(*added_time_ids.shape[:3])
                    added_time_ids = torch.cat([
                        torch.stack([
                            batch["fps"].unsqueeze(-1).unsqueeze(-1)
                            .repeat(1, sequence_length, view_count),
                            uncondition_value,
                            uncondition_value
                        ], -1),
                        added_time_ids
                    ], 0)

                added_time_ids = added_time_ids.to(device)
            elif common_config["added_time_ids"] == "fps_camera_transforms":
                assert "fps" in batch and "camera_intrinsics" in batch and \
                    "camera_transforms" in batch and \
                    "camera_intrinsic_embedding_indices" in common_config and \
                    "camera_intrinsic_denom_embedding_indices" in common_config and \
                    "camera_transform_embedding_indices" in common_config
                added_time_ids = torch.cat([
                    batch["fps"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    .repeat(1, sequence_length, view_count, 1),
                    batch["camera_intrinsics"].flatten(-2, -1)[
                        ...,
                        common_config["camera_intrinsic_embedding_indices"]
                    ] / batch["image_size"][
                        ...,
                        common_config[
                            "camera_intrinsic_denom_embedding_indices"
                        ]
                    ],
                    batch["camera_transforms"].flatten(-2, -1)[
                        ...,
                        common_config["camera_transform_embedding_indices"]
                    ]
                ], -1)
                if do_classifier_free_guidance:
                    added_time_ids = torch.cat(
                        [added_time_ids, added_time_ids], 0)

                added_time_ids = added_time_ids.to(device)
            else:
                added_time_ids = None

        if common_config.get("explicit_view_modeling", False):
            assert "camera_intrinsics" in batch and \
                "camera_transforms" in batch

            # without ego_transforms or single frame
            if "ego_transforms" not in batch:
                ego_transforms = torch.eye(4).to(batch["camera_transforms"])
                ego_transforms = ego_transforms.unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(
                    batch["camera_transforms"].shape[0],
                    batch["camera_transforms"].shape[1],
                    batch["camera_transforms"].shape[2], -1, -1)
            else:
                ego_transforms = batch["ego_transforms"][
                    :, :, -batch["camera_transforms"].shape[2]:, ...]

            camera2world = ego_transforms@batch["camera_transforms"]
            camera2referego = torch.linalg.inv(
                ego_transforms[:, 0, 0, :, :].unsqueeze(1).unsqueeze(2)) @ camera2world

            camera_intrinsics_norm = batch["camera_intrinsics"].clone()
            camera_intrinsics_norm[..., 0, 0] = \
                camera_intrinsics_norm[..., 0, 0] / batch["image_size"][..., 0]
            camera_intrinsics_norm[..., 1, 1] = \
                camera_intrinsics_norm[..., 1, 1] / batch["image_size"][..., 1]
            camera_intrinsics_norm[..., 0, 2] = \
                camera_intrinsics_norm[..., 0, 2] / batch["image_size"][..., 0]
            camera_intrinsics_norm[..., 1, 2] = \
                camera_intrinsics_norm[..., 1, 2] / batch["image_size"][..., 1]

            # adapt for datasets without camera calibration
            if "is_uncalibrated" in batch:
                camera_intrinsics_norm[batch["is_uncalibrated"], :, :] = \
                    torch.eye(3).to(batch["camera_transforms"])
                camera2referego[batch["is_uncalibrated"], :, :] = \
                    torch.eye(4).to(batch["camera_transforms"])
            if explicit_view_modeling_mask is not None:
                camera_intrinsics_norm[
                    explicit_view_modeling_mask.logical_not().to(device)] = \
                    torch.eye(3).to(batch["camera_transforms"])
                camera2referego[
                    explicit_view_modeling_mask.logical_not().to(device)] = \
                    torch.eye(4).to(batch["camera_transforms"])
            
            if do_classifier_free_guidance:
                camera_intrinsics_norm = torch.cat(
                    [camera_intrinsics_norm, camera_intrinsics_norm], 0)
                camera2referego = torch.cat(
                    [camera2referego, camera2referego], 0)

            camera_intrinsics_norm = camera_intrinsics_norm.to(device)
            camera2referego = camera2referego.to(device)

        if drop_temporal_mask is not None:
            disable_temporal = drop_temporal_mask.to(device)
        else:
            disable_temporal = torch.tensor(
                [common_config["disable_temporal"]],
                device=device).repeat(batch_size) \
                if "disable_temporal" in common_config else None

        return {
            "encoder_hidden_states": prompt_embeds.to(dtype=dtype),
            "pooled_projections": pooled_prompt_embeds.to(dtype=dtype),
            "condition_image_tensor": condition_image_tensor,

            "disable_crossview": torch.tensor(
                [common_config["disable_crossview"]],
                device=device).repeat(batch_size)
            if "disable_crossview" in common_config else None,

            "disable_temporal": disable_temporal,

            "crossview_attention_mask": (
                torch.cat([batch["crossview_mask"], batch["crossview_mask"]])
                    if do_classifier_free_guidance else batch["crossview_mask"]
            ).to(device)
            if "crossview_mask" in batch else None,

            "crossview_attention_index": (
                torch.cat([batch["crossview_attention_index"],
                          batch["crossview_attention_index"]])
                    if do_classifier_free_guidance else batch["crossview_attention_index"]
            ).to(device)
            if "crossview_attention_index" in batch else None,

            "camera_intrinsics_norm": camera_intrinsics_norm
            if common_config.get("explicit_view_modeling", False) else None,

            "camera2referego": camera2referego
            if common_config.get("explicit_view_modeling", False) else None,

            "added_time_ids": added_time_ids
            if "added_time_ids" in common_config else None
        }

    @staticmethod
    def fill_svd_mask(input_cfg, latent):
        ori_values = latent.new_zeros(latent.shape)
        mask = latent.new_zeros(
            list(latent.shape[:-3]) + [1] + list(latent.shape[-2:]))

        ori_values[:, :input_cfg['num_init_frames']
                   ] = latent[:, :input_cfg['num_init_frames']]
        mask[:, :input_cfg['num_init_frames']] = 1
        sum_v = ori_values.abs().sum(list(range(1, ori_values.ndim)), keepdim=True)
        mask *= (sum_v > 0)

        return ori_values, mask

    def gen_ar_input(
        self, noise, latent, timesteps, infer,
        cxt_condition_mask=None, first_autoregressive=False
    ):
        if self.common_config['ar_input_type'] == 'sd':
            return noise, timesteps, None

        elif self.common_config['ar_input_type'] == 'svd':
            if cxt_condition_mask is not None:
                for i in range(cxt_condition_mask.shape[0]):
                    if not cxt_condition_mask[i]:
                        latent[i] = 0

            if infer and self.dynamic_cfg.get("infer_without_cxt", False):
                print("===Set cxt to zero")
                latent *= 0

            extra_latents, mask = Unimlvg.fill_svd_mask(
                self.common_config['ar_input_cfg'], latent)
            noise = torch.cat([noise, extra_latents, mask], dim=-3)
            return noise, timesteps, None

        elif self.common_config['ar_input_type'] == 'vista':
            num_max_vista_frames = 4
            probs = [2**i for i in range(num_max_vista_frames)]
            probs = [i/sum(probs) for i in probs]
            random_size = random.choices(
                list(range(num_max_vista_frames)), weights=probs, k=1)[0]

            extra_latents, mask = Unimlvg.fill_svd_mask(
                dict(num_init_frames=random_size), latent, infer)
            noise = torch.cat([noise, extra_latents, mask], dim=-3)
            return noise, timesteps, None

        elif self.common_config['ar_input_type'] == 'pred':
            b, l, v = noise.shape[0], noise.shape[1], noise.shape[2]
            ar_condition_mask = torch.zeros(
                b, l, v, dtype=torch.float32, device=self.device)
            ar_condition_mask[:, :self.common_config["visible_frame"], :] = 1

            if not infer:
                ratio_vg = self.training_config.get("video_gen_ratio", 0)
                ratio_ip = self.training_config.get("image_pred_ratio", 0)

                for i in range(b):
                    r = torch.rand(1, generator=self.generator)
                    if r < ratio_vg:
                        # Case 1: Set the entire mask to 0
                        ar_condition_mask[i] = 0
                    elif ratio_vg < r < ratio_vg+ratio_ip:
                        # Case 2: Randomly mask some of views
                        change_l_v = torch.rand(
                            l, v, generator=self.generator) < 0.5
                        ar_condition_mask[i, change_l_v] = 0

            elif self.inference_config.get("disable_reference", False) and first_autoregressive:
                ar_condition_mask[:] = 0

            timesteps = ar_condition_mask * self.common_config.get("reference_timesteps", 0) + \
                (1 - ar_condition_mask) * timesteps
            ar_condition_mask = ar_condition_mask.unsqueeze(
                -1).unsqueeze(-1).unsqueeze(-1)
            noise = ar_condition_mask * latent + \
                (1 - ar_condition_mask) * noise

            return noise, timesteps, ar_condition_mask

        else:
            raise NotImplementedError

    def training_losses(
        self, latent, batch, model_kwargs=None, cxt_condition_mask=None
    ):
        precondition_outputs = self.training_config.get(
            "precondition_outputs", True)

        batch_size, sequence_length, view_count = latent.shape[:3]

        noise = torch.randn_like(latent, dtype=latent.dtype)
        u = Unimlvg.compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=latent.shape[0],
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (
            u * self.noise_scheduler_train.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler_train.timesteps[indices].to(
            device=latent.device)

        # Add noise according to flow matching.
        sigmas = Unimlvg.get_sigmas(
            self.noise_scheduler_train,
            timesteps,
            n_dim=latent.ndim,
            dtype=latent.dtype,
            device=latent.device,
        )
        noisy_model_input_ = sigmas * noise + (1.0 - sigmas) * latent
        timesteps = timesteps.repeat(
            sequence_length*view_count).view(batch_size, sequence_length, view_count)

        if self.common_config.get('ar_input_type') is not None:
            noisy_model_input, timesteps, ar_condition_mask = self.gen_ar_input(
                noisy_model_input_,
                latent,
                timesteps,
                infer=False,
                cxt_condition_mask=cxt_condition_mask)
        else:
            noisy_model_input = noisy_model_input_

        with self.get_autocast_context():
            model_pred, _, _ = self.DiT_wrapper(
                sample=noisy_model_input,
                timestep=timesteps,
                **model_kwargs,
            )

        if precondition_outputs:
            pred = model_pred[0] * (-sigmas) + noisy_model_input_
        else:
            pred = model_pred[0]

        if precondition_outputs:
            target = latent
        else:
            target = noise - latent

        if self.common_config.get('ar_input_type') == "pred":
            pred = pred*(1-ar_condition_mask)
            target = target*(1-ar_condition_mask)

        loss_dict = {}
        loss_dict["diffusion_loss"] = torch.nn.functional.mse_loss(
            pred.float(), target.float(), reduction="mean")

        return loss_dict

    def __init__(
        self,
        output_path,
        config: dict,
        device,
        common_config: dict,
        training_config: dict,
        inference_config: dict,
        pretrained_model_name_or_path: str,
        DiT=None,
        DiT_checkpoint_path: str = None,
        layout_encoder_config: dict = None,
        checkpointing=True,
        resume_from=None,
        metrics: dict = {}
    ):
        self.ddp = torch.distributed.is_initialized()
        self.should_save = not torch.distributed.is_initialized() or \
            torch.distributed.get_rank() == 0
        self.config = config
        self.training_config = training_config
        self.inference_config = inference_config
        self.dynamic_cfg = dict()
        self.common_config = common_config
        self.device = device
        self.output_path = output_path

        self.single_frame = self.common_config.get("single_frame", True)
        self.use_action = False if self.single_frame \
            else self.common_config.get("use_action", False)

        self.generator = torch.Generator()
        if "generator_seed" in config:
            self.generator.manual_seed(config["generator_seed"])
        else:
            self.generator.seed()

        # Load the tokenizers
        tokenizer_one = transformers.CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
        )
        tokenizer_two = transformers.CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer_2",
        )
        if self.common_config.get("enable_T5", False):
            if self.should_save:
                print("enable T5")
            tokenizer_three = transformers.T5TokenizerFast.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="tokenizer_3"
            )
        else:
            tokenizer_three = None
        self.tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]

        # Load text encoders
        text_encoder_one = transformers.CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        )
        text_encoder_two = transformers.CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16,
        )
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        text_encoder_one.to(device, dtype=torch.float16)
        text_encoder_two.to(device, dtype=torch.float16)
        text_encoder_one.eval()
        text_encoder_two.eval()
        if self.common_config.get("enable_T5", False):
            text_encoder_three = transformers.T5EncoderModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="text_encoder_3",
                torch_dtype=torch.float16,
            )
            text_encoder_three.requires_grad_(False)
            text_encoder_three.to(dtype=torch.float16)
            text_encoder_three.eval()
        else:
            text_encoder_three = None

        if text_encoder_three is not None:
            text_encoder_three = FSDP(
                text_encoder_three, device_id=torch.cuda.current_device(),
                **self.common_config.get("t5_fsdp_wrapper_settings", {}))
        self.text_encoders = [text_encoder_one,
                              text_encoder_two, text_encoder_three]

        # Load vae
        self.vae = diffusers.AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
        )
        self.vae.requires_grad_(False)
        self.vae.to(device)
        self.vae.eval()

        # Load DiT
        DiT.requires_grad_(True)
        self.DiT_wrapper = self.DiT = DiT
        if self.common_config.get("distribution_framework") is None\
                or self.common_config.get("distribution_framework") == "ddp":
            self.DiT.to(device)

        if resume_from is not None:
            dit_ckpt = Unimlvg.load_state(
                os.path.join(
                    output_path, "checkpoints", "{}.pth".format(resume_from)))
            self.DiT.load_state_dict(dit_ckpt)

        elif DiT_checkpoint_path is not None:
            checkpoint = Unimlvg.load_state(DiT_checkpoint_path)
            if self.common_config.get("ar_input_type", None) in ['svd', 'vista']:
                checkpoint['pos_embed.proj.weight'] = torch.cat(
                    [checkpoint["pos_embed.proj.weight"],
                     torch.zeros_like(
                        checkpoint["pos_embed.proj.weight"], requires_grad=True),
                     torch.zeros(1536, 1, 2, 2, requires_grad=True)], dim=-3)
            self.DiT.load_state_dict(checkpoint, strict=False)

        if "freezing_pattern" in training_config:
            pattern = re.compile(training_config["freezing_pattern"])
            frozen_module_count = 0
            for name, module in self.DiT.named_modules():
                if pattern.match(name) is not None:
                    module.requires_grad_(False)
                    # module.to(device, dtype=torch.float16)
                    frozen_module_count += 1
                    if self.should_save:
                        print("{} is frozen.".format(name))

            if self.should_save:
                print("{} modules are frozen.".format(frozen_module_count))

        dit_params = list(self.DiT.parameters())
        gradient_params = filter(lambda p: p.requires_grad, dit_params)
        num_params = sum(p.numel() for p in gradient_params) / 1_000_000
        if self.should_save:
            print(f'wm {num_params} M params are trainable!')

        if checkpointing:
            self.DiT.gradient_checkpointing = True
        if self.should_save:
            print("DiT gradient checkpointing: {}"
                  .format(self.DiT.gradient_checkpointing))
            if self.DiT.condition_image_adapter is not None:
                print("DiT condition image adapter gradient checkpointing: {}"
                      .format(self.DiT.condition_image_adapter.gradient_checkpointing))
            if self.DiT.crossview_transformer_blocks is not None:
                print("DiT crossview gradient checkpointing: {}"
                      .format(self.DiT.crossview_gradient_checkpointing))
            if self.DiT.temporal_transformer_blocks is not None:
                print("DiT temporal gradient checkpointing: {}"
                      .format(self.DiT.temporal_gradient_checkpointing))

        if self.training_config.get("enable_grad_scaler", True):
            distribution_framework = self.common_config.get(
                "distribution_framework", "fsdp")
            if distribution_framework == "ddp":
                self.grad_scaler = torch.cuda.amp.GradScaler()
            elif distribution_framework == "fsdp":
                self.grad_scaler = torch.distributed.fsdp.sharded_grad_scaler\
                    .ShardedGradScaler()

        self.distribution_framework = self.common_config.get(
            "distribution_framework", "ddp")
        if self.ddp:
            if self.distribution_framework == "ddp":
                self.DiT_wrapper = torch.nn.parallel.DistributedDataParallel(
                    self.DiT, device_ids=[int(os.environ["LOCAL_RANK"])],
                    **self.common_config["ddp_wrapper_settings"])
            elif self.distribution_framework == "fsdp":
                if "fsdp_ignored_module_pattern" in common_config:
                    pattern = re.compile(
                        common_config["fsdp_ignored_module_pattern"])
                    ignored_named_modules = [
                        (name, module)
                        for name, module in self.DiT.named_modules()
                        if pattern.match(name) is not None
                    ]
                    ignored_modules = [i[1] for i in ignored_named_modules]
                    if self.should_save:
                        print(
                            "{} modules are ignored by FSDP."
                            .format(len(ignored_named_modules)))
                        print(
                            "These ignored modules are {}."
                            .format([i[0] for i in ignored_named_modules]))
                else:
                    ignored_modules = None

                self.DiT_wrapper = FSDP(
                    self.DiT, device_id=torch.cuda.current_device(),
                    ignored_modules=ignored_modules,
                    **self.common_config["ddp_wrapper_settings"])

            else:
                raise Exception(
                    "Unknown data parallel framework {}."
                    .format(self.distribution_framework))


        self.layout_encoder = None
        self.image_processor = diffusers.image_processor.VaeImageProcessor(
            vae_scale_factor=2 ** (len(self.vae.config.block_out_channels) - 1)
        )

        self.noise_scheduler_train = (
            diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
                pretrained_model_name_or_path, subfolder="scheduler"
            )
        )
        self.noise_scheduler_val = (
            diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
                pretrained_model_name_or_path, subfolder="scheduler"
            )
        )

        self.metrics = metrics
        for i in self.metrics.values():
            i.to(self.device)

        # setup training parts
        self.loss_report_list = []
        self.step_duration = 0

        if self.should_save and output_path is not None:
            self.summary = torch.utils.tensorboard.SummaryWriter(
                os.path.join(output_path, "log")
            )

        self.optimizer = dwm.common.create_instance(
            self.training_config["optimizer_name"],
            **self.training_config["optimizer_arguments"],
            params=self.DiT_wrapper.parameters(),
        )
        if resume_from is not None:
            optimizer_state_path = os.path.join(
                output_path, "optimizer", "{}.pth".format(resume_from))
            optimizer_state = torch.load(
                optimizer_state_path, map_location="cpu")
            if self.ddp and self.distribution_framework == "fsdp":
                optimizer_state = FSDP.optim_state_dict_to_load(
                    self.DiT_wrapper, self.optimizer, optimizer_state)
            self.optimizer.load_state_dict(optimizer_state)

        self.lr_scheduler = diffusers.optimization.get_scheduler(
            **self.training_config["optimization_scheduler"], optimizer=self.optimizer
        )

    def save_checkpoint(self, output_path: str, steps: int):
        if self.ddp and self.distribution_framework == "fsdp":
            sdc = torch.distributed.fsdp.FullStateDictConfig(rank0_only=True)
            osdc = torch.distributed.fsdp.FullOptimStateDictConfig(
                rank0_only=True)
            with FSDP.state_dict_type(
                self.DiT_wrapper,
                torch.distributed.fsdp.StateDictType.FULL_STATE_DICT,
                state_dict_config=sdc, optim_state_dict_config=osdc
            ):
                dit_state_dict = self.DiT_wrapper.state_dict()
                optimizer_state_dict = FSDP.optim_state_dict(
                    self.DiT_wrapper, self.optimizer)

        elif self.should_save:
            dit_state_dict = self.DiT.state_dict()
            optimizer_state_dict = self.optimizer.state_dict()

        if self.should_save:
            os.makedirs(
                os.path.join(output_path, "checkpoints"), exist_ok=True)
            torch.save(
                dit_state_dict,
                os.path.join(
                    output_path, "checkpoints", "{}.pth".format(steps)))

            os.makedirs(os.path.join(output_path, "optimizer"), exist_ok=True)
            torch.save(
                optimizer_state_dict,
                os.path.join(output_path, "optimizer", "{}.pth".format(steps)))

        if self.ddp:
            torch.distributed.barrier()

    def log(self, global_step: int, log_steps: int):
        if self.should_save:
            if len(self.loss_report_list) > 0 and \
                    isinstance(self.loss_report_list[0], dict):
                loss_values = {
                    i: sum([j[i] for j in self.loss_report_list]) /
                    len(self.loss_report_list)
                    for i in self.loss_report_list[0].keys()
                }
                loss_message = ", ".join([
                    "{}: {:.4f}".format(k, v) for k, v in loss_values.items()
                ])
                print(
                    "Step {} ({:.1f} s/step), {}".format(
                        global_step, self.step_duration / log_steps,
                        loss_message))
                for key, value in loss_values.items():
                    self.summary.add_scalar(
                        "train/{}".format(key), value, global_step)

            else:
                loss_value = sum(self.loss_report_list) / \
                    len(self.loss_report_list)
                print(
                    "Step {} ({:.1f} s/step), loss: {:.4f}".format(
                        global_step, self.step_duration / log_steps, loss_value))
                self.summary.add_scalar("train/Loss", loss_value, global_step)

        self.loss_report_list.clear()
        self.step_duration = 0

    def train_step(self, batch: dict, global_step: int):
        t0 = time.time()
        self.DiT_wrapper.train()

        batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
        image_tensor = self.image_processor.preprocess(
            batch["vae_images"].flatten(0, 2).to(
                self.device, non_blocking=True)
        )

        latent_sample_method = (
            self.training_config["latent_sample_method"]
            if "latent_sample_method" in self.training_config
            else "mode"
        )
        latents = dwm.functional.memory_efficient_split_call(
            self.vae, image_tensor,
            lambda block, tensor: (
                block.encode(tensor).latent_dist.sample() - block.config.shift_factor
            ) * block.config.scaling_factor,
            self.common_config.get("memory_efficient_batch", -1))
        latents = latents.view(
            batch_size, sequence_length, view_count, *latents.shape[1:]
        )

        # prepare conditions
        text_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("text_prompt_condition_ratio", 1.0))\
            .tolist()
        _3dbox_condition_mask = (
            (
                torch.rand((batch_size,), generator=self.generator)
                < self.training_config["3dbox_condition_ratio"]
            )
            if "3dbox_condition_ratio" in self.training_config
            else None
        )
        hdmap_condition_mask = (
            (
                torch.rand((batch_size,), generator=self.generator)
                < self.training_config["hdmap_condition_ratio"]
            )
            if "hdmap_condition_ratio" in self.training_config
            else None
        )
        if self.common_config.get("explicit_view_modeling", False):
            explicit_view_modeling_mask = (
                torch.rand((batch_size,), generator=self.generator) <
                self.training_config.get("explicit_view_modeling_ratio", 1.0))\
                .to(self.device)
        else:
            explicit_view_modeling_mask = None

        cxt_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("cxt_condition_ratio", 1.0)).to(self.device)
        drop_temporal_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("drop_temporal_ratio", 0)).to(self.device)

        DiT_conditions = Unimlvg.get_DiT_conditions(
            self.text_encoders,
            self.tokenizers,
            self.layout_encoder,
            batch,
            self.device,
            torch.float32,
            text_condition_mask,
            _3dbox_condition_mask,
            hdmap_condition_mask,
            explicit_view_modeling_mask,
            drop_temporal_mask,
            joint_attention_dim=self.DiT.config.joint_attention_dim,
            common_config=self.common_config
        )

        loss_dict = self.training_losses(
            latents,
            batch,
            model_kwargs=DiT_conditions,
            cxt_condition_mask=cxt_condition_mask
        )

        loss = torch.stack([i[1] for i in loss_dict.items()]).sum()
        loss_report = {"loss": loss.item()}
        if len(loss_dict.items()) > 1:
            for i in loss_dict.items():
                loss_report[i[0]] = i[1].item()

        self.loss_report_list.append(loss_report)
        # loss = loss_dict["loss"].sum() / image_tensor.shape[0]
        if self.training_config.get("enable_grad_scaler", True):
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        # optimize parameters
        should_optimize = (
            "gradient_accumulation_steps" not in self.training_config
        ) or (
            "gradient_accumulation_steps" in self.training_config
            and (global_step + 1) % self.training_config["gradient_accumulation_steps"]
            == 0
        )

        if should_optimize:
            if self.training_config.get("enable_grad_scaler", True) and \
                    self.training_config.get("enable_grad_clip", True):

                self.grad_scaler.unscale_(self.optimizer)
                self.DiT_wrapper.clip_grad_norm_(10)

            if self.training_config.get("enable_grad_scaler", True):
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        self.lr_scheduler.step()
        self.loss_report_list.append(loss_report)
        self.step_duration += time.time() - t0

    @torch.no_grad()
    def inference_pipeline(self, batch, output_type, latent_shape=None):
        self.DiT_wrapper.eval()
        total_batch = batch

        do_classifier_free_guidance = "guidance_scale" in self.inference_config
        if do_classifier_free_guidance:
            guidance_scale = self.inference_config["guidance_scale"]
        scale = 2 if do_classifier_free_guidance else 1

        if "vae_images" not in total_batch:
            total_batch["vae_images"] = torch.zeros(tuple(latent_shape[:3]) + (
                3,
                latent_shape[-2]
                * (2 ** (len(self.vae.config.down_block_types) - 1)),
                latent_shape[-1]
                * (2 ** (len(self.vae.config.down_block_types) - 1)),
            ))

        pred_frame = total_batch["vae_images"].shape[1]        # 16
        batch_frame = self.common_config["batch_frame"]         # 6
        visible_frame = self.common_config["visible_frame"]     # 3

        for i in range(0, pred_frame-batch_frame+1, batch_frame-visible_frame):
            batch = {}
            for key, value in total_batch.items():
                if key == "clip_text":
                    batch[key] = [
                        [value[k][j] for k in range(len(total_batch[key]))
                         for j in range(i, i+batch_frame)]]
                elif key == "vae_images" or key == "pred_vae_images":
                    if "carla_pred" in self.inference_config:
                        if i == 0:
                            total_pred_image = total_batch[
                                "pred_vae_images"].clone()
                            total_pred_image[:, visible_frame:] = 0
                        batch["vae_images"] = \
                            total_pred_image[:, i:i+batch_frame, ...]
                    else:
                        if i == 0:
                            total_pred_image = total_batch[key].clone()
                            total_pred_image[:, visible_frame:] = 0
                        batch[key] = total_pred_image[:, i:i+batch_frame, ...]
                elif isinstance(value, torch.Tensor) and value.ndim > 1 and \
                        value.shape[1] == pred_frame:
                    batch[key] = total_batch[key][:, i:i+batch_frame, ...]
                elif isinstance(value, torch.Tensor) and \
                        value.shape[0] == pred_frame:
                    batch[key] = total_batch[key][i:i+batch_frame]
                else:
                    batch[key] = total_batch[key]

            latents = torch.randn(
                tuple(batch["vae_images"].shape[:3]) + (
                    self.vae.config.latent_channels,
                    batch["vae_images"].shape[-2]
                    // (2 ** (len(self.vae.config.down_block_types) - 1)),
                    batch["vae_images"].shape[-1]
                    // (2 ** (len(self.vae.config.down_block_types) - 1)),
                ),
                generator=self.generator,
                dtype=torch.float32
            ).to(self.device)

            batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]

            self.noise_scheduler_val.set_timesteps(
                self.inference_config["inference_steps"], device=self.device
            )

            DiT_conditions = Unimlvg.get_DiT_conditions(
                self.text_encoders,
                self.tokenizers,
                self.layout_encoder,
                batch,
                self.device,
                torch.float32,
                do_classifier_free_guidance=do_classifier_free_guidance,
                joint_attention_dim=self.DiT.config.joint_attention_dim,
                common_config=self.common_config
            )

            if self.common_config.get('ar_input_type') is not None:
                image_tensor = self.image_processor.preprocess(
                    batch["vae_images"].flatten(0, 2).to(
                        self.device, non_blocking=True)
                )
                latent_sample_method = (
                    self.training_config["latent_sample_method"]
                    if "latent_sample_method" in self.training_config
                    else "mode"
                )
                image_latents = dwm.functional.memory_efficient_split_call(
                    self.vae, image_tensor,
                    lambda block, tensor: (
                        block.encode(tensor).latent_dist.sample() - block.config.shift_factor
                    ) * block.config.scaling_factor,
                    self.common_config.get("memory_efficient_batch", -1))
                image_latents = image_latents.view(
                    batch_size, sequence_length, view_count,
                    *image_latents.shape[1:]
                )
                image_latents = image_latents.repeat(2, 1, 1, 1, 1, 1) if \
                    do_classifier_free_guidance else image_latents

            for t in self.noise_scheduler_val.timesteps:
                latent_model_input = latents.repeat(2, 1, 1, 1, 1, 1) if \
                    do_classifier_free_guidance else latents
                timestep = t.repeat(
                    batch_size*scale*sequence_length*view_count).view(
                        batch_size*scale, sequence_length, view_count)

                if self.common_config.get('ar_input_type', None) is not None:
                    latent_model_input, timestep, _ = self.gen_ar_input(
                        latent_model_input,
                        image_latents,
                        timestep,
                        infer=True,
                        first_autoregressive=i == 0
                    )

                with self.get_autocast_context():
                    dit_output, _, _ = self.DiT_wrapper(
                        sample=latent_model_input,
                        timestep=timestep,
                        **DiT_conditions
                    )
                noise_pred = dit_output[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )

                latents = self.noise_scheduler_val.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

            new_images = dwm.functional.memory_efficient_split_call(
                self.vae, latents.flatten(0, 2).to(dtype=self.vae.dtype),
                lambda block, tensor: block.decode(
                    tensor / block.config.scaling_factor + block.config.shift_factor,
                    return_dict=False
                )[0],
                self.common_config.get("memory_efficient_batch", -1))
            new_images = self.image_processor.postprocess(
                new_images, output_type=output_type).view(
                    batch_size, sequence_length, view_count, *
                    new_images.shape[1:]
            )

            total_pred_image[:, i+visible_frame:i +
                             batch_frame] = new_images[:, visible_frame:].cpu()

        result = {}
        result['images'] = total_pred_image.flatten(0, 2)

        return result

    @torch.no_grad()
    def preview_pipeline(
        self, batch: dict, output_path: str, global_step: int,
        separate_save=False, save_visible_frame=False
    ):
        batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
        pipeline_output = self.inference_pipeline(batch, "pt")
        output_images = pipeline_output["images"]

        collected_names = []
        collected_images = [batch["vae_images"]]
        collected_names.append("image_GT")

        if "3dbox_images_denorm" in batch:
            collected_images.append(
                batch["3dbox_images_denorm"])
            collected_names.append("3dbox_images")

        if "hdmap_images_denorm" in batch:
            collected_images.append(
                batch["hdmap_images_denorm"])
            collected_names.append("hdmap_images")

        if "foreground_region_images_denorm" in batch:
            collected_images.append(
                batch["foreground_region_images_denorm"])
            collected_names.append("foreground_region_images")

        collected_images.append(
            output_images.cpu().view(
                batch_size, sequence_length, view_count, *
                output_images.shape[1:]
            )
        )
        collected_names.append("images_gen")

        stacked_images = torch.stack(collected_images)  
        if not save_visible_frame:
            stacked_images = stacked_images[:,:,self.common_config["visible_frame"]:,...]
        resized_images = torch.nn.functional.interpolate(
            stacked_images.flatten(0, 3),
            tuple(self.inference_config["preview_image_size"][::-1]),
        )
        resized_images = resized_images.view(
            *stacked_images.shape[:4], -1, *resized_images.shape[-2:]
        ) # [k,b,t,v,c,h,w]

        if self.ddp:
            id = str(torch.distributed.get_rank())
        else:
            id = ""
        if self.should_save:
            os.makedirs(os.path.join(output_path, "preview", id), exist_ok=True)

        if sequence_length == 1:
            # [C, B * T * S * H, V * W]
            preview_image = (
                resized_images.permute(
                    4, 1, 2, 0, 5, 3, 6).flatten(-2).flatten(1, 4)
            )
            if self.should_save:
                image_output_path = os.path.join(
                    output_path, "preview", id, "{}.png".format(global_step)
                )
                torchvision.transforms.functional.to_pil_image(preview_image).save(
                    image_output_path
                )
        else:
            # [T, C, B * S * H, V * W]
            preview_image = (
                resized_images.permute(
                    2, 4, 1, 0, 5, 3, 6).flatten(-2).flatten(2, 4)
            )
            if self.should_save:
                video_output_path = os.path.join(
                    output_path, "preview", id, "{}.mp4".format(global_step)
                )
                with av.open(video_output_path, mode="w") as container:
                    stream = container.add_stream("libx264", int(batch["fps"][0].item()))
                    stream.width = preview_image.shape[-1]
                    stream.height = preview_image.shape[-2]
                    stream.pix_fmt = "yuv420p"
                    stream.options = {"crf": "16"}
                    for i in preview_image:
                        frame = av.VideoFrame.from_image(
                            torchvision.transforms.functional.to_pil_image(i)
                        )
                        for p in stream.encode(frame):
                            container.mux(p)

                    for p in stream.encode():
                        container.mux(p)

        if separate_save:
            assert batch_size == 1 and view_count == 6
            resized_images = resized_images.squeeze(1)
            resized_images = resized_images.view(
                resized_images.shape[0], resized_images.shape[1], 2, 3, *resized_images.shape[-3:]
            ) # [k, t, 2, 3, c, h, w]
            preview_image = (
                resized_images.permute(
                    0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
            ) # [k, t, c, 2*h, 3*w]
            if self.should_save:
               for name, images in zip(collected_names, preview_image):
                   os.makedirs(os.path.join(output_path, "preview", id, name), exist_ok=True)
                   for i, image in enumerate(images):
                        image_output_path = os.path.join(
                            output_path, "preview", id, name, "{}_{}.png".format(global_step, i)
                        )
                        torchvision.transforms.functional.to_pil_image(image).save(
                            image_output_path
                        )
 
        if self.ddp:
            torch.distributed.barrier()

    @torch.no_grad()
    def evaluate_pipeline(
        self,
        global_step: int,
        dataset_length: int,
        validation_dataloader: torch.utils.data.DataLoader,
        validation_datasampler=None,
    ):
        if torch.distributed.is_initialized():
            validation_datasampler.set_epoch(0)

        world_size = torch.distributed.get_world_size() \
            if torch.distributed.is_initialized() else 1
        iteration_count = \
            self.inference_config["evaluation_item_count"] // world_size \
            if "evaluation_item_count" in self.inference_config else None

        for i, batch in enumerate(validation_dataloader):
            batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
            if (
                iteration_count is not None and
                i * batch_size >= iteration_count
            ):
                break

            pipeline_output = self.inference_pipeline(batch, "pt")
            pipeline_output["images"] = pipeline_output["images"].view(
                batch_size, sequence_length, view_count, *
                pipeline_output["images"].shape[1:])
            fake_images = pipeline_output["images"][
                :, self.common_config["visible_frame"]:, ...].to(self.device)

            real_images = batch[
                "vae_images"][:, self.common_config["visible_frame"]:, ...].to(self.device)

            if "fid" in self.metrics:
                self.metrics["fid"].update(
                    real_images.flatten(0, 2),
                    real=True)
                self.metrics["fid"].update(
                    fake_images.flatten(0, 2), real=False)

            if "fvd" in self.metrics:
                self.metrics["fvd"].update(
                    einops.rearrange(
                        real_images,
                        "b t v c h w -> (b v) t c h w"),
                    real=True)
                self.metrics["fvd"].update(
                    einops.rearrange(
                        fake_images, "b t v c h w -> (b v) t c h w"),
                    real=False)
        
        text = "Step {},".format(global_step)
        for k, metric in self.metrics.items():
            value = metric.compute()
            metric.reset()
            text += " {}: {:.3f}".format(k, value)
            if self.should_save:
                self.summary.add_scalar(
                    "evaluation/{}".format(k), value, global_step)

        if self.should_save:
            print(text)
