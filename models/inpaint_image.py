from typing import Union, Optional, List, Dict, Any, Callable

import PIL
import torch
from diffusers import PaintByExamplePipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_pix2pix_zero import Pix2PixInversionPipelineOutput, \
    prepare_unet
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor


class InpaintPipeline(PaintByExamplePipeline):
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline
    # .enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def _encode_image(self, image, device, num_images_per_prompt, do_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(images=image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings, negative_prompt_embeds = self.image_encoder(image, return_uncond_vector=True)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, image_embeddings.shape[0], 1)
            negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_images_per_prompt, 1, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

        return image_embeddings

    # def _encode_image(self, image, device, num_images_per_prompt, do_classifier_free_guidance):
    #     dtype = next(self.image_encoder.parameters()).dtype
    #
    #     if not isinstance(image, torch.Tensor):
    #         image = self.feature_extractor(images=image, return_tensors="pt").pixel_values
    #
    #     image = image.to(device=device, dtype=dtype)
    #     image_embeddings = self.image_encoder(image, )
    #     image_embeddings = image_embeddings.unsqueeze(1)
    #
    #     # duplicate image embeddings for each generation per prompt, using mps friendly method
    #     bs_embed, seq_len, _ = image_embeddings.shape
    #     image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
    #     image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
    #
    #     if do_classifier_free_guidance:
    #         negative_prompt_embeds = torch.zeros_like(image_embeddings)
    #
    #         # For classifier free guidance, we need to do two forward passes.
    #         # Here we concatenate the unconditional and text embeddings into a single batch
    #         # to avoid doing two forward passes
    #         image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])
    #
    #     return image_embeddings

    def encode_image(self, images, device=None):
        device, num_images_per_prompt = self._execution_device if device is None \
            else device, len(images) if isinstance(images, List) else 1
        return self._encode_image(images, device, num_images_per_prompt, False)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img
    # .StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]

        return timesteps, num_inference_steps - t_start

    def get_inverse_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)

        # safety for t_start overflow to prevent empty timsteps slice
        if t_start == 0:
            return self.inverse_scheduler.timesteps, num_inference_steps
        timesteps = self.inverse_scheduler.timesteps[:-t_start]

        return timesteps, num_inference_steps - t_start

    def prepare_image_latents(self, image, batch_size, dtype, device, generator):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == 4:
            latents = image
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
            latents = self._encode_vae_image(image, generator)

        if batch_size != latents.shape[0]:
            if batch_size % latents.shape[0] == 0:
                # expand image_latents for batch_size
                deprecation_message = (
                    f"You have passed {batch_size} text prompts (`prompt`), but only {latents.shape[0]} initial"
                    " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                    " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to "
                    "update your script to pass as many initial images as text prompts to suppress this warning."
                )
                deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
                additional_latents_per_image = batch_size // latents.shape[0]
                latents = torch.cat([latents] * additional_latents_per_image, dim=0)
            else:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {latents.shape[0]} to {batch_size} text prompts."
                )
        else:
            latents = torch.cat([latents], dim=0)

        return latents

    def prepare_embeds(self,
                       prompt_embeds: Optional[torch.FloatTensor],
                       device,
                       num_images_per_prompt):

        if self.image_encoder is not None:
            prompt_embeds_dtype = self.image_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    @torch.no_grad()
    def generate_mask(
            self,
            image: Union[torch.FloatTensor, PIL.Image.Image] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            target_prompt_embeds: Optional[torch.FloatTensor] = None,
            source_prompt_embeds: Optional[torch.FloatTensor] = None,
            num_maps_per_mask: Optional[int] = 10,
            mask_encode_strength: Optional[float] = 0.5,
            mask_thresholding_ratio: Optional[float] = 3.0,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if (num_maps_per_mask is None) or (
                num_maps_per_mask is not None and (not isinstance(num_maps_per_mask, int) or num_maps_per_mask <= 0)
        ):
            raise ValueError(
                f"`num_maps_per_mask` has to be a positive integer but is {num_maps_per_mask} of type"
                f" {type(num_maps_per_mask)}."
            )

        if mask_thresholding_ratio is None or mask_thresholding_ratio <= 0:
            raise ValueError(
                f"`mask_thresholding_ratio` has to be positive but is {mask_thresholding_ratio} of type"
                f" {type(mask_thresholding_ratio)}."
            )

        # 2. Define call parameters
        # if target_prompt is not None and isinstance(target_prompt, str):
        #     batch_size = 1
        # elif target_prompt is not None and isinstance(target_prompt, list):
        #     batch_size = len(target_prompt)
        # else:
        batch_size = target_prompt_embeds.shape[0]
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompts
        (cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None)
        target_prompt_embeds = self.prepare_embeds(
            target_prompt_embeds,
            device,
            num_maps_per_mask,
        )

        source_prompt_embeds = self.prepare_embeds(
            source_prompt_embeds,
            device,
            num_maps_per_mask,
        )

        # 4. Preprocess image
        image = self.image_processor.preprocess(image).repeat_interleave(num_maps_per_mask, dim=0)

        # 5. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, _ = self.get_timesteps(num_inference_steps, mask_encode_strength, device)
        encode_timestep = timesteps[0]

        # 6. Prepare image latents and add noise with specified strength
        image_latents = self.prepare_image_latents(
            image, batch_size * num_maps_per_mask, self.vae.dtype, device, generator
        )
        noise = randn_tensor(image_latents.shape, generator=generator, device=device, dtype=self.vae.dtype)
        image_latents = self.scheduler.add_noise(image_latents, noise, encode_timestep)

        latent_model_input = torch.cat([image_latents] * 2)
        zero_mask_shape = list(latent_model_input.shape)
        zero_mask_shape[1] = self.unet.config.in_channels - zero_mask_shape[1]
        zero_mask_latents = torch.zeros(zero_mask_shape, device=device, dtype=image_latents.dtype)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, encode_timestep)
        latent_model_input = torch.cat([latent_model_input, zero_mask_latents], dim=1)

        # 7. Predict the noise residual
        prompt_embeds = torch.cat([source_prompt_embeds, target_prompt_embeds])
        noise_pred = self.unet(
            latent_model_input,
            encode_timestep,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample

        if do_classifier_free_guidance:
            noise_pred_neg_src, noise_pred_source, noise_pred_uncond, noise_pred_target = noise_pred.chunk(4)
            noise_pred_source = noise_pred_neg_src + guidance_scale * (noise_pred_source - noise_pred_neg_src)
            noise_pred_target = noise_pred_uncond + guidance_scale * (noise_pred_target - noise_pred_uncond)
        else:
            noise_pred_source, noise_pred_target = noise_pred.chunk(2)

        # 8. Compute the mask from the absolute difference of predicted noise residuals
        # TODO: Consider smoothing mask guidance map
        mask_guidance_map = (
            torch.abs(noise_pred_target - noise_pred_source)
            .reshape(batch_size, num_maps_per_mask, *noise_pred_target.shape[-3:])
            .mean([1, 2])
        )
        mask_guidance_map = self.image_processor.resize(mask_guidance_map.unsqueeze(0), height, width)
        clamp_magnitude = mask_guidance_map.mean() * mask_thresholding_ratio
        semantic_mask_image = mask_guidance_map.clamp(0, clamp_magnitude) / clamp_magnitude
        semantic_mask_image = torch.where(semantic_mask_image <= 0.5, 0., 1.)
        mask_image = semantic_mask_image.cpu().squeeze(0)

        # Offload all models
        self.maybe_free_model_hooks()

        return mask_image