from typing import Any, Callable
from typing import Dict, List, Optional, Union

import PIL
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers.pipelines.paint_by_example.pipeline_paint_by_example import prepare_mask_and_masked_image
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.utils import (
    deprecate,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from transformers.image_transforms import to_pil_image

logger = logging.get_logger(__name__)


class UnclipMergePipeline(StableUnCLIPImg2ImgPipeline):
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
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


    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = self.vae.encode(image).latent_dist.sample(generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    def prepare_image_latents(self, image, batch_size, dtype, device, generator):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )
        # image = self.feature_extractor(images=image, return_tensors="pt").pixel_values
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

    @torch.no_grad()
    def generate_mask(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        target_prompt: Optional[Union[str, List[str]]] = None,
        target_negative_prompt: Optional[Union[str, List[str]]] = None,
        target_prompt_embeds: Optional[torch.FloatTensor] = None,
        target_negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        source_prompt: Optional[Union[str, List[str]]] = None,
        source_negative_prompt: Optional[Union[str, List[str]]] = None,
        source_prompt_embeds: Optional[torch.FloatTensor] = None,
        source_negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        num_maps_per_mask: Optional[int] = 10,
        mask_encode_strength: Optional[float] = 0.5,
        mask_thresholding_ratio: Optional[float] = 3.0,
        noise_level: Optional[float] = 0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "np",
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
        if target_prompt is not None and isinstance(target_prompt, str):
            batch_size = 1
        elif target_prompt is not None and isinstance(target_prompt, list):
            batch_size = len(target_prompt)
        else:
            batch_size = target_prompt_embeds.shape[0]
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        if isinstance(image, PIL.Image.Image):
            height, width = image.height, image.width
        else:
            height, width = image.shape[-2]

        # 3. Encode input prompts
        (cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None)
        target_negative_prompt_embeds, target_prompt_embeds = self.encode_prompt(
            target_prompt,
            device,
            num_maps_per_mask,
            do_classifier_free_guidance,
            target_negative_prompt,
            prompt_embeds=target_prompt_embeds,
            negative_prompt_embeds=target_negative_prompt_embeds,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            target_prompt_embeds = torch.cat([target_negative_prompt_embeds, target_prompt_embeds])

        source_negative_prompt_embeds, source_prompt_embeds = self.encode_prompt(
            source_prompt,
            device,
            num_maps_per_mask,
            do_classifier_free_guidance,
            source_negative_prompt,
            prompt_embeds=source_prompt_embeds,
            negative_prompt_embeds=source_negative_prompt_embeds,
        )
        if do_classifier_free_guidance:
            source_prompt_embeds = torch.cat([source_negative_prompt_embeds, source_prompt_embeds])

        image_embeds = self._encode_image(
            image=image,
            device=device,
            batch_size=batch_size,
            num_images_per_prompt=num_maps_per_mask,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_level=noise_level,
            generator=generator,
            image_embeds=image_embeds,
        ).repeat_interleave(num_maps_per_mask, dim=0)
        if do_classifier_free_guidance:
            image_embeds = torch.cat([image_embeds, image_embeds])

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

        latent_model_input = torch.cat([image_latents] * (4 if do_classifier_free_guidance else 2))
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, encode_timestep)

        # 7. Predict the noise residual
        prompt_embeds = torch.cat([source_prompt_embeds, target_prompt_embeds])
        noise_pred = self.unet(
            latent_model_input,
            encode_timestep,
            class_labels=image_embeds,
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
        mask_guidance_map = (
            torch.abs(noise_pred_target - noise_pred_source)
            .reshape(batch_size, num_maps_per_mask, *noise_pred_target.shape[-3:])
            .mean([1, 2])
        )
        mask_guidance_map = self.image_processor.resize(mask_guidance_map.unsqueeze(0),
                                                        height, width)
        clamp_magnitude = mask_guidance_map.mean() * mask_thresholding_ratio
        semantic_mask_image = mask_guidance_map.clamp(0, clamp_magnitude) / clamp_magnitude
        semantic_mask_image = torch.where(semantic_mask_image <= 0.5, 0, 255)
        mask_image = semantic_mask_image.cpu().squeeze(0).numpy()

        # 9. Convert to Numpy array or PIL.
        if output_type == "pil":
            mask_image = to_pil_image(mask_image)
            mask_image = cv2.medianBlur(np.asarray(mask_image), 59)
            mask_image = to_pil_image(mask_image.unsqueeze(0))


        # Offload all models
        self.maybe_free_model_hooks()

        return mask_image

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents


    @torch.no_grad()
    def generate(
        self,
        target_image: Union[torch.FloatTensor, PIL.Image.Image],
        source_image: Union[torch.FloatTensor, PIL.Image.Image],
        mask_image: Union[torch.FloatTensor, PIL.Image.Image],
        num_inference_steps: int = 50,
        noise_level: int = 0,
        guidance_scale: float = 5.0,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
    ):
        # 1. Define call parameters
        if isinstance(source_image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(source_image, list):
            batch_size = len(source_image)
        else:
            batch_size = source_image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 2. Preprocess mask and image
        mask, masked_image = prepare_mask_and_masked_image(source_image, mask_image)
        height, width = masked_image.shape[-2:]

        # 4. Encode input image
        source_image_embeddings = self._encode_image(
            image=source_image,
            device=device,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_level=noise_level,
            generator=generator,
            image_embeds=None
        )

        target_image_embeddings = self._encode_image(
            image=target_image,
            device=device,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_level=noise_level,
            generator=generator,
            image_embeds=None
        )

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            source_image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare mask latent variables
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            target_image_embeddings.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        # 8. Check that sizes of mask, masked image and latents match
        num_channels_mask = mask.shape[1]
        num_channels_masked_image = masked_image_latents.shape[1]
        if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                " `pipeline.unet` or your `mask_image` or `image` input."
            )

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, masked_image_latents, mask], dim=1)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t,
                                       class_labels=source_image_embeddings,
                                       encoder_hidden_states=target_image_embeddings).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        self.maybe_free_model_hooks()

        if not output_type == "latent":
            source_image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            source_image, has_nsfw_concept = self.run_safety_checker(source_image, device, source_image_embeddings.dtype)
        else:
            source_image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * source_image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        source_image = self.image_processor.postprocess(source_image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return source_image, has_nsfw_concept

        return StableDiffusionPipelineOutput(images=source_image, nsfw_content_detected=has_nsfw_concept)