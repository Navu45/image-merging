from typing import Callable
from typing import List, Optional, Union

import PIL
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import ImagePipelineOutput, KandinskyV22InpaintPipeline
from diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2_inpainting import downscale_height_and_width, prepare_mask,\
    prepare_mask_and_masked_image

from diffusers.utils import (
    logging,
)

logger = logging.get_logger(__name__)


class Img2ImgInpaintPipeline(KandinskyV22InpaintPipeline):

    # Copied from diffusers.pipelines.kandinsky.pipeline_kandinsky_img2img.KandinskyImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    # @torch.no_grad()
    # def generate_mask(
    #     self,
    #     image: Union[torch.FloatTensor, PIL.Image.Image] = None,
    #     image_embeds: Optional[torch.FloatTensor] = None,
    #     target_prompt: Optional[Union[str, List[str]]] = None,
    #     target_negative_prompt: Optional[Union[str, List[str]]] = None,
    #     target_prompt_embeds: Optional[torch.FloatTensor] = None,
    #     target_negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    #     source_prompt: Optional[Union[str, List[str]]] = None,
    #     source_negative_prompt: Optional[Union[str, List[str]]] = None,
    #     source_prompt_embeds: Optional[torch.FloatTensor] = None,
    #     source_negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    #     num_maps_per_mask: Optional[int] = 10,
    #     mask_encode_strength: Optional[float] = 0.5,
    #     mask_thresholding_ratio: Optional[float] = 3.0,
    #     noise_level: Optional[float] = 0,
    #     num_inference_steps: int = 50,
    #     guidance_scale: float = 7.5,
    #     generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    #     output_type: Optional[str] = "np",
    #     cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    # ):
    #
    #     if (num_maps_per_mask is None) or (
    #         num_maps_per_mask is not None and (not isinstance(num_maps_per_mask, int) or num_maps_per_mask <= 0)
    #     ):
    #         raise ValueError(
    #             f"`num_maps_per_mask` has to be a positive integer but is {num_maps_per_mask} of type"
    #             f" {type(num_maps_per_mask)}."
    #         )
    #
    #     if mask_thresholding_ratio is None or mask_thresholding_ratio <= 0:
    #         raise ValueError(
    #             f"`mask_thresholding_ratio` has to be positive but is {mask_thresholding_ratio} of type"
    #             f" {type(mask_thresholding_ratio)}."
    #         )
    #
    #     # 2. Define call parameters
    #     if target_prompt is not None and isinstance(target_prompt, str):
    #         batch_size = 1
    #     elif target_prompt is not None and isinstance(target_prompt, list):
    #         batch_size = len(target_prompt)
    #     else:
    #         batch_size = target_prompt_embeds.shape[0]
    #     if cross_attention_kwargs is None:
    #         cross_attention_kwargs = {}
    #
    #     device = self._execution_device
    #     # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    #     # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    #     # corresponds to doing no classifier free guidance.
    #     do_classifier_free_guidance = guidance_scale > 1.0
    #     if isinstance(image, PIL.Image.Image):
    #         height, width = image.height, image.width
    #     else:
    #         height, width = image.shape[-2]
    #
    #     # 3. Encode input prompts
    #     (cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None)
    #     target_negative_prompt_embeds, target_prompt_embeds = self.encode_prompt(
    #         target_prompt,
    #         device,
    #         num_maps_per_mask,
    #         do_classifier_free_guidance,
    #         target_negative_prompt,
    #         prompt_embeds=target_prompt_embeds,
    #         negative_prompt_embeds=target_negative_prompt_embeds,
    #     )
    #     # For classifier free guidance, we need to do two forward passes.
    #     # Here we concatenate the unconditional and text embeddings into a single batch
    #     # to avoid doing two forward passes
    #     if do_classifier_free_guidance:
    #         target_prompt_embeds = torch.cat([target_negative_prompt_embeds, target_prompt_embeds])
    #
    #     source_negative_prompt_embeds, source_prompt_embeds = self.encode_prompt(
    #         source_prompt,
    #         device,
    #         num_maps_per_mask,
    #         do_classifier_free_guidance,
    #         source_negative_prompt,
    #         prompt_embeds=source_prompt_embeds,
    #         negative_prompt_embeds=source_negative_prompt_embeds,
    #     )
    #     if do_classifier_free_guidance:
    #         source_prompt_embeds = torch.cat([source_negative_prompt_embeds, source_prompt_embeds])
    #
    #     image_embeds = self._encode_image(
    #         image=image,
    #         device=device,
    #         batch_size=batch_size,
    #         num_images_per_prompt=num_maps_per_mask,
    #         do_classifier_free_guidance=do_classifier_free_guidance,
    #         noise_level=noise_level,
    #         generator=generator,
    #         image_embeds=image_embeds,
    #     ).repeat_interleave(num_maps_per_mask, dim=0)
    #     if do_classifier_free_guidance:
    #         image_embeds = torch.cat([image_embeds, image_embeds])
    #
    #     # 4. Preprocess image
    #     image = self.image_processor.preprocess(image).repeat_interleave(num_maps_per_mask, dim=0)
    #
    #     # 5. Set timesteps
    #     self.scheduler.set_timesteps(num_inference_steps, device=device)
    #     timesteps, _ = self.get_timesteps(num_inference_steps, mask_encode_strength, device)
    #     encode_timestep = timesteps[0]
    #
    #     # 6. Prepare image latents and add noise with specified strength
    #     image_latents = self.prepare_image_latents(
    #         image, batch_size * num_maps_per_mask, self.vae.dtype, device, generator
    #     )
    #     noise = randn_tensor(image_latents.shape, generator=generator, device=device, dtype=self.vae.dtype)
    #     image_latents = self.scheduler.add_noise(image_latents, noise, encode_timestep)
    #
    #     latent_model_input = torch.cat([image_latents] * (4 if do_classifier_free_guidance else 2))
    #     latent_model_input = self.scheduler.scale_model_input(latent_model_input, encode_timestep)
    #
    #     # 7. Predict the noise residual
    #     prompt_embeds = torch.cat([source_prompt_embeds, target_prompt_embeds])
    #     noise_pred = self.unet(
    #         latent_model_input,
    #         encode_timestep,
    #         class_labels=image_embeds,
    #         encoder_hidden_states=prompt_embeds,
    #         cross_attention_kwargs=cross_attention_kwargs,
    #     ).sample
    #
    #     if do_classifier_free_guidance:
    #         noise_pred_neg_src, noise_pred_source, noise_pred_uncond, noise_pred_target = noise_pred.chunk(4)
    #         noise_pred_source = noise_pred_neg_src + guidance_scale * (noise_pred_source - noise_pred_neg_src)
    #         noise_pred_target = noise_pred_uncond + guidance_scale * (noise_pred_target - noise_pred_uncond)
    #     else:
    #         noise_pred_source, noise_pred_target = noise_pred.chunk(2)
    #
    #     # 8. Compute the mask from the absolute difference of predicted noise residuals
    #     mask_guidance_map = (
    #         torch.abs(noise_pred_target - noise_pred_source)
    #         .reshape(batch_size, num_maps_per_mask, *noise_pred_target.shape[-3:])
    #         .mean([1, 2])
    #     )
    #     mask_guidance_map = self.image_processor.resize(mask_guidance_map.unsqueeze(0),
    #                                                     height, width)
    #     clamp_magnitude = mask_guidance_map.mean() * mask_thresholding_ratio
    #     semantic_mask_image = mask_guidance_map.clamp(0, clamp_magnitude) / clamp_magnitude
    #     semantic_mask_image = torch.where(semantic_mask_image <= 0.5, 0, 255)
    #     mask_image = semantic_mask_image.cpu().squeeze(0).numpy()
    #
    #     # 9. Convert to Numpy array or PIL.
    #     if output_type == "pil":
    #         mask_image = to_pil_image(mask_image)
    #         mask_image = cv2.medianBlur(np.asarray(mask_image), 59)
    #         mask_image = to_pil_image(mask_image.unsqueeze(0))
    #
    #
    #     # Offload all models
    #     self.maybe_free_model_hooks()
    #
    #     return mask_image

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.FloatTensor, PIL.Image.Image],
        source_image_embeds: Union[torch.FloatTensor, List[torch.FloatTensor]],
        target_image_embeds: Union[torch.FloatTensor, List[torch.FloatTensor]],
        mask_image: Union[torch.FloatTensor, PIL.Image.Image, np.ndarray],
        source_negative_image_embeds: Union[torch.FloatTensor, List[torch.FloatTensor]] = None,
        target_negative_image_embeds: Union[torch.FloatTensor, List[torch.FloatTensor]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 100,
        strength: float = 0.3,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        return_dict: bool = True,
    ):
        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(source_image_embeds, list):
            source_image_embeds = torch.cat(source_image_embeds, dim=0)
        if isinstance(source_negative_image_embeds, list):
            source_negative_image_embeds = torch.cat(source_negative_image_embeds, dim=0)

        if isinstance(target_image_embeds, list):
            target_image_embeds = torch.cat(target_image_embeds, dim=0)
        if isinstance(target_negative_image_embeds, list):
            target_negative_image_embeds = torch.cat(target_negative_image_embeds, dim=0)

        batch_size = source_image_embeds.shape[0] * num_images_per_prompt

        if do_classifier_free_guidance:
            source_image_embeds = source_image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            source_negative_image_embeds = source_negative_image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

            source_image_embeds = torch.cat([source_negative_image_embeds, source_image_embeds], dim=0).to(
                dtype=self.unet.dtype, device=device
            )

            target_image_embeds = target_image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            target_negative_image_embeds = target_negative_image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

            target_image_embeds = torch.cat([target_negative_image_embeds, target_image_embeds], dim=0).to(
                dtype=self.unet.dtype, device=device
            )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps_tensor = self.scheduler.timesteps

        # preprocess image and mask
        mask_image, source_image = prepare_mask_and_masked_image(source_image, mask_image, height, width)

        source_image = source_image.to(dtype=source_image_embeds.dtype, device=device)
        source_image = self.movq.encode(source_image)["latents"]

        mask_image = mask_image.to(dtype=target_image_embeds.dtype, device=device)

        image_shape = tuple(source_image.shape[-2:])
        mask_image = F.interpolate(
            mask_image,
            image_shape,
            mode="nearest",
        )
        mask_image = prepare_mask(mask_image)
        masked_image = source_image * mask_image

        mask_image = mask_image.repeat_interleave(num_images_per_prompt, dim=0)
        masked_image = masked_image.repeat_interleave(num_images_per_prompt, dim=0)
        if do_classifier_free_guidance:
            mask_image = mask_image.repeat(2, 1, 1, 1)
            masked_image = masked_image.repeat(2, 1, 1, 1)

        num_channels_latents = self.movq.config.latent_channels

        height, width = downscale_height_and_width(height, width, self.movq_scale_factor)

        # create initial latent
        latents = self.movq.encode(source_image)["latents"]
        latents = latents.repeat_interleave(num_images_per_prompt, dim=0)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        height, width = downscale_height_and_width(height, width, self.movq_scale_factor)
        latents = self.prepare_latents(
            latents, latent_timestep, batch_size, num_images_per_prompt, source_image_embeds.dtype, device, generator
        )
        noise = torch.clone(latents)
        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = torch.cat([latent_model_input, masked_image, mask_image], dim=1)

            added_cond_kwargs = {"image_embeds": target_image_embeds}
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                class_labels=source_image_embeds,
                encoder_hidden_states=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred, variance_pred = noise_pred.split(latents.shape[1], dim=1)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                _, variance_pred_text = variance_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = torch.cat([noise_pred, variance_pred_text], dim=1)

            if not (
                    hasattr(self.scheduler.config, "variance_type")
                    and self.scheduler.config.variance_type in ["learned", "learned_range"]
            ):
                noise_pred, _ = noise_pred.split(latents.shape[1], dim=1)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
                generator=generator,
            )[0]
            init_latents_proper = source_image[:1]
            init_mask = mask_image[:1]

            if i < len(timesteps_tensor) - 1:
                noise_timestep = timesteps_tensor[i + 1]
                init_latents_proper = self.scheduler.add_noise(
                    init_latents_proper, noise, torch.tensor([noise_timestep])
                )

            latents = init_mask * init_latents_proper + (1 - init_mask) * latents

            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

        # post-processing
        latents = mask_image[:1] * source_image[:1] + (1 - mask_image[:1]) * latents
        source_image = self.movq.decode(latents, force_not_quantize=True)["sample"]

        # Offload all models
        self.maybe_free_model_hooks()

        if output_type not in ["pt", "np", "pil"]:
            raise ValueError(f"Only the output types `pt`, `pil` and `np` are supported not output_type={output_type}")

        if output_type in ["np", "pil"]:
            source_image = source_image * 0.5 + 0.5
            source_image = source_image.clamp(0, 1)
            source_image = source_image.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            source_image = self.numpy_to_pil(source_image)

        if not return_dict:
            return (source_image,)

        return ImagePipelineOutput(images=source_image)
