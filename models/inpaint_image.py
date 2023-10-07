from typing import Union, Optional, List, Dict, Any

import PIL
import torch
from diffusers import PaintByExamplePipeline
from diffusers.utils.torch_utils import randn_tensor


class StableDiffusionInpaintPipeline(PaintByExamplePipeline):
    def encode_image(self, images):
        device, num_images_per_prompt = self._execution_device, len(images) if isinstance(images, List) else 1
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

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_pix2pix_zero
    # .StableDiffusionPix2PixZeroPipeline.prepare_image_latents
    def prepare_image_latents(self, image, batch_size, dtype, device, generator=None):
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

            if isinstance(generator, list):
                latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0)
            else:
                latents = self.vae.encode(image).latent_dist.sample(generator)

            latents = self.vae.config.scaling_factor * latents

        if batch_size != latents.shape[0]:
            if batch_size % latents.shape[0] == 0:
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
            target_prompt_embeds: Optional[torch.FloatTensor] = None,
            source_prompt_embeds: Optional[torch.FloatTensor] = None,
            num_maps_per_mask: Optional[int] = 10,
            mask_encode_strength: Optional[float] = 0.5,
            mask_thresholding_ratio: Optional[float] = 3.0,
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
        image = self.image_processor.preprocess(image).repeat_interleave(self.unet.config.in_channels, dim=0)

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
        print(latent_model_input.size())
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
        clamp_magnitude = mask_guidance_map.mean() * mask_thresholding_ratio
        semantic_mask_image = mask_guidance_map.clamp(0, clamp_magnitude) / clamp_magnitude
        semantic_mask_image = torch.where(semantic_mask_image <= 0.5, 0, 1)
        mask_image = semantic_mask_image.cpu().numpy()

        # 9. Convert to Numpy array or PIL.
        if output_type == "pil":
            mask_image = self.image_processor.numpy_to_pil(mask_image)

        # Offload all models
        self.maybe_free_model_hooks()

        return mask_image
