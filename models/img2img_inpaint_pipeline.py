from typing import Callable, Dict, Any, Union
from typing import List, Optional, Union

import PIL
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import ImagePipelineOutput, KandinskyV22InpaintPipeline, UNet2DConditionModel, DDPMScheduler, VQModel
from diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2_inpainting import prepare_mask, \
    prepare_mask_and_masked_image
from diffusers.utils import (
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from kornia.filters import median_blur
from torchvision.transforms.functional import resize
from transformers import CLIPImageProcessor
from transformers.image_transforms import to_pil_image

logger = logging.get_logger(__name__)


class Img2ImgInpaintPipeline(KandinskyV22InpaintPipeline):

    def __init__(self,
                 unet: UNet2DConditionModel,
                 scheduler: DDPMScheduler,
                 movq: VQModel,
                 image_processor: CLIPImageProcessor):
        super().__init__(unet, scheduler, movq)

        self.register_modules(
            image_processor=image_processor
        )

    def generate_mask(
            self,
            image: Union[torch.FloatTensor, PIL.Image.Image] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            target_image_embeds: Optional[torch.FloatTensor] = None,
            target_negative_image_embeds: Optional[torch.FloatTensor] = None,
            source_image_embeds: Optional[torch.FloatTensor] = None,
            source_negative_image_embeds: Optional[torch.FloatTensor] = None,
            num_maps_per_mask: Optional[int] = 10,
            mask_encode_strength: Optional[float] = 0.5,
            mask_thresholding_ratio: Optional[float] = 3.0,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            output_type: str = 'np',
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
        batch_size = target_image_embeds.shape[0]
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            source_image_embeds = source_image_embeds.repeat_interleave(num_maps_per_mask, dim=0)
            source_negative_image_embeds = source_negative_image_embeds.repeat_interleave(num_maps_per_mask, dim=0)

            source_image_embeds = torch.cat([source_negative_image_embeds, source_image_embeds], dim=0).to(
                dtype=self.unet.dtype, device=device
            )

            target_image_embeds = target_image_embeds.repeat_interleave(num_maps_per_mask, dim=0)
            target_negative_image_embeds = target_negative_image_embeds.repeat_interleave(num_maps_per_mask, dim=0)

            target_image_embeds = torch.cat([target_negative_image_embeds, target_image_embeds], dim=0).to(
                dtype=self.unet.dtype, device=device
            )

        # 4. Preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            image = [i.resize((width, height), resample=Image.BICUBIC, reducing_gap=1) for i in image]
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        if isinstance(image, np.ndarray):
            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(dtype=self.unet.dtype) / 127.5 - 1.0

        image = image.repeat_interleave(num_maps_per_mask, dim=0)

        # 5. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, _ = self.get_timesteps(num_inference_steps, mask_encode_strength, device)
        encode_timestep = timesteps[0]

        # 6. Prepare image latents and add noise with specified strength
        image_latents = self.prepare_latents(
            image, encode_timestep, batch_size, num_maps_per_mask, self.unet.dtype, device, generator
        )
        latent_model_input = torch.cat([image_latents] * (4 if do_classifier_free_guidance else 2))
        zero_mask_shape = list(latent_model_input.shape)
        zero_mask_shape[1] = self.unet.config.in_channels - zero_mask_shape[1]
        zero_mask_latents = torch.zeros(zero_mask_shape, device=device, dtype=image_latents.dtype)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, encode_timestep)
        latent_model_input = torch.cat([latent_model_input, zero_mask_latents], dim=1)

        # 7. Predict the noise residual
        image_embeds = torch.cat([source_image_embeds, target_image_embeds])
        added_cond_kwargs = {"image_embeds": image_embeds}
        noise_pred = self.unet(
            sample=latent_model_input,
            timestep=encode_timestep,
            encoder_hidden_states=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
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
        print('clamp_magnitude=', torch.isnan(clamp_magnitude).any())
        semantic_mask_image = mask_guidance_map.clamp(torch.tensor(0,
                                                                   dtype=clamp_magnitude.dtype,
                                                                   device=clamp_magnitude.device),
                                                      clamp_magnitude) / clamp_magnitude
        semantic_mask_image = resize(semantic_mask_image, [height, width])
        print('semantic_mask_image=', torch.isnan(semantic_mask_image).any())
        mask_image = torch.where(semantic_mask_image <= 0.5, 0., 255.).cpu().unsqueeze(0)
        print('mask_image_pipe=', torch.isnan(mask_image).any())
        mask_image = mask_image.squeeze(0)
        if output_type == "pil":
            mask_image = to_pil_image(mask_image)
        elif output_type == 'np':
            mask_image = mask_image.numpy()

        # Offload all models
        self.maybe_free_model_hooks()

        return mask_image, image_latents

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    self.movq.encode(image[i: i + 1]).latents for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.movq.encode(image).latents

            init_latents = self.movq.config.scaling_factor * init_latents

        init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        latents = init_latents

        return latents

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

        batch_size = 1
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

        # create initial latent
        latents = source_image
        latents = latents.repeat_interleave(num_images_per_prompt, dim=0)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
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
