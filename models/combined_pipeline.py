from typing import Callable, List, Optional, Union, Dict

import PIL.Image
import torch
from diffusers import DiffusionPipeline, PriorTransformer, KandinskyV22PriorEmb2EmbPipeline, UnCLIPScheduler
from diffusers.loaders import LoraLoaderMixin
from diffusers.models import UNet2DConditionModel, VQModel
from diffusers.schedulers import DDPMScheduler
from diffusers.utils import (
    logging, )
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

from models.img2img_inpaint_pipeline import Img2ImgInpaintPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CombinedPipeline(DiffusionPipeline, LoraLoaderMixin):
    model_cpu_offload_seq = "image_encoder->unet->movq"
    _load_connected_pipes = True

    unet: UNet2DConditionModel
    scheduler: DDPMScheduler
    movq: VQModel

    image_encoder: CLIPVisionModelWithProjection
    image_processor: CLIPImageProcessor

    def __init__(
            self,
            unet: UNet2DConditionModel,
            scheduler: DDPMScheduler,
            movq: VQModel,
            image_encoder: CLIPVisionModelWithProjection,
            image_processor: CLIPImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
            image_encoder=image_encoder,
            image_processor=image_processor
        )

        self.decoder_pipe = Img2ImgInpaintPipeline(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
            image_processor=image_processor,
        )

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
        self.decoder_pipe.enable_xformers_memory_efficient_attention(attention_op)

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        self.decoder_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)

    def progress_bar(self, iterable=None, total=None):
        self.decoder_pipe.progress_bar(iterable=iterable, total=total)
        self.decoder_pipe.enable_model_cpu_offload()

    def set_progress_bar_config(self, **kwargs):
        self.decoder_pipe.set_progress_bar_config(**kwargs)

    # Copied from diffusers.pipelines.kandinsky.pipeline_kandinsky_prior.KandinskyPriorPipeline.get_zero_embed
    def get_zero_embed(self, batch_size=1, device=None):
        device = device or self.device
        zero_img = torch.zeros(1, 3, self.image_encoder.config.image_size, self.image_encoder.config.image_size).to(
            device=device, dtype=self.image_encoder.dtype
        )
        zero_image_emb = self.image_encoder(zero_img)["image_embeds"]
        zero_image_emb = zero_image_emb.repeat(batch_size, 1)
        return zero_image_emb

    @torch.no_grad()
    def encode_image(
            self,
            image: Union[torch.Tensor, List[PIL.Image.Image]],
            device,
            batch_size,
            num_images_per_prompt,
    ):
        if not isinstance(image, torch.Tensor):
            image = self.image_processor(image, return_tensors="pt").pixel_values.to(
                dtype=self.image_encoder.dtype, device=device
            )

        image_emb = self.image_encoder(image)["image_embeds"]  # B, D
        image_emb = image_emb.repeat_interleave(num_images_per_prompt, dim=0)
        image_emb.to(device=device)

        # Offload all models
        self.maybe_free_model_hooks()

        return image_emb, self.get_zero_embed(batch_size)

    @torch.no_grad()
    def __call__(
            self,
            source_image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]],
            target_image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]],
            mask_image: Union[torch.FloatTensor, PIL.Image.Image,
            List[torch.FloatTensor], List[PIL.Image.Image]] = None,
            prompt: Union[str, List[str]] = None,
            negative_prompt: Union[str, List[str]] = None,
            num_inference_steps: int = 100,
            guidance_scale: float = 4.0,
            num_images_per_prompt: int = 1,
            height: int = 512,
            width: int = 512,
            strength: float = 0.3,
            num_maps_per_mask: int = 10,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            output_type: Optional[str] = "pil",
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            return_dict: bool = True,
            cross_attention_kwargs: Dict = None,
    ):
        batch_size = 1
        device = self._execution_device

        source_image_embeds, source_negative_image_embeds = self.encode_image(source_image,
                                                                              device,
                                                                              batch_size,
                                                                              num_images_per_prompt)

        target_image_embeds, target_negative_image_embeds = self.encode_image(target_image,
                                                                              device,
                                                                              batch_size,
                                                                              num_images_per_prompt)

        # Offload all models
        self.maybe_free_model_hooks()

        if mask_image is None:
            mask_image, image_latents = self.decoder_pipe.generate_mask(
                image=source_image,
                height=height,
                width=width,
                target_image_embeds=target_image_embeds,
                target_negative_image_embeds=target_negative_image_embeds,
                source_image_embeds=source_image_embeds,
                source_negative_image_embeds=source_negative_image_embeds,
                output_type='pil',
                cross_attention_kwargs=cross_attention_kwargs
            )
        mask_image.save('mask.png')

        source_image = [source_image] if isinstance(source_image, PIL.Image.Image) else source_image
        mask_image = [mask_image] if isinstance(mask_image, PIL.Image.Image) else mask_image

        outputs = self.decoder_pipe(
            source_image,
            source_image_embeds,
            target_image_embeds,
            mask_image,
            source_negative_image_embeds=source_negative_image_embeds,
            target_negative_image_embeds=target_negative_image_embeds,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            strength=strength,
            generator=generator,
            guidance_scale=guidance_scale,
            output_type=output_type,
            callback=callback,
            callback_steps=callback_steps,
            return_dict=return_dict,
        )

        # Offload all models
        self.maybe_free_model_hooks()
        return outputs
