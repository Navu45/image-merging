from typing import Callable, List, Optional, Union

import PIL.Image
import torch
from diffusers import DiffusionPipeline
from diffusers.models import UNet2DConditionModel, VQModel
from diffusers.schedulers import DDPMScheduler
from diffusers.utils import (
    logging,
)
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

from models.img2img_inpaint_pipeline import Img2ImgInpaintPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CombinedPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->movq"
    _load_connected_pipes = True

    def __init__(
            self,
            unet: UNet2DConditionModel,
            scheduler: DDPMScheduler,
            movq: VQModel,
            image_encoder: CLIPVisionModelWithProjection,
            text_encoder: CLIPTextModelWithProjection,
            tokenizer: CLIPTokenizer,
            image_processor: CLIPImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_processor=image_processor
        )

        self.decoder_pipe = Img2ImgInpaintPipeline(
            unet, scheduler, movq
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

    @torch.no_grad()
    def _encode_image(
        self,
        image: Union[torch.Tensor, List[PIL.Image.Image]],
        device,
        num_images_per_prompt,
    ):
        if not isinstance(image, torch.Tensor):
            image = self.image_processor(image, return_tensors="pt").pixel_values.to(
                dtype=self.image_encoder.dtype, device=device
            )

        image_emb = self.image_encoder(image)["image_embeds"]  # B, D
        image_emb = image_emb.repeat_interleave(num_images_per_prompt, dim=0)
        image_emb.to(device=device)

        return image_emb

    @torch.no_grad()
    def __call__(
            self,
            image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]],
            example_image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]],
            mask_image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]],
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_inference_steps: int = 100,
            guidance_scale: float = 4.0,
            num_images_per_prompt: int = 1,
            height: int = 512,
            width: int = 512,
            strength: float = 0.3,
            prior_guidance_scale: float = 4.0,
            prior_num_inference_steps: int = 25,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            return_dict: bool = True,
    ):
        source_image_embeds, target_image_embeds = self._encode_image([image, example_image],
                                                                      self._execution_device,
                                                                      num_images_per_prompt)

        image = [image] if isinstance(image, PIL.Image.Image) else image
        mask_image = [mask_image] if isinstance(mask_image, PIL.Image.Image) else mask_image

        if (
                isinstance(image, (list, tuple))
                and len(image) < source_image_embeds.shape[0]
                and source_image_embeds.shape[0] % len(image) == 0
        ):
            image = (source_image_embeds.shape[0] // len(image)) * image

        if (
                isinstance(mask_image, (list, tuple))
                and len(mask_image) < source_image_embeds.shape[0]
                and source_image_embeds.shape[0] % len(mask_image) == 0
        ):
            mask_image = (source_image_embeds.shape[0] // len(mask_image)) * mask_image

        outputs = self.decoder_pipe(
            image,
            source_image_embeds,
            target_image_embeds,
            mask_image,
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
        return outputs
