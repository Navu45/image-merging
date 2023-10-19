from typing import Callable, List, Optional, Union

import PIL.Image
import torch
from diffusers import DiffusionPipeline, PriorTransformer, KandinskyV22PriorEmb2EmbPipeline, UnCLIPScheduler
from diffusers.models import UNet2DConditionModel, VQModel
from diffusers.schedulers import DDPMScheduler
from diffusers.utils import (
    logging, )
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

from models.img2img_inpaint_pipeline import Img2ImgInpaintPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CombinedPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->movq"
    _load_connected_pipes = True

    unet: UNet2DConditionModel
    scheduler: DDPMScheduler
    movq: VQModel

    prior: PriorTransformer
    image_encoder: CLIPVisionModelWithProjection
    text_encoder: CLIPTextModelWithProjection

    tokenizer: CLIPTokenizer
    image_processor: CLIPImageProcessor

    def __init__(
            self,
            unet: UNet2DConditionModel,
            scheduler: DDPMScheduler,
            movq: VQModel,
            prior: PriorTransformer,
            prior_scheduler: UnCLIPScheduler,
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
            prior=prior,
            prior_scheduler=prior_scheduler,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_processor=image_processor
        )

        self.prior_pipe = KandinskyV22PriorEmb2EmbPipeline(
            prior=prior,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=prior_scheduler,
            image_processor=image_processor,
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
        self.prior_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
        self.decoder_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)

    def progress_bar(self, iterable=None, total=None):
        self.prior_pipe.progress_bar(iterable=iterable, total=total)
        self.decoder_pipe.progress_bar(iterable=iterable, total=total)
        self.decoder_pipe.enable_model_cpu_offload()

    def set_progress_bar_config(self, **kwargs):
        self.prior_pipe.set_progress_bar_config(**kwargs)
        self.decoder_pipe.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    def encode_image(
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

        # Offload all models
        self.maybe_free_model_hooks()

        return image_emb

    @torch.no_grad()
    def __call__(
            self,
            source_image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]],
            target_image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]],
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
    ):
        source_image_embeds, source_negative_image_embeds = self.prior_pipe(
            'a photograph of a human wearing specific clothes',
            source_image,
            negative_prompt='a picture of background behind the human'
        ).to_tuple()

        target_image_embeds, target_negative_image_embeds = self.prior_pipe(
            'a photograph of a human wearing specific clothes',
            target_image,
            negative_prompt='a picture of background behind the human'
        ).to_tuple()

        mask_image = self.decoder_pipe.generate_mask(
            image=source_image,
            height=height,
            width=width,
            target_image_embeds=target_image_embeds,
            target_negative_image_embeds=target_negative_image_embeds,
            source_image_embeds=source_image_embeds,
            source_negative_image_embeds=source_negative_image_embeds,
            output_type='pil'
        )

        # source_image = [source_image] if isinstance(source_image, PIL.Image.Image) else source_image
        # mask_image = [mask_image] if isinstance(mask_image, PIL.Image.Image) else mask_image
        #
        # outputs = self.decoder_pipe(
        #     source_image,
        #     source_image_embeds,
        #     target_image_embeds,
        #     mask_image,
        #     width=width,
        #     height=height,
        #     num_inference_steps=num_inference_steps,
        #     strength=strength,
        #     generator=generator,
        #     guidance_scale=guidance_scale,
        #     output_type=output_type,
        #     callback=callback,
        #     callback_steps=callback_steps,
        #     return_dict=return_dict,
        # )

        # Offload all models
        self.maybe_free_model_hooks()
        return mask_image