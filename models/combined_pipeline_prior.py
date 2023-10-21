from typing import Callable, List, Optional, Union, Dict

import PIL.Image
import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.loaders import LoraLoaderMixin
from diffusers.models import UNet2DConditionModel, VQModel
from diffusers.schedulers import DDPMScheduler
from diffusers.utils import (
    logging, numpy_to_pil, )
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, \
    CLIPSegProcessor, CLIPSegForImageSegmentation, PreTrainedModel
from transformers.image_processing_utils import BaseImageProcessor

from models.img2img_inpaint_pipeline import Img2ImgInpaintPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CombinedPipelineV2(DiffusionPipeline, LoraLoaderMixin):
    model_cpu_offload_seq = "image_encoder->unet->movq"
    _load_connected_pipes = True

    def __init__(
            self,
            unet: UNet2DConditionModel,
            scheduler: DDPMScheduler,
            movq: VQModel,
            image_encoder: CLIPVisionModelWithProjection,
            image_processor: CLIPImageProcessor,
            segmentation_processor: CLIPSegProcessor,
            segmentation_model: CLIPSegForImageSegmentation,
            classifier_processor: BaseImageProcessor,
            classifier_model: PreTrainedModel,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
            image_encoder=image_encoder,
            image_processor=image_processor,
            segmentation_processor=segmentation_processor,
            segmentation_model=segmentation_model,
            classifier_processor=classifier_processor,
            classifier_model=classifier_model
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

    def get_mask(self, image, prompts, mask_threshold, device=None):
        inputs = self.segmentation_processor(
            text=prompts,
            images=[image] * len(prompts), padding="max_length", return_tensors="pt"
        ).to(device=device)
        inputs['pixel_values'] = inputs['pixel_values'].to(dtype=self.segmentation_model.dtype)
        outputs = self.segmentation_model(**inputs)

        mask = np.zeros(tuple(outputs.logits[0].shape) + (1,))
        for logits in outputs.logits:
            m = torch.sigmoid(logits).cpu().detach().unsqueeze(-1).numpy()
            m[m < mask_threshold] = 0
            m[m >= mask_threshold] = 1
            mask += m
            mask[mask < mask_threshold] = 0
            mask[mask >= mask_threshold] = 1
        return mask

    def generate_mask(self,
                      image,
                      prompt,
                      negative_prompt,
                      mask_threshold,
                      device,
                      height=512,
                      width=512,
                      output_type='pil'):

        mask = self.get_mask(image, prompt, mask_threshold, device)
        negative_mask = self.get_mask(image, negative_prompt, mask_threshold, device)
        mask -= negative_mask
        mask[mask < mask_threshold] = 0
        mask[mask >= mask_threshold] = 1
        if output_type == 'pil':
            mask = [m.resize((height, width)) for m in numpy_to_pil(mask)]
        elif output_type == 'pt':
            mask = torch.from_numpy(mask)

        return mask

    @torch.no_grad()
    def __call__(
            self,
            source_image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]],
            target_image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]],
            source_image_embeds: torch.FloatTensor,
            source_negative_image_embeds: torch.FloatTensor,
            target_image_embeds: torch.FloatTensor,
            target_negative_image_embeds: torch.FloatTensor,
            source_prompt: Union[str, List[str]] = None,
            source_negative_prompt: Union[str, List[str]] = None,
            target_prompt: Union[str, List[str]] = None,
            target_negative_prompt: Union[str, List[str]] = None,
            num_inference_steps: int = 100,
            guidance_scale: float = 4.0,
            num_images_per_prompt: int = 1,
            height: int = 512,
            width: int = 512,
            strength: float = 0.3,
            num_maps_per_mask: int = 10,
            mask_threshold: float = 0.3,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            output_type: Optional[str] = "pil",
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            return_dict: bool = True,
            cross_attention_kwargs: Dict = None,
    ):
        batch_size = 1
        device = self._execution_device

        # Offload all models
        self.maybe_free_model_hooks()

        source_mask_image = self.generate_mask(source_image,
                                               source_prompt,
                                               source_negative_prompt,
                                               mask_threshold,
                                               device,
                                               height=height,
                                               width=width)

        # target_mask_image = self.generate_mask(target_image,
        #                                        target_prompt,
        #                                        target_negative_prompt,
        #                                        mask_threshold,
        #                                        device,
        #                                        height=height,
        #                                        width=width, )

        # masked_target_image = numpy_to_pil(np.array(target_image) *
        #                                    np.array(target_mask_image[0].convert('RGB')))

        source_image_embeds, source_negative_image_embeds = self.encode_image(source_image,
                                                                              device,
                                                                              batch_size,
                                                                              num_images_per_prompt)

        target_image_embeds, target_negative_image_embeds = self.encode_image(target_image,
                                                                              device,
                                                                              batch_size,
                                                                              num_images_per_prompt)
        outputs = self.decoder_pipe(
            source_image,
            source_image_embeds,
            target_image_embeds,
            source_mask_image,
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

        return outputs.images[0], source_mask_image[0]
