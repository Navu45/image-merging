from typing import Callable, List, Optional, Union, Dict

import PIL.Image
import torch
from diffusers.utils import (
    logging, )

from models.combined_pipeline_prior import CombinedPipelineV2

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class StyleInpaintPipeline(CombinedPipelineV2):
    @torch.no_grad()
    def __call__(
            self,
            source_image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]],
            target_image: Union[torch.FloatTensor, PIL.Image.Image,
                                List[torch.FloatTensor], List[PIL.Image.Image]] = None,
            style_image: Union[torch.FloatTensor, PIL.Image.Image,
                               List[torch.FloatTensor], List[PIL.Image.Image]] = None,
            prompts: Union[str, List[str]] = None,
            negative_prompts: Union[str, List[str]] = None,
            source_image_embeds: torch.FloatTensor = None,
            source_negative_image_embeds: torch.FloatTensor = None,
            target_image_embeds: torch.FloatTensor = None,
            target_negative_image_embeds: torch.FloatTensor = None,
            style_image_embeds: torch.FloatTensor = None,
            style_negative_image_embeds: torch.FloatTensor = None,
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
            return_masked_image: bool = False,
            use_only_target_embeds: bool = False,
            frequency: float = 2.0):
        batch_size = 1
        device = self._execution_device

        # Offload all models
        self.maybe_free_model_hooks()

        source_mask_image = self.generate_mask(source_image,
                                               prompts,
                                               negative_prompts,
                                               mask_threshold,
                                               device,
                                               height=height,
                                               width=width)

        style_embeds = (style_image_embeds, style_negative_image_embeds) if style_image_embeds is not None else \
            self.encode_image(style_image,
                              device,
                              batch_size,
                              num_images_per_prompt)
        style_image_embeds, style_negative_image_embeds = style_embeds

        target_embeds = (target_image_embeds, target_negative_image_embeds) if target_image_embeds is not None else \
            self.encode_image(target_image,
                              device,
                              batch_size,
                              num_images_per_prompt)
        target_image_embeds, target_negative_image_embeds = target_embeds

        outputs = self.decoder_pipe(
            source_image,
            source_image_embeds=target_image_embeds,
            target_image_embeds=style_image_embeds,
            mask_image=source_mask_image,
            target_negative_image_embeds=style_negative_image_embeds,
            source_negative_image_embeds=target_negative_image_embeds,
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
            return_masked_image=return_masked_image,
            use_alternation=True,
            frequency=frequency
        )

        # Offload all models
        self.maybe_free_model_hooks()

        return outputs.images[0], source_mask_image[0]
