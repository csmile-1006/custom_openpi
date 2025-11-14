# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import logging

import random
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tree
from einops import rearrange
from PIL import Image
from pydantic import Field, PrivateAttr
from transformers import AutoProcessor, ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import InvertibleModalityTransform

from .backbone.eagle_backbone import DEFAULT_EAGLE_PATH
from .backbone.qwen_2_5_vl_backbone import DEFAULT_QWEN_PATH
from .backbone.paligemma_backbone import DEFAULT_PALIGEMMA_PATH

from qwen_vl_utils import process_vision_info as qwen_process_vision_info

from PIL import Image, ImageEnhance, ImageFilter

def formalize_language(language: str) -> str:
    """
    1. Force lowercase
    2. Remove all punctuations
    """
    language = language.lower()
    language = re.sub(r"[^\w\s]", "", language)
    return language


def build_eagle_processor(eagle_path: str) -> ProcessorMixin:
    eagle_processor = AutoProcessor.from_pretrained(
        eagle_path, trust_remote_code=True, use_fast=True
    )
    eagle_processor.tokenizer.padding_side = "left"
    return eagle_processor


def collate(features: List[dict], eagle_processor) -> dict:
    batch = {}
    keys = features[0].keys()

    text_list = []
    image_inputs = []
    for key in keys:
        logging.debug(f"[DEBUG] Handling collate key: {key}")
        values = [elem[key] for elem in features]

        if key == "eagle_content":
            for v in values:
                curr_text_list = v["text_list"]
                curr_image_inputs = v["image_inputs"]
                text_list += curr_text_list
                image_inputs += curr_image_inputs
        elif key == "eagle_content_aug":
            for v in values:
                curr_text_list = v["text_list"]
                curr_image_inputs = v["image_inputs"]
                text_list += curr_text_list
                image_inputs += curr_image_inputs
        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            batch[key] = torch.cat(values)
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            batch[key] = torch.from_numpy(np.stack(values))
    
    eagle_inputs = eagle_processor(
        text=text_list, images=image_inputs, return_tensors="pt", padding=True
    )
    for k, v in eagle_inputs.items():
        k = "eagle_" + k
        batch[k] = v
    
    # print(f"[DEBUG] batch keys: {batch.keys()}")
    # print(f"[DEBUG] shape of eagle_inputs: {batch['eagle_input_ids'].shape=}, {batch['eagle_attention_mask'].shape=}, {batch['eagle_pixel_values'].shape=}")
    # print(f"[DEBUG] Non eagle shape: {batch['state'].shape=}, {batch['state_mask'].shape=}, {batch['action'].shape=}, {batch['action_mask'].shape=}")

    return batch

def build_qwen_processor(qwen_path: str) -> ProcessorMixin:
    qwen_processor = AutoProcessor.from_pretrained(
        qwen_path, trust_remote_code=True, use_fast=True
    )
    qwen_processor.tokenizer.padding_side = "left"
    return qwen_processor

def qwen_collate(features: List[dict], qwen_processor) -> dict:
    batch = {}
    keys = features[0].keys()
    # print(f"[DEBUG] Qwen collate keys: {keys}")
    for key in keys:
        values = [elem[key] for elem in features]

        if key == "qwen_content":
            text_list = []
            image_inputs = []
            for v in values:
                curr_text_list = v["text_list"]
                curr_image_inputs = v["image_inputs"]
                text_list += curr_text_list
                image_inputs += curr_image_inputs
            qwen_inputs = qwen_processor(
                text=text_list, images=image_inputs, return_tensors="pt", padding=True
            )
            for k, v in qwen_inputs.items():
                k = "qwen_" + k
                batch[k] = v
        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            batch[key] = torch.cat(values)
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            try:
                batch[key] = torch.from_numpy(np.stack(values))
            except Exception as e:
                raise ValueError(
                    f"Error stacking values for key '{key}': {e}. "
                    f"Values: {values}. Shapes: {[v.shape if isinstance(v, torch.Tensor) else np.array(v).shape for v in values]}"
                )
    return batch

def build_paligemma_processor(paligemma_path: str) -> ProcessorMixin:
    paligemma_processor = AutoProcessor.from_pretrained(
        paligemma_path, trust_remote_code=True, use_fast=True
    )
    paligemma_processor.tokenizer.padding_side = "left"
    return paligemma_processor

def paligemma_collate(features: List[dict], paligemma_processor) -> dict:
    batch = {}
    keys = features[0].keys()

    for key in keys:
        values = [elem[key] for elem in features]

        if key == "paligemma_content":
            text_list = []
            image_inputs = []
            for v in values:
                curr_text_list = v["text_list"]
                curr_image_inputs = v["image_inputs"]
                text_list += curr_text_list
                image_inputs += curr_image_inputs
            # print(f"[DEBUG] Paligemma processor length of text_list: {len(text_list[0])} and image_inputs: {len(image_inputs[0])}")
            # print(f"[DEBUG] Paligemma processor text_list: {text_list}")
            # print(f"[DEBUG] Paligemma processor image_inputs: {image_inputs}")
            paligemma_inputs = paligemma_processor(
                text=text_list, images=image_inputs, return_tensors="pt", padding=True
            )
            for k, v in paligemma_inputs.items():
                k = "paligemma_" + k
                batch[k] = v
        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            batch[key] = torch.cat(values)
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            batch[key] = torch.from_numpy(np.stack(values))
    return batch

class DefaultDataCollator(DataCollatorMixin):
    def __init__(self, eagle_path: str = DEFAULT_EAGLE_PATH):
        super().__init__()
        self.eagle_processor = build_eagle_processor(eagle_path)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate(features, self.eagle_processor)

class QwenDataCollator(DataCollatorMixin):
    def __init__(self, qwen_path: str = DEFAULT_QWEN_PATH):
        super().__init__()
        self.qwen_processor = build_qwen_processor(qwen_path)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return qwen_collate(features, self.qwen_processor)
    
class PaligemmaDataCollator(DataCollatorMixin):
    def __init__(self, paligemma_path: str = DEFAULT_PALIGEMMA_PATH):
        super().__init__()
        self.paligemma_processor = build_paligemma_processor(paligemma_path)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return paligemma_collate(features, self.paligemma_processor)

class GR00TTransform(InvertibleModalityTransform):

    backbone_model_type: str = "eagle"

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )
    formalize_language: bool = Field(default=False, description="Formalize language if True.")
    embodiment_tag_mapping: dict[str, int] = Field(
        description="The projector index of each embodiment tag.",
        default=EMBODIMENT_TAG_MAPPING,
    )
    language_dropout_prob: float = Field(
        default=0.0,
        description="Dropout probability for language.",
    )

    # Private attributes to keep track of shapes/dimensions across apply/unapply
    _language_key: Optional[list[str]] = PrivateAttr(default=None)

    eagle_processor: ProcessorMixin = Field(default=build_eagle_processor(DEFAULT_EAGLE_PATH))
    qwen_processor: ProcessorMixin = Field(default=build_qwen_processor(DEFAULT_QWEN_PATH))
    paligemma_processor: ProcessorMixin = Field(default=build_paligemma_processor(DEFAULT_PALIGEMMA_PATH))
    # XEmbDiT arguments
    default_instruction: str = Field(default="Perform the default behavior.")
    max_state_dim: int
    max_action_dim: int
    state_horizon: int
    action_horizon: int

    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata for the transform."""
        super().set_metadata(dataset_metadata)
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert (
            self.embodiment_tag is not None
        ), "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            if "annotation" in key:
                modality = "language"
            else:
                try:
                    modality, _ = key.split(".")
                except:  # noqa: E722
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")

        # Handle language
        if "language" in grouped_keys:
            language_keys = grouped_keys["language"]
            assert len(language_keys) == 1, f"{language_keys=}"
            self._language_key = language_keys[0]
        return is_batched, batch_size

    def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
        """
        Args:
            batch:
                video: [V, T, C, H, W]
        Returns: required input with the format `BatchFeature`
        """
        if self.backbone_model_type == "eagle":
            images = batch["images"]  # [V, T, C, H, W]
            images.shape[0]

            np_images = rearrange(images, "v t c h w -> (t v) c h w")
            text_content = []

            # handle language
            lang = batch["language"]
            if isinstance(lang, list):
                lang = lang[0]
            text_content.append({"type": "text", "text": lang})
            eagle_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
            eagle_image = [{"type": "image", "image": img} for img in eagle_images]
            eagle_conversation = [
                {
                    "role": "user",
                    "content": eagle_image + text_content,
                }
            ]

            text_list = [
                self.eagle_processor.apply_chat_template(
                    eagle_conversation, tokenize=False, add_generation_prompt=True
                )
            ]
            image_inputs, video_inputs = self.eagle_processor.process_vision_info(eagle_conversation)
            eagle_content = {
                "image_inputs": image_inputs,
                "video_inputs": video_inputs,
                "text_list": text_list,
            }
            inputs = {}
            inputs["eagle_content"] = eagle_content
            return inputs
        
        elif self.backbone_model_type == "qwen2_5_vl" or self.backbone_model_type == "qwen2_5_vl_7b":
            images = batch["images"]  # [V, T, C, H, W]
            images.shape[0]

            np_images = rearrange(images, "v t c h w -> (t v) c h w")
            text_content = []

            # handle language
            lang = batch["language"]
            if isinstance(lang, list):
                lang = lang[0]
            text_content.append({"type": "text", "text": lang})
            # print(f"[DEBUG] Processing Qwen2.5 VL inputs with {len(np_images)} images.")
            qwen_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
            qwen_image = [{"type": "image", "image": img} for img in qwen_images]
            qwen_conversation = [
                {
                    "role": "user",
                    "content": qwen_image + text_content,
                }
            ]
            text_list = [
                self.qwen_processor.apply_chat_template(
                    qwen_conversation, tokenize=False, add_generation_prompt=True
                )
            ]
            image_inputs, video_inputs = qwen_process_vision_info(qwen_conversation)
            qwen_content = {
                "image_inputs": image_inputs,
                "video_inputs": video_inputs,
                "text_list": text_list,
            }
            inputs = {}
            inputs["qwen_content"] = qwen_content
            return inputs
        elif self.backbone_model_type == "paligemma":
            images = batch["images"]  # [V, T, C, H, W]
            images.shape[0]

            np_images = rearrange(images, "v t c h w -> (t v) c h w")
            text_content = []

            # handle language
            lang = batch["language"]
            if isinstance(lang, list):
                lang = lang[0]
            paligemma_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
            text_content.append("<image>\n<image>\nanswer en " + lang)

            text_list = [
                text_content
            ][0]
            paligemma_content = {
                "image_inputs": [paligemma_images],
                "text_list": text_list,
            }
            inputs = {}
            inputs["paligemma_content"] = paligemma_content
            return inputs
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported: eagle, qwen2_5_vl, paligemma.")

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        ## TODO(YL, FH): check if this is correct
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        return images

    def _prepare_language(self, data: dict):
        """Tokenize data['language'] (or default_instruction if missing)."""
        if self._language_key is not None:
            raw_language = data[self._language_key]
            if isinstance(raw_language, list):
                raw_language = raw_language[0]

            # Language dropout
            if self.training and self.language_dropout_prob > 1e-9:
                if random.random() < self.language_dropout_prob:
                    raw_language = self.default_instruction
        else:
            raw_language = self.default_instruction

        # print(f"RAW Language: {raw_language=}")

        return raw_language

    def _prepare_state(self, data: dict):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        Return (state, state_mask, n_state_tokens).
        """
        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = data["state"]
        
        if len(state.shape) == 1:
            state = state[None, :]
        assert state.shape[0] == self.state_horizon, f"{state.shape=}, {self.state_horizon=}"

        n_state_dims = state.shape[-1]
        # print(f"state shape: {state.shape=}, should be: {np.zeros((self.state_horizon, self.max_state_dim)).shape=}, max_state_dim: {self.max_state_dim=}")
        # Instead of asserting, just take the first max_state_dim dimensions if needed
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            # Pad up to max_state_dim if smaller
            state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")

        # Create mask for real state dims
        state_mask = np.zeros_like(state).astype(bool)
        state_mask[:, :n_state_dims] = True

        # We only have 1 "proprio" token to represent the entire state
        n_state_tokens = state.shape[0]
        return state, state_mask, n_state_tokens

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]
        assert actions.shape[0] == self.action_horizon, f"{actions.shape=}, {self.action_horizon=}"

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens

    def apply_single(self, data: dict) -> dict:
        transformed_data = {}

        # 1) Prepare video and language with vlm processing.
        images = self._prepare_video(data)
        images = images.astype(np.uint8)
        language = self._prepare_language(data)
        batch_data = {"images": images, "language": language}
        vlm_outputs = self._apply_vlm_processing(batch_data)

        # 2) Prepare state
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        if self.training:
            # 3) Prepare actions
            transformed_data["segmentation_target"] = np.zeros((2,))
            transformed_data["segmentation_target_mask"] = np.zeros((1,))
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask

        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        if self.training:
            action_and_mask_keys = ["action", "action_mask"]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape
                for key in action_and_mask_keys
            ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"

        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension.
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        if self.backbone_model_type == "eagle":
            # Eagle processor collates the data.
            return collate(data_split_processed, self.eagle_processor)
        elif self.backbone_model_type == "qwen2_5_vl" or self.backbone_model_type == "qwen2_5_vl_7b":
            # Qwen processor collates the data.
            return qwen_collate(data_split_processed, self.qwen_processor)
        elif self.backbone_model_type == "paligemma":
            # Paligemma processor collates the data.
            return paligemma_collate(data_split_processed, self.paligemma_processor)
        else:
            raise ValueError(f"Unsupported model type: {self.backbone_model_type}. Supported: eagle, qwen2_5_vl, paligemma.")

    def apply(self, data: dict) -> dict:
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)


class GR00TTransformCL(InvertibleModalityTransform):

    backbone_model_type: str = "eagle"

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )
    formalize_language: bool = Field(default=False, description="Formalize language if True.")
    embodiment_tag_mapping: dict[str, int] = Field(
        description="The projector index of each embodiment tag.",
        default=EMBODIMENT_TAG_MAPPING,
    )
    language_dropout_prob: float = Field(
        default=0.0,
        description="Dropout probability for language.",
    )

    # Private attributes to keep track of shapes/dimensions across apply/unapply
    _language_key: Optional[list[str]] = PrivateAttr(default=None)

    eagle_processor: ProcessorMixin = Field(default=build_eagle_processor(DEFAULT_EAGLE_PATH))
    qwen_processor: ProcessorMixin = Field(default=build_qwen_processor(DEFAULT_QWEN_PATH))
    paligemma_processor: ProcessorMixin = Field(default=build_paligemma_processor(DEFAULT_PALIGEMMA_PATH))
    # XEmbDiT arguments
    default_instruction: str = Field(default="Perform the default behavior.")
    max_state_dim: int
    max_action_dim: int
    state_horizon: int
    action_horizon: int

    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None

    # contrastive_augmentations: Optional[ContrastiveAugmentations] = PrivateAttr(default=None)

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata for the transform."""
        super().set_metadata(dataset_metadata)
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert (
            self.embodiment_tag is not None
        ), "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            if "annotation" in key:
                modality = "language"
            else:
                try:
                    modality, _ = key.split(".")
                except:  # noqa: E722
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")

        # Handle language
        if "language" in grouped_keys:
            language_keys = grouped_keys["language"]
            assert len(language_keys) == 1, f"{language_keys=}"
            self._language_key = language_keys[0]
        return is_batched, batch_size

    def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
        """
        Args:
            batch:
                video: [V, T, C, H, W]
        Returns: required input with the format `BatchFeature`
        """
        if self.backbone_model_type == "eagle":
            images = batch["images"]  # [V, T, C, H, W]
            images.shape[0]

            np_images = rearrange(images, "v t c h w -> (t v) c h w")
            text_content = []

            # handle language
            lang = batch["language"]
            if isinstance(lang, list):
                lang = lang[0]
            text_content.append({"type": "text", "text": lang})
            eagle_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
            eagle_image = [{"type": "image", "image": img} for img in eagle_images]
            eagle_conversation = [
                {
                    "role": "user",
                    "content": eagle_image + text_content,
                }
            ]

            text_list = [
                self.eagle_processor.apply_chat_template(
                    eagle_conversation, tokenize=False, add_generation_prompt=True
                )
            ]
            image_inputs, video_inputs = self.eagle_processor.process_vision_info(eagle_conversation)
            eagle_content = {
                "image_inputs": image_inputs,
                "video_inputs": video_inputs,
                "text_list": text_list,
            }
            inputs = {}
            inputs["eagle_content"] = eagle_content
            return inputs
        
        elif self.backbone_model_type == "qwen2_5_vl" or self.backbone_model_type == "qwen2_5_vl_7b":
            images = batch["images"]  # [V, T, C, H, W]
            images.shape[0]

            np_images = rearrange(images, "v t c h w -> (t v) c h w")
            text_content = []

            # handle language
            lang = batch["language"]
            if isinstance(lang, list):
                lang = lang[0]
            text_content.append({"type": "text", "text": lang})
            # print(f"[DEBUG] Processing Qwen2.5 VL inputs with {len(np_images)} images.")
            qwen_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
            qwen_image = [{"type": "image", "image": img} for img in qwen_images]
            qwen_conversation = [
                {
                    "role": "user",
                    "content": qwen_image + text_content,
                }
            ]
            text_list = [
                self.qwen_processor.apply_chat_template(
                    qwen_conversation, tokenize=False, add_generation_prompt=True
                )
            ]
            image_inputs, video_inputs = qwen_process_vision_info(qwen_conversation)
            qwen_content = {
                "image_inputs": image_inputs,
                "video_inputs": video_inputs,
                "text_list": text_list,
            }
            inputs = {}
            inputs["qwen_content"] = qwen_content
            return inputs
        elif self.backbone_model_type == "paligemma":
            images = batch["images"]  # [V, T, C, H, W]
            images.shape[0]

            np_images = rearrange(images, "v t c h w -> (t v) c h w")
            text_content = []

            # handle language
            lang = batch["language"]
            if isinstance(lang, list):
                lang = lang[0]
            paligemma_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
            text_content.append("<image>\n<image>\nanswer en " + lang)

            text_list = [
                text_content
            ][0]
            paligemma_content = {
                "image_inputs": [paligemma_images],
                "text_list": text_list,
            }
            inputs = {}
            inputs["paligemma_content"] = paligemma_content
            return inputs
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported: eagle, qwen2_5_vl, paligemma.")

    def _prepare_video(self, data: dict, augmentation: bool = False):
        """Process, stack, and pad images from data['video']."""
        ## TODO(YL, FH): check if this is correct
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )

        if augmentation:

            augmented_images = []
            # Apply augmentations to each image in the video
            for v in range(images.shape[0]):  # For each view
                for t in range(images.shape[1]):  # For each timestep
                    img = images[v, t]
                    
                    # Convert to PIL for easier augmentation
                    pil_img = Image.fromarray(np.transpose(img, (1, 2, 0)))
                    
                    if random.random() > 0.5:
                        brightness = random.uniform(0.9, 1.1)
                        contrast = random.uniform(0.9, 1.1)
                        saturation = random.uniform(0.9, 1.1)
                        pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness)
                        pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast)
                        pil_img = ImageEnhance.Color(pil_img).enhance(saturation)

                    # 2. Gaussian noise (add to numpy after all PIL ops)
                    # (Defer until after augmentations, see below)

                    # 3. Gaussian blur (rare, small)
                    if random.random() < 0.1:
                        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1))

                    # 4. Random grayscale (rare)
                    if random.random() < 0.05:
                        pil_img = pil_img.convert('L').convert('RGB')

                    # 5. Gamma (exposure)
                    if random.random() < 0.3:
                        gamma = random.uniform(0.95, 1.05)
                        img_np = np.array(pil_img).astype(np.float32) / 255.0
                        img_np = np.power(img_np, gamma)
                        img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_np)

                    # 6. Cutout / Random Erasing (optional)
                    if random.random() < 0.1:
                        width, height = pil_img.size
                        erase_w = int(width * random.uniform(0.05, 0.15))
                        erase_h = int(height * random.uniform(0.05, 0.15))
                        x0 = random.randint(0, width - erase_w)
                        y0 = random.randint(0, height - erase_h)
                        pil_img.paste((0, 0, 0), (x0, y0, x0 + erase_w, y0 + erase_h))

                    # Convert back to numpy
                    img_array = np.transpose(np.array(pil_img), (2, 0, 1)).astype(np.float32)

                    # 2. Gaussian noise (now add to numpy array)
                    if random.random() < 0.3:
                        noise = np.random.normal(0, 4, size=img_array.shape)  # mean 0, std 4
                        img_array = img_array + noise
                        img_array = np.clip(img_array, 0, 255)

                    img_array = img_array.astype(np.uint8)
                    augmented_images.append(img_array)

            # Reconstruct the augmented images tensor
            images = np.stack(augmented_images).reshape(images.shape)

        return images

    def _prepare_language(self, data: dict):
        """Tokenize data['language'] (or default_instruction if missing)."""
        if self._language_key is not None:
            raw_language = data[self._language_key]
            if isinstance(raw_language, list):
                raw_language = raw_language[0]

            # Language dropout
            if self.training and self.language_dropout_prob > 1e-9:
                if random.random() < self.language_dropout_prob:
                    raw_language = self.default_instruction
        else:
            raw_language = self.default_instruction

        # print(f"RAW Language: {raw_language=}")

        return raw_language

    def _prepare_state(self, data: dict):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        Return (state, state_mask, n_state_tokens).
        """
        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = data["state"]
        
        if len(state.shape) == 1:
            state = state[None, :]
        assert state.shape[0] == self.state_horizon, f"{state.shape=}, {self.state_horizon=}"

        n_state_dims = state.shape[-1]
        # print(f"state shape: {state.shape=}, should be: {np.zeros((self.state_horizon, self.max_state_dim)).shape=}, max_state_dim: {self.max_state_dim=}")
        # Instead of asserting, just take the first max_state_dim dimensions if needed
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            # Pad up to max_state_dim if smaller
            state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")

        # Create mask for real state dims
        state_mask = np.zeros_like(state).astype(bool)
        state_mask[:, :n_state_dims] = True

        # We only have 1 "proprio" token to represent the entire state
        n_state_tokens = state.shape[0]
        return state, state_mask, n_state_tokens

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]
        assert actions.shape[0] == self.action_horizon, f"{actions.shape=}, {self.action_horizon=}"

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens

    def apply_single(self, data: dict) -> dict:
        transformed_data = {}

        # 1) Prepare video and language with vlm processing.
        images = self._prepare_video(data)
        images = images.astype(np.uint8)

        images_aug = self._prepare_video(data, augmentation=True)
        images_aug = images_aug.astype(np.uint8)

        language = self._prepare_language(data)
        batch_data = {"images": images, "language": language}
        vlm_outputs = self._apply_vlm_processing(batch_data)
        batch_data_aug = {"images": images_aug, "language": language}
        vlm_outputs_aug = self._apply_vlm_processing(batch_data_aug)

        # 2) Prepare state
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        if self.training:
            # 3) Prepare actions
            transformed_data["segmentation_target"] = np.zeros((2,))
            transformed_data["segmentation_target_mask"] = np.zeros((1,))
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask

        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        for k, v in vlm_outputs_aug.items():
            transformed_data[k + "_aug"] = v

        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        if self.training:
            action_and_mask_keys = ["action", "action_mask"]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape
                for key in action_and_mask_keys
            ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"

        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension.
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        if self.backbone_model_type == "eagle":
            # Eagle processor collates the data.
            return collate(data_split_processed, self.eagle_processor)
        elif self.backbone_model_type == "qwen2_5_vl" or self.backbone_model_type == "qwen2_5_vl_7b":
            # Qwen processor collates the data.
            return qwen_collate(data_split_processed, self.qwen_processor)
        elif self.backbone_model_type == "paligemma":
            # Paligemma processor collates the data.
            return paligemma_collate(data_split_processed, self.paligemma_processor)
        else:
            raise ValueError(f"Unsupported model type: {self.backbone_model_type}. Supported: eagle, qwen2_5_vl, paligemma.")

    def apply(self, data: dict) -> dict:
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)


class GR00TTransformCL2(InvertibleModalityTransform):

    backbone_model_type: str = "eagle"

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )
    formalize_language: bool = Field(default=False, description="Formalize language if True.")
    embodiment_tag_mapping: dict[str, int] = Field(
        description="The projector index of each embodiment tag.",
        default=EMBODIMENT_TAG_MAPPING,
    )
    language_dropout_prob: float = Field(
        default=0.0,
        description="Dropout probability for language.",
    )

    # Private attributes to keep track of shapes/dimensions across apply/unapply
    _language_key: Optional[list[str]] = PrivateAttr(default=None)

    eagle_processor: ProcessorMixin = Field(default=build_eagle_processor(DEFAULT_EAGLE_PATH))
    qwen_processor: ProcessorMixin = Field(default=build_qwen_processor(DEFAULT_QWEN_PATH))
    paligemma_processor: ProcessorMixin = Field(default=build_paligemma_processor(DEFAULT_PALIGEMMA_PATH))
    # XEmbDiT arguments
    default_instruction: str = Field(default="Perform the default behavior.")
    max_state_dim: int
    max_action_dim: int
    state_horizon: int
    action_horizon: int

    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None

    # contrastive_augmentations: Optional[ContrastiveAugmentations] = PrivateAttr(default=None)

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata for the transform."""
        super().set_metadata(dataset_metadata)
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert (
            self.embodiment_tag is not None
        ), "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            if "annotation" in key:
                modality = "language"
            else:
                try:
                    modality, _ = key.split(".")
                except:  # noqa: E722
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")

        # Handle language
        if "language" in grouped_keys:
            language_keys = grouped_keys["language"]
            assert len(language_keys) == 1, f"{language_keys=}"
            self._language_key = language_keys[0]
        return is_batched, batch_size

    def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
        """
        Args:
            batch:
                video: [V, T, C, H, W]
        Returns: required input with the format `BatchFeature`
        """
        if self.backbone_model_type == "eagle":
            images = batch["images"]  # [V, T, C, H, W]
            images.shape[0]

            np_images = rearrange(images, "v t c h w -> (t v) c h w")
            text_content = []

            # handle language
            lang = batch["language"]
            if isinstance(lang, list):
                lang = lang[0]
            text_content.append({"type": "text", "text": lang})
            eagle_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
            eagle_image = [{"type": "image", "image": img} for img in eagle_images]
            eagle_conversation = [
                {
                    "role": "user",
                    "content": eagle_image + text_content,
                }
            ]

            text_list = [
                self.eagle_processor.apply_chat_template(
                    eagle_conversation, tokenize=False, add_generation_prompt=True
                )
            ]
            image_inputs, video_inputs = self.eagle_processor.process_vision_info(eagle_conversation)
            eagle_content = {
                "image_inputs": image_inputs,
                "video_inputs": video_inputs,
                "text_list": text_list,
            }
            inputs = {}
            inputs["eagle_content"] = eagle_content
            return inputs
        
        elif self.backbone_model_type == "qwen2_5_vl" or self.backbone_model_type == "qwen2_5_vl_7b":
            images = batch["images"]  # [V, T, C, H, W]
            images.shape[0]

            np_images = rearrange(images, "v t c h w -> (t v) c h w")
            text_content = []

            # handle language
            lang = batch["language"]
            if isinstance(lang, list):
                lang = lang[0]
            text_content.append({"type": "text", "text": lang})
            # print(f"[DEBUG] Processing Qwen2.5 VL inputs with {len(np_images)} images.")
            qwen_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
            qwen_image = [{"type": "image", "image": img} for img in qwen_images]
            qwen_conversation = [
                {
                    "role": "user",
                    "content": qwen_image + text_content,
                }
            ]
            text_list = [
                self.qwen_processor.apply_chat_template(
                    qwen_conversation, tokenize=False, add_generation_prompt=True
                )
            ]
            image_inputs, video_inputs = qwen_process_vision_info(qwen_conversation)
            qwen_content = {
                "image_inputs": image_inputs,
                "video_inputs": video_inputs,
                "text_list": text_list,
            }
            inputs = {}
            inputs["qwen_content"] = qwen_content
            return inputs
        elif self.backbone_model_type == "paligemma":
            images = batch["images"]  # [V, T, C, H, W]
            images.shape[0]

            np_images = rearrange(images, "v t c h w -> (t v) c h w")
            text_content = []

            # handle language
            lang = batch["language"]
            if isinstance(lang, list):
                lang = lang[0]
            paligemma_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
            text_content.append("<image>\n<image>\nanswer en " + lang)

            text_list = [
                text_content
            ][0]
            paligemma_content = {
                "image_inputs": [paligemma_images],
                "text_list": text_list,
            }
            inputs = {}
            inputs["paligemma_content"] = paligemma_content
            return inputs
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported: eagle, qwen2_5_vl, paligemma.")

    def _prepare_video(self, data: dict, augmentation: bool = False):
        """Process, stack, and pad images from data['video']."""
        ## TODO(YL, FH): check if this is correct
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )

        if augmentation:

            augmented_images = []
            # Apply augmentations to each image in the video
            for v in range(images.shape[0]):  # For each view
                for t in range(images.shape[1]):  # For each timestep
                    img = images[v, t]
                    
                    # Convert to PIL for easier augmentation
                    pil_img = Image.fromarray(np.transpose(img, (1, 2, 0)))
                    
                    if random.random() > 0.5:
                        brightness = random.uniform(0.9, 1.1)
                        contrast = random.uniform(0.9, 1.1)
                        saturation = random.uniform(0.9, 1.1)
                        pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness)
                        pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast)
                        pil_img = ImageEnhance.Color(pil_img).enhance(saturation)

                    # 2. Gaussian noise (add to numpy after all PIL ops)
                    # (Defer until after augmentations, see below)

                    # 3. Gaussian blur (rare, small)
                    if random.random() < 0.1:
                        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1))

                    # 4. Random grayscale (rare)
                    if random.random() < 0.05:
                        pil_img = pil_img.convert('L').convert('RGB')

                    # 5. Gamma (exposure)
                    if random.random() < 0.3:
                        gamma = random.uniform(0.95, 1.05)
                        img_np = np.array(pil_img).astype(np.float32) / 255.0
                        img_np = np.power(img_np, gamma)
                        img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_np)

                    # 6. Cutout / Random Erasing (optional)
                    if random.random() < 0.1:
                        width, height = pil_img.size
                        erase_w = int(width * random.uniform(0.05, 0.15))
                        erase_h = int(height * random.uniform(0.05, 0.15))
                        x0 = random.randint(0, width - erase_w)
                        y0 = random.randint(0, height - erase_h)
                        pil_img.paste((0, 0, 0), (x0, y0, x0 + erase_w, y0 + erase_h))

                    # Convert back to numpy
                    img_array = np.transpose(np.array(pil_img), (2, 0, 1)).astype(np.float32)

                    # 2. Gaussian noise (now add to numpy array)
                    if random.random() < 0.3:
                        noise = np.random.normal(0, 4, size=img_array.shape)  # mean 0, std 4
                        img_array = img_array + noise
                        img_array = np.clip(img_array, 0, 255)

                    img_array = img_array.astype(np.uint8)
                    augmented_images.append(img_array)

            # Reconstruct the augmented images tensor
            images = np.stack(augmented_images).reshape(images.shape)

        return images

    def _prepare_language(self, data: dict):
        """Tokenize data['language'] (or default_instruction if missing)."""
        if self._language_key is not None:
            raw_language = data[self._language_key]
            if isinstance(raw_language, list):
                raw_language = raw_language[0]

            # Language dropout
            if self.training and self.language_dropout_prob > 1e-9:
                if random.random() < self.language_dropout_prob:
                    raw_language = self.default_instruction
        else:
            raw_language = self.default_instruction

        # print(f"RAW Language: {raw_language=}")

        return raw_language

    def _prepare_state(self, data: dict):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        Return (state, state_mask, n_state_tokens).
        """
        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = data["state"]
        
        if len(state.shape) == 1:
            state = state[None, :]
        assert state.shape[0] == self.state_horizon, f"{state.shape=}, {self.state_horizon=}"

        n_state_dims = state.shape[-1]
        # print(f"state shape: {state.shape=}, should be: {np.zeros((self.state_horizon, self.max_state_dim)).shape=}, max_state_dim: {self.max_state_dim=}")
        # Instead of asserting, just take the first max_state_dim dimensions if needed
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            # Pad up to max_state_dim if smaller
            state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")

        # Create mask for real state dims
        state_mask = np.zeros_like(state).astype(bool)
        state_mask[:, :n_state_dims] = True

        # We only have 1 "proprio" token to represent the entire state
        n_state_tokens = state.shape[0]
        return state, state_mask, n_state_tokens

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]
        assert actions.shape[0] == self.action_horizon, f"{actions.shape=}, {self.action_horizon=}"

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens

    def apply_single(self, data: dict) -> dict:
        transformed_data = {}
        # print(f"[DEBUG] Applying GR00TTransformCL2 to data with keys: {data.keys()}")
        # for key, value in data.items():
        #     print(f"[DEBUG] {key}: {value.shape if hasattr(value, 'shape') else value}")
        # assert False

        # 1) Prepare video and language with vlm processing.
        prepared_images = self._prepare_video(data)
        images = prepared_images[:,0:1, ...]
        images = images.astype(np.uint8)

        images_aug = prepared_images[:, 1:2, ...]
        # images_aug = prepared_images[random.randint(1, 2), ...][None]
        images_aug = images_aug.astype(np.uint8)

        language = self._prepare_language(data)
        batch_data = {"images": images, "language": language}
        vlm_outputs = self._apply_vlm_processing(batch_data)

        batch_data_aug = {"images": images_aug, "language": language}
        vlm_outputs_aug = self._apply_vlm_processing(batch_data_aug)

        # 2) Prepare state
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        if self.training:
            # 3) Prepare actions
            transformed_data["segmentation_target"] = np.zeros((2,))
            transformed_data["segmentation_target_mask"] = np.zeros((1,))
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask

        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        for k, v in vlm_outputs_aug.items():
            assert k in transformed_data, f"Key {k} doesn't exist in transformed_data."
            transformed_data[k + "_aug"] = v

        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        if self.training:
            action_and_mask_keys = ["action", "action_mask"]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape
                for key in action_and_mask_keys
            ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"

        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension.
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        if self.backbone_model_type == "eagle":
            # Eagle processor collates the data.
            return collate(data_split_processed, self.eagle_processor)
        elif self.backbone_model_type == "qwen2_5_vl" or self.backbone_model_type == "qwen2_5_vl_7b":
            # Qwen processor collates the data.
            return qwen_collate(data_split_processed, self.qwen_processor)
        elif self.backbone_model_type == "paligemma":
            # Paligemma processor collates the data.
            return paligemma_collate(data_split_processed, self.paligemma_processor)
        else:
            raise ValueError(f"Unsupported model type: {self.backbone_model_type}. Supported: eagle, qwen2_5_vl, paligemma.")

    def apply(self, data: dict) -> dict:
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)
