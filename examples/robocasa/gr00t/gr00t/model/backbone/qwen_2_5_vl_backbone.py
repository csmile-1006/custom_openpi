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

import torch
from torch import nn
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.feature_extraction_utils import BatchFeature

DEFAULT_QWEN_PATH="Qwen/Qwen2.5-VL-3B-Instruct"


class Qwen2_5VLBackbone(nn.Module):

    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        qwen_path: str = DEFAULT_QWEN_PATH,
        project_to_dim: int = 1536,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: False)
            tune_visual: whether to tune the visual model (default: False)
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"
        logging.warning(f"Loading Qwen2.5-VL model from {qwen_path}")
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_path,
            trust_remote_code=True,
        )

        if project_to_dim is not None:
            hidden_size = self.qwen_model.model.config.hidden_size
            logging.warning(f"[DEBUG] Qwen2.5-VL hidden size: {hidden_size}")
            self.qwen_linear = torch.nn.Linear(hidden_size, project_to_dim) # hidden size same to 2048
        else:
            logging.warning(f"[DEBUG] No projection to dim, using identity layer")
            self.qwen_linear = torch.nn.Identity()

        # needed since we don't use these layers. Also saves compute
        logging.warning(f"[DEBUG] Select layer: {select_layer} out of {len(self.qwen_model.model.layers)}")
        while len(self.qwen_model.model.layers) > select_layer:
            self.qwen_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual)

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.qwen_model.model.requires_grad_(False)
            self.qwen_model.lm_head.requires_grad_(False)
        if not tune_visual:
            self.qwen_model.visual.requires_grad_(False)
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.qwen_model and not self.tune_llm:
                self.qwen_model.eval()
            if self.qwen_model.visual and not self.tune_visual:
                self.qwen_model.visual.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward_qwen(self, vl_input: BatchFeature) -> BatchFeature:
        qwen_prefix = "qwen_"
        qwen_input = {
            k.removeprefix(qwen_prefix): v
            for k, v in vl_input.items()
            if k.startswith(qwen_prefix)
        }

        qwen_output = self.qwen_model(**qwen_input, output_hidden_states=True, return_dict=True)
        qwen_features = qwen_output.hidden_states[self.select_layer]

        qwen_features = self.qwen_linear(qwen_features)
        return qwen_features, qwen_input["attention_mask"]

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()


        qwen_embeds, qwen_mask = self.forward_qwen(vl_input)
        return BatchFeature(
            data={"backbone_features": qwen_embeds, "backbone_attention_mask": qwen_mask}
        )  # [B, T2, hidden_size]
