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

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

import gr00t

DEFAULT_EAGLE_PATH = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)


class EagleBackboneCL(nn.Module):

    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str | None = None,
        project_to_dim: int = 1536,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)

        self.dtype = self.eagle_model.dtype
        if project_to_dim is not None:
            # self.eagle_linear = torch.nn.Sequential(
            #     torch.nn.LayerNorm(2048, dtype=self.dtype),
            #     torch.nn.Linear(2048, project_to_dim, dtype=self.dtype),
            #     torch.nn.GELU(),
            #     
            #     torch.nn.Linear(project_to_dim, project_to_dim, dtype=self.dtype),
            # )
            self.norm = torch.nn.LayerNorm(project_to_dim, elementwise_affine=False)
            self.linear1 = torch.nn.Linear(2048, project_to_dim)
            self.relu = torch.nn.GELU()
            self.linear2 = torch.nn.Linear(project_to_dim, project_to_dim)

            # self.eagle_linear.apply(self._init_weights)
            print(f"[DEBUG] eagle_backbone_cl.py: Projecting eagle features to {project_to_dim} dimensions")

        else:
            self.eagle_linear = torch.nn.Identity()

        # needed since we don't use these layers. Also saves compute
        print(f"[DEBUG] eagle_backbone_cl.py: Removing layers after {select_layer} for eagle model")
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)
            
        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Initialize with a normal distribution with a small std dev
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # Initialize bias to zero
                torch.nn.init.zeros_(module.bias)

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.eagle_model.language_model.requires_grad_(False)
        if not tune_visual:
            self.eagle_model.vision_model.requires_grad_(False)
            self.eagle_model.mlp1.requires_grad_(False)
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
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if self.eagle_model.vision_model and not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]

        eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]
            # --- Start Debugging Block ---
        
        x = self.linear1(eagle_features)
        x = self.relu(x)
        x = self.norm(x.float())
        x = x.to(self.dtype)
        eagle_features_projected = self.linear2(x)
        # --- End Debugging Block ---

        # print("eagle_features before linear:", torch.isnan(eagle_features).any(), eagle_features.mean().item(), eagle_features.std().item())
        # print("eagle_features after linear:", torch.isnan(eagle_features_projected).any(), eagle_features_projected.mean().item(), eagle_features_projected.std().item())
        eagle_features = eagle_features_projected
        return eagle_features, eagle_input["attention_mask"]

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        eagle_embeds, eagle_mask = self.forward_eagle(vl_input)
        return BatchFeature(
            data={"backbone_features": eagle_embeds, "backbone_attention_mask": eagle_mask}
        )  # [B, T2, hidden_size]
