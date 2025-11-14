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


from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from .backbone import EagleBackbone, Qwen2_5VLBackbone, PaligemmaBackbone, EagleBackboneCL

from torch import nn
import torch.nn.functional as F
import math

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
@dataclass
class GR00T_N1_5_Config(PretrainedConfig):
    model_type = "gr00t_n1_5"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    backbone_model_type: str = field(default="eagle", metadata={
        "help": "Type of the backbone model. Supported types are 'eagle', 'qwen2_5_vl', and 'paligemma'."
    })  

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class GR00T_N1_5(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1_5_Config,
        local_model_path: str = None,
        **kwargs
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)
        super().__init__(config)
        self.local_model_path = local_model_path
        backbone_model_type = getattr(config, "backbone_model_type", "eagle")
        print(f"[DEBUG] groot_n1_cl.py: Backbone model type: {backbone_model_type}")

        self.loss_type = kwargs.pop("loss_type", None)
        self.dispersive_tau = kwargs.pop("dispersive_tau", 1.0)

        self.contrastive_lambda = kwargs.pop("contrastive_lambda", 0.1)

        self.projector = kwargs.pop("projector", True)

        project_to_dim = kwargs.pop("project_to_dim", None)
        if project_to_dim is not None:
            print(f"[DEBUG] groot_n1_cl.py: Projecting backbone features to {project_to_dim} dimensions")
            config.backbone_cfg["project_to_dim"] = project_to_dim

        if backbone_model_type == "eagle":
            self.backbone = EagleBackboneCL(**config.backbone_cfg)
        elif backbone_model_type == "qwen2_5_vl":
            self.backbone = Qwen2_5VLBackbone(**config.backbone_cfg)
        elif backbone_model_type == "paligemma":
            self.backbone = PaligemmaBackbone(**config.backbone_cfg)
        else:
            raise ValueError(
                f"Unsupported backbone model type: {config.backbone_cfg.get('model_type', 'unknown')}. "
                "Supported types are 'eagle', 'qwen2_5_vl', and 'paligemma'."
            )
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

        self.backbone_model_type = backbone_model_type


        ## Contrastive Projector
        hidden_dim = config.backbone_cfg["project_to_dim"]
        assert hidden_dim is not None, "hidden_dim must be specified in the backbone config for contrastive projector"
        proj_dim = 128
        dtype = self.backbone.dtype
        if self.projector:
            self.contrastive_projector = nn.Sequential(
                nn.Linear(hidden_dim, proj_dim, dtype=dtype),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim, dtype=dtype),
            )
            self.contrastive_projector.apply(self._init_weights)
        else:
            self.contrastive_projector = nn.Identity()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Initialize with a normal distribution with a small std dev
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # Initialize bias to zero
                torch.nn.init.zeros_(module.bias)

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def _nt_xnet_loss(self, queries, keys, temperature=0.1):
        """
        Computes the NT-Xent loss between queries and keys.
        Args:
            queries: Tensor of shape (batch_size, feature_dim)
            keys: Tensor of shape (batch_size, feature_dim)
            temperature: Temperature parameter for scaling
        Returns:
            loss: Scalar tensor representing the NT-Xent loss
        """
        # Normalize queries and keys
        queries = nn.functional.normalize(queries, dim=-1)
        keys = nn.functional.normalize(keys, dim=-1)

        # Compute cosine similarity
        similarity_matrix = torch.matmul(queries, keys.T) / temperature

        # Create labels for positive pairs
        batch_size = queries.size(0)
        labels = torch.arange(batch_size, device=queries.device)

        # Compute NT-Xent loss
        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)

        return loss
        
    def _dispersive_infonce_loss(self, queries, keys, tau=1.0):
        """
        Dispersive InfoNCE loss that encourages dispersion between query-key pairs.
        
        Based on Equation (6) from "Diffuse and Disperse: Image Generation with Representation Regularization"
        L_Disp = log E_{i,j} [exp(-D(z_i, z_j) / τ)]
        
        Args:
            queries: Tensor of shape [batch_size, feature_dim]
            keys: Tensor of shape [batch_size, feature_dim] 
            tau: Temperature parameter (default: 1.0)
            
        Returns:
            Dispersive loss scalar
        """
        batch_size = queries.size(0)
        queries = F.normalize(queries, p=2, dim=1)
        keys = F.normalize(keys, p=2, dim=1)
        
        # Compute pairwise squared L2 distances between queries and keys
        # queries: [B, D], keys: [B, D]
        # Expand to [B, 1, D] and [1, B, D] for broadcasting
        queries_expanded = queries.unsqueeze(1)  # [B, 1, D]
        keys_expanded = keys.unsqueeze(0)        # [1, B, D]
        
        # Compute squared L2 distance: ||q_i - k_j||^2
        distances = torch.sum((queries_expanded - keys_expanded) ** 2, dim=2)  # [B, B]

        diagonal_distances = torch.diag(distances)
        
        # Apply dispersive loss: log E_{i,j} [exp(-D(q_i, k_j) / τ)]
        # This is equivalent to: log(mean(exp(-distances / tau)))
        exp_neg_distances = torch.exp(-diagonal_distances / tau)
        dispersive_loss = torch.log(torch.mean(exp_neg_distances))
        
        return dispersive_loss

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs) #shape?

        B = backbone_outputs[BACKBONE_FEATURE_KEY].shape[0]
        query_backbone_outputs = BatchFeature(
            data={"backbone_features": backbone_outputs[BACKBONE_FEATURE_KEY][:B//2, ...], "backbone_attention_mask": backbone_outputs['backbone_attention_mask'][:B//2, ...]}
        )
        key_backbone_outputs = BatchFeature(
            data={"backbone_features": backbone_outputs[BACKBONE_FEATURE_KEY][B//2:, ...], "backbone_attention_mask": backbone_outputs['backbone_attention_mask'][B//2:, ...]}
        )

        def _contrastive_project_fn(x):
            if len(x.shape) > 2:
                # If we have multiple time steps, average across them
                x = torch.mean(x, dim=1) # Average pooling
            return self.contrastive_projector(x)
        queries, keys = map(
            _contrastive_project_fn,
            (query_backbone_outputs[BACKBONE_FEATURE_KEY], key_backbone_outputs[BACKBONE_FEATURE_KEY]),
        )
        if self.loss_type == "contrastive":
            contrastive_loss = self._nt_xnet_loss(queries, keys, temperature=0.1)

        elif self.loss_type == "dispersive":
            contrastive_loss = self._dispersive_infonce_loss(queries, keys, tau=self.dispersive_tau)
        
        else:
            contrastive_loss = torch.tensor(0.0, device=queries.device, dtype=queries.dtype)

        action_head_outputs = self.action_head(query_backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, query_backbone_outputs, is_training=True)

        action_head_outputs["flow_loss"] = action_head_outputs["loss"]
        action_head_outputs["contrastive_loss"] = contrastive_loss
        
        action_head_outputs["loss"] = (
            action_head_outputs["flow_loss"]
            + self.contrastive_lambda * action_head_outputs["contrastive_loss"]
        )
        # print(f"[DEBUG]: flow_loss: {action_head_outputs['flow_loss'].item()}, contrastive_loss: {action_head_outputs['contrastive_loss'].item()}")
        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)
        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, load_action_head: bool=True, **kwargs):
        tune_visual = kwargs.pop("tune_visual", False)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        pretrained_model = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, **kwargs
        )

        if not load_action_head:
            print("TY: Initializing action head from scratch. Only loading backbone.")
            action_head_cfg = FlowmatchingActionHeadConfig(**pretrained_model.config.action_head_cfg)
            pretrained_model.action_head = FlowmatchingActionHead(action_head_cfg)


        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model

    @classmethod
    def from_config(cls, config, **kwargs):

        tune_visual = kwargs.pop("tune_visual", False)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)

        print(f"Loading pretrained dual brain from config: {config}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")
        model = super()._from_config(config, **kwargs)
        model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return model


# register
AutoConfig.register("gr00t_n1_5", GR00T_N1_5_Config)
AutoModel.register(GR00T_N1_5_Config, GR00T_N1_5)
