import dataclasses

import einops
import numpy as np
import torch

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RobocasaInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    offline_sampling: bool = False

    def __call__(self, data: dict) -> dict:
        # state : torch.Size([53]) -> [16] = (D)
        state_base_pos = data["state"][:3]
        state_base_rot = data["state"][3:7]  # quaternion
        state_end_eff_pos_rel = data["state"][10:13]
        state_end_eff_rot_rel = data["state"][17:21]  # quaternion
        state_gripper_qpos = data["state"][21:23]

        state = np.concatenate([state_base_pos, state_base_rot, state_end_eff_pos_rel, state_end_eff_rot_rel, state_gripper_qpos])
         # since the robocasa action_dim = 12, which is < state_dim = 16, so pad is skipped.
        state = transforms.pad_to_dim(state, self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        left_image = _parse_image(data["video.left_view"])
        right_image = _parse_image(data["video.right_view"])
        wrist_image = _parse_image(data["video.wrist_view"])

        match self.model_type:
            case _model.ModelType.PI0:
                #NOTE : ordered from left arm to right arm
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (left_image, right_image, wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI05:
                # NOTE : ordered from left arm to right arm
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (left_image, right_image, wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (left_image, right_image, wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "annotation.human.action.task_description" in data:
            inputs["prompt"] = data["annotation.human.action.task_description"]

        if "reward" in data:
            inputs["reward"] = data["reward"]
        if "done" in data:
            inputs["done"] = data["done"]

        if "action" in data: #SFT Data
            # action : torch.Size([16, 12]) = (H, D)
            action_base_motion = data["action"][:,:4]
            action_control_mode = data["action"][:,4:5] #binary
            action_end_eff_pos = data["action"][:,5:8]
            action_end_eff_rot = data["action"][:,8:11] #axis angle
            action_gripper_close = data["action"][:,11:12] #binary

            inputs["actions"] = np.concatenate([action_base_motion, action_control_mode, action_end_eff_pos, action_end_eff_rot, action_gripper_close], axis=1)

        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaRLInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    offline_sampling: bool = False

    def __call__(self, data: dict) -> dict:
        assert self.model_type == _model.ModelType.PI0_DEAS, "PI0_DEAS model only supports RL inputs"
        # for key, value in data.items():
        #     if isinstance(value, torch.Tensor):
        #         print(f"key: {key}, type: {type(value)}, shape: {value.shape}")
        #     else:
        #         print(f"key: {key}, type: {type(value)}, value: {value}")
        # state : torch.Size([batch_size, 2, 53]) -> split into (batch_size, 53) each
        # Split data["state"] with shape (batch_size, 2, 53) into state and next_state
        state = data["state"][0]  # shape: (53)
        next_state = data["state"][1]  # shape: (53)

        state_base_pos = state[:3]
        state_base_rot = state[3:7]  # quaternion
        state_end_eff_pos_rel = state[10:13]
        state_end_eff_rot_rel = state[17:21]  # quaternion
        state_gripper_qpos = state[21:23]

        state = np.concatenate(
            [state_base_pos, state_base_rot, state_end_eff_pos_rel, state_end_eff_rot_rel, state_gripper_qpos]
        )
        # since the robocasa action_dim = 12, which is < state_dim = 16, so pad is skipped.
        state = transforms.pad_to_dim(state, self.action_dim)

        next_state_base_pos = next_state[:3]
        next_state_base_rot = next_state[3:7]  # quaternion
        next_state_end_eff_pos_rel = next_state[10:13]
        next_state_end_eff_rot_rel = next_state[17:21]  # quaternion
        next_state_gripper_qpos = next_state[21:23]

        next_state = np.concatenate(
            [
                next_state_base_pos,
                next_state_base_rot,
                next_state_end_eff_pos_rel,
                next_state_end_eff_rot_rel,
                next_state_gripper_qpos,
            ],
            axis=0,
        )
        next_state = transforms.pad_to_dim(next_state, self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference

        left_image = _parse_image(data["video.left_view"][0])
        right_image = _parse_image(data["video.right_view"][0])
        wrist_image = _parse_image(data["video.wrist_view"][0])

        next_left_image = _parse_image(data["video.left_view"][1])
        next_right_image = _parse_image(data["video.right_view"][1])
        next_wrist_image = _parse_image(data["video.wrist_view"][1])

        # NOTE : ordered from left arm to right arm
        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (left_image, right_image, wrist_image)
        image_masks = (np.True_, np.True_, np.True_)

        next_images = (next_left_image, next_right_image, next_wrist_image)

        inputs = {
            "state": state,
            "next_state": next_state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
            "next_image": dict(zip(names, next_images, strict=True)),
            "next_image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "annotation.human.action.task_description" in data:
            inputs["prompt"] = data["annotation.human.action.task_description"]

        if "reward" in data:
            inputs["reward"] = data["reward"]
            if torch.sum(inputs["reward"]) > 0:
                reward_range = 15
                idx = torch.where(inputs["reward"] == 1)[0]
                if len(idx) > 0 and idx[0] - reward_range >= 0:
                    start = idx[0] - reward_range
                    inputs["reward"][start : idx[0]] = 1
            inputs["reward_pad"] = data["reward_pad"]
        if "done" in data:
            inputs["done"] = data["done"]

        if "action" in data:  # SFT Data
            # action : torch.Size([16, 12]) = (H, D)
            action_base_motion = data["action"][:, :4]
            action_control_mode = data["action"][:, 4:5]  # binary
            action_end_eff_pos = data["action"][:, 5:8]
            action_end_eff_rot = data["action"][:, 8:11]  # axis angle
            action_gripper_close = data["action"][:, 11:12]  # binary

            inputs["actions"] = np.concatenate(
                [action_base_motion, action_control_mode, action_end_eff_pos, action_end_eff_rot, action_gripper_close],
                axis=1,
            )

        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Robocasa, return 12 actions.
        return {**data, "actions": np.asarray(data["actions"][..., :12])}
