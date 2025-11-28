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
        print(f"data['state']: {data['state']}")
        raise Exception("stop")
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

        def extract_state_fields(state):
            return np.concatenate(
                [
                    state[:3],  # base_pos
                    state[3:7],  # base_rot (quaternion)
                    state[10:13],  # end_eff_pos_rel
                    state[17:21],  # end_eff_rot_rel (quaternion)
                    state[21:23],  # gripper_qpos
                ],
                axis=0,
            )

        def build_images(data, idx=None):
            get = lambda view: _parse_image(  # noqa: E731
                data[f"video.{view}_view"][idx] if idx is not None else data[f"video.{view}_view"]
            )
            return tuple(get(view) for view in ("left", "right", "wrist"))

        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        image_masks = tuple([np.True_] * 3)

        # Process state and images
        is_next = data["state"].shape[0] == 2
        if is_next:
            # Current and next state
            state = extract_state_fields(data["state"][0])
            next_state = transforms.pad_to_dim(extract_state_fields(data["state"][1]), self.action_dim)
            state = transforms.pad_to_dim(state, self.action_dim)
            images = build_images(data, idx=0)
            next_images = build_images(data, idx=1)
        else:
            state = extract_state_fields(data["state"])
            state = transforms.pad_to_dim(state, self.action_dim)
            images = build_images(data)
            next_state = None
            next_images = None

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }
        if is_next:
            inputs.update(
                {
                    "next_state": next_state,
                    "next_image": dict(zip(names, next_images, strict=True)),
                    "next_image_mask": dict(zip(names, image_masks, strict=True)),
                }
            )

        # Optional fields
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
            inputs["reward_pad"] = data.get("reward_pad")  # Use get for safety

        if "done" in data:
            inputs["done"] = data["done"]

        if "action" in data:
            action = data["action"]
            actions = np.concatenate(
                [
                    action[:, :4],  # base_motion
                    action[:, 4:5],  # control_mode
                    action[:, 5:8],  # end_eff_pos
                    action[:, 8:11],  # end_eff_rot
                    action[:, 11:12],  # gripper_close
                ],
                axis=1,
            )
            inputs["actions"] = actions

        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Robocasa, return 12 actions.
        return {**data, "actions": np.asarray(data["actions"][..., :12])}
