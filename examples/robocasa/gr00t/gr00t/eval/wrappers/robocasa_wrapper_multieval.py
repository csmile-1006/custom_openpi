from pathlib import Path
from functools import partial
from typing import Optional

import gymnasium as gym
import numpy as np
import robocasa  # noqa
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import GymWrapper
from gymnasium.wrappers import TimeLimit
from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS, MULTI_STAGE_TASK_DATASETS

from gr00t.eval.wrappers.multistep_wrapper_multieval import MultiStepWrapper
from gr00t.eval.wrappers.data_collection_wrapper_dev import DataCollectionWrapper
from gr00t.eval.wrappers.record_video_multieval import RecordVideo


def get_env_horizon(env_name):
    if env_name in SINGLE_STAGE_TASK_DATASETS:
        ds_config = SINGLE_STAGE_TASK_DATASETS[env_name]
    elif env_name in MULTI_STAGE_TASK_DATASETS:
        ds_config = MULTI_STAGE_TASK_DATASETS[env_name]
    else:
        raise ValueError(f"Environment {env_name} not found in dataset registry")
    return ds_config["horizon"]


def load_robocasa_gym_env(env_name, n_envs=1, **kwargs):        
    episodes_per_env = kwargs.pop('n_episodes', 50) // n_envs
    skip_episodes_per_env = kwargs.pop('skip_episodes_per_env', [0] * n_envs)
    env_fns = [partial(create_robocasa_gym_env, env_name=env_name, env_idx = i, **kwargs) for i in range(n_envs)]
    
    if n_envs == 1:
        return  gym.vector.SyncVectorEnv(env_fns)
    else: # multi-env settings
        return gym.vector.CustomAsyncVectorEnv(
            env_fns,
            episodes_per_env=episodes_per_env,
            skip_episodes_per_env=skip_episodes_per_env,
            shared_memory=False,
            copy=False,
            context="spawn",
            observation_mode="same",
        )
        

def create_robocasa_gym_env(
    env_name,
    env_idx=0,
    seed=None,
    # robosuite-related configs
    robots="PandaOmron",
    camera_names=[
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ],
    camera_widths=256,
    camera_heights=256,
    render_onscreen=False,
    # robocasa-related configs
    obj_instance_split=None,
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=None,
    layout_ids=None,
    style_ids=None,
    # data collection configs
    collect_data: bool = False,
    collect_directory: Path = None,
    collect_freq: int = 1,
    flush_freq: int = 100,
    # video configs
    video_path: Optional[str] = None,
    # multi-step configs
    action_horizon: int = 16,
    video_delta_indices: np.ndarray = np.array([0]),
    state_delta_indices: np.ndarray = np.array([0]),
):
    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots if isinstance(robots, str) else robots[0],
    )

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_config,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=render_onscreen,
        has_offscreen_renderer=(not render_onscreen),
        ignore_done=False,
        use_object_obs=True,
        use_camera_obs=(not render_onscreen),
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        layout_ids=layout_ids,
        style_ids=style_ids,
        translucent_robot=False,
    )

    env = robosuite.make(**env_kwargs)

    if collect_data:
        if collect_directory is not None and collect_directory.exists():
            collect_directory.mkdir(parents=True, exist_ok=True)
        env = DataCollectionWrapper(env, collect_directory, collect_freq=collect_freq, flush_freq=flush_freq)

    env = GymWrapper(
        env,
        flatten_obs=False,
        keys=[
            "robot0_base_pos",
            "robot0_base_quat",
            "robot0_eef_pos",
            "robot0_base_to_eef_pos",
            "robot0_eef_quat",
            "robot0_base_to_eef_quat",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel",
            "robot0_joint_pos",
            "robot0_joint_pos_cos",
            "robot0_joint_pos_sin",
            "robot0_joint_vel",
            "robot0_agentview_left_image",
            "robot0_agentview_right_image",
            "robot0_eye_in_hand_image",
        ],
    )

    env = RoboCasaWrapper(env, env_idx=env_idx)
    record_video = video_path is not None
    if record_video:
        video_base_path = Path(video_path)
        video_base_path.mkdir(parents=True, exist_ok=True)
        episode_trigger = lambda t: t % 1 == 0  # 모든 episode 기록
        env = RecordVideo(
            env,
            video_base_path,
            disable_logger=True,
            episode_trigger=episode_trigger,
            fps=20,
            name_prefix=f"{env_name}-env{env_idx}", 
        )
    
    env = TimeLimit(env, max_episode_steps=get_env_horizon(env_name))
    env = MultiStepWrapper(
        env,
        video_delta_indices=video_delta_indices,
        state_delta_indices=state_delta_indices,
        n_action_steps=action_horizon,
    )

    return env


class RoboCasaWrapper(gym.Wrapper):
    def __init__(self, env, env_idx=0):
        super().__init__(env)
        self.env_idx = env_idx
                
        self._robocasa_keys_to_gr00t_keys = {
            "robot0_base_pos": "state.base_position",
            "robot0_base_quat": "state.base_rotation",
            "robot0_eef_pos": "state.end_effector_position_absolute",
            "robot0_base_to_eef_pos": "state.end_effector_position_relative",
            "robot0_eef_quat": "state.end_effector_rotation_absolute",
            "robot0_base_to_eef_quat": "state.end_effector_rotation_relative",
            "robot0_gripper_qpos": "state.gripper_qpos",
            "robot0_gripper_qvel": "state.gripper_qvel",
            "robot0_joint_pos": "state.joint_position",
            "robot0_joint_pos_cos": "state.joint_position_cos",
            "robot0_joint_pos_sin": "state.joint_position_sin",
            "robot0_joint_vel": "state.joint_velocity",
            "robot0_agentview_left_image": "video.left_view",
            "robot0_agentview_right_image": "video.right_view",
            "robot0_eye_in_hand_image": "video.wrist_view",
        }

        self._observation_space = self._convert_observation_space()
        self._action_space = self._convert_action_space()


    def _convert_action_space(self):
        original_action_space = self.env.action_space
        # Split original action space into parts
        low = original_action_space.low
        high = original_action_space.high
        dtype = original_action_space.dtype

        new_action_space = gym.spaces.Dict(
            {
                "action.end_effector_position": gym.spaces.Box(low=low[0:3], high=high[0:3], dtype=dtype),
                "action.end_effector_rotation": gym.spaces.Box(low=low[3:6], high=high[3:6], dtype=dtype),
                "action.gripper_close": gym.spaces.Box(low=low[6:7], high=high[6:7], dtype=np.int64),
                "action.base_motion": gym.spaces.Box(low=low[7:11], high=high[7:11], dtype=dtype),
                "action.control_mode": gym.spaces.Box(low=low[11:12], high=high[11:12], dtype=np.int64),
            }
        )
        self.action_space_keys = [
            "action.end_effector_position",
            "action.end_effector_rotation",
            "action.gripper_close",
            "action.base_motion",
            "action.control_mode",
        ]
        return new_action_space

    def _convert_observation_space(self):
        original_observation_space = self.env.observation_space
        new_observation_space = {}
        for key, value in original_observation_space.items():
            if key in self._robocasa_keys_to_gr00t_keys:
                new_observation_space[self._robocasa_keys_to_gr00t_keys[key]] = value
        new_observation_space["annotation.human.action.task_description"] = gym.spaces.Sequence(
            gym.spaces.Text(max_length=1000)
        )
        return gym.spaces.Dict(new_observation_space)

    @property
    def language_instruction(self):
        return self.env.get_ep_meta()["lang"]

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info["success"] = self.is_success()["task"]
        return self.convert_observation(obs), info

    def render(self, mode="rgb_array"):
        return self.env.unwrapped.sim.render(camera_name="robot0_agentview_center", height=512, width=512)[::-1]

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        succ = self.env._check_success()
        if isinstance(succ, dict):
            assert "task" in succ
            return succ
        return {"task": succ}
    
    def convert_observation(self, obs):
        new_obs = {}
        for key, value in obs.items():
            if key in self._robocasa_keys_to_gr00t_keys:
                new_obs[self._robocasa_keys_to_gr00t_keys[key]] = value
        new_obs["annotation.human.action.task_description"] = [self.language_instruction]
        for key in new_obs.keys():
            if key.startswith("video"):
                #NOTE : observation flip is already adopted!
                new_obs[key] = np.flip(new_obs[key], axis=[0])
        return new_obs

    def convert_action(self, action):
        # binarize the gripper close and control mode action
        for key in ["action.gripper_close", "action.control_mode"]:
            action[key] = np.where(action[key] > 0, 1, -1)
        elems = [
            action[key] for key in self.action_space_keys
        ]  # this is to strictly follow the order of the action space
        return np.concatenate(elems, axis=-1)

    def step(self, action):
        action = self.convert_action(action)
        obs, reward, terminated, truncated, info = super().step(action)
        info["success"] = self.is_success()["task"]
        terminated = terminated or info["success"]
        return self.convert_observation(obs), reward, terminated, truncated, info

    def close(self):
        return self.env.close()


if __name__ == "__main__":
    _env_name = "CloseDoubleDoor"

    # <MultiStepWrapper<TimeLimit<RoboCasaWrapper<GymWrapper instance>>>>
    env = create_robocasa_gym_env(_env_name)

    obs, _ = env.reset()

    for i in range(10):
        print(f"Step {i}")
        
        #개별 action_key : (H,D)
        action = env.action_space.sample()
        
        import pdb; pdb.set_trace()
        
        obs, reward, terminated, truncated, info = env.step(action)
