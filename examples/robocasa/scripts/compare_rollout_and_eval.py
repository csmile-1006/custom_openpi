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
import cv2
import json
from collections import defaultdict
from collections import deque
from pathlib import Path
import importlib
from tqdm import tqdm, trange
import numpy as np
import dataclasses
from typing import List, Optional, Dict, Any
import tyro
import logging
import csv

from robosuite.controllers import load_composite_controller_config

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gr00t.eval.robocasa_simulation import (
    MultiStepConfig,
    SimulationConfig,
    SimulationInferenceClient,
    VideoConfig,
)
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.record_video import RecordVideo
from gr00t.eval.wrappers.robocasa_wrapper import RoboCasaWrapper, load_robocasa_gym_env

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

PALIGEMMA_ACTION_END_TOKEN = 235371


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def flatten(d, parent_key="", sep="."):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def load_tsv_data(tsv_path: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    TSV 파일에서 데이터를 로드하여 episode별로 정리 (dtype 복원)
    Returns:
        Dict[episode_index, List[step_data]]
    """
    episode_data = defaultdict(list)
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            episode_idx = int(row['episode_index'])
            step = int(row['step'])
            
            # JSON 형태로 저장된 데이터들을 파싱 (dtype 복원)
            scores_list_data = json.loads(row['scores_list'])
            action_list_path = row['action_list']  # .npy 파일 경로
            best_index = int(row['best_index'])
            best_score = float(row['best_score'])
            best_action_path = row['best_action']  # .npy 파일 경로
            
            # scores_list 복원 (numpy array)
            if isinstance(scores_list_data, dict) and 'data' in scores_list_data:
                scores_list = np.array(scores_list_data['data'], dtype=scores_list_data['dtype'])
            else:
                scores_list = np.array(scores_list_data) if isinstance(scores_list_data, list) else scores_list_data
            
            # action_list를 .npy 파일에서 로드
            if action_list_path.endswith('.npy'):
                action_list = np.load(action_list_path)
            else:
                # 기존 JSON 방식 (호환성 유지)
                action_list = json.loads(action_list_path)
            
            # best_action을 .npy 파일에서 로드
            if best_action_path.endswith('.npy'):
                best_action = np.load(best_action_path)
            else:
                # 기존 JSON 방식 (호환성 유지)
                best_action_data = json.loads(best_action_path)
                if isinstance(best_action_data, dict) and 'data' in best_action_data:
                    best_action = np.array(best_action_data['data'], dtype=best_action_data['dtype'])
                else:
                    best_action = np.array(best_action_data)
            
            # action_list가 (N, B, 16, 12) 형태인 경우 B 차원을 squeeze
            if len(action_list.shape) == 4 and action_list.shape[1] == 1:
                action_list = np.squeeze(action_list, axis=1)  # (N, 16, 12)
            
            # 로드한 데이터의 shape 확인
            assert len(action_list.shape) == 3, f"action_list shape {action_list.shape}가 예상과 다릅니다. 예상: (N, 16, 12)"
            assert action_list.shape[1:] == (16, 12), f"action_list shape {action_list.shape}의 마지막 두 차원이 예상과 다릅니다. 예상: (N, 16, 12)"
            assert len(scores_list.shape) == 1, f"scores_list shape {scores_list.shape}가 예상과 다릅니다. 예상: (N,)"
            assert len(best_action.shape) == 2, f"best_action shape {best_action.shape}가 예상과 다릅니다. 예상: (16, 12)"
            assert best_action.shape == (16, 12), f"best_action shape {best_action.shape}가 예상과 다릅니다. 예상: (16, 12)"
            assert action_list.shape[0] == len(scores_list), f"action_list의 candidate 수 {action_list.shape[0]}와 scores_list 길이 {len(scores_list)}가 일치하지 않습니다"
            
            step_data = {
                'step': step,
                'scores_list': scores_list,
                'action_list': action_list,
                'best_index': best_index,
                'best_score': best_score,
                'best_action': best_action
            }
            
            episode_data[episode_idx].append(step_data)
    
    # 각 episode 내에서 step 순으로 정렬
    for episode_idx in episode_data:
        episode_data[episode_idx].sort(key=lambda x: x['step'])
    
    return episode_data

# postprocess function of action, to handle the case where number of dimensions are not the same
def postprocess_action(action):
    new_action = {}
    for k, v in action.items():
        if v.ndim == 1:
            new_action[k] = v[..., None]
        else:
            new_action[k] = v
    return new_action


def negative_entropy_reward(entropies: np.ndarray):
    return -entropies

def best_of_n_action_selection(action_tokens: np.ndarray, actions: np.ndarray, rewards: np.ndarray, strategy: str):
    #action tokens : (N, 1, 256) or (256,)
    while action_tokens.ndim < 3:
        action_tokens = action_tokens[None, ...]
    while actions.ndim < 3:
        actions = actions[None, ...]
    while rewards.ndim < 3:
        rewards = rewards[None, ...]
    
    action_eos = np.argmax(action_tokens == PALIGEMMA_ACTION_END_TOKEN, axis=-1) #(N, 1 )
    has_action_eos = np.any(action_tokens == PALIGEMMA_ACTION_END_TOKEN, axis=-1) #(N, 1)
    action_eos = np.where(has_action_eos, action_eos, action_tokens.shape[-1]).squeeze(-1) #(N,)
    
    T = action_tokens.shape[-1]
    
    # Create a mask for valid tokens for each item in the batch
    #NOTE : we should execute tokens after the first 3 tokens (Action: ) & before ACTION_EOS_TOKEN ("|") + 5 : base_motion + control_mode ?)
    indices = np.arange(T)
    mask = (indices >= 8) & (indices < action_eos[:, np.newaxis]) #(N, 1)
    masked_rewards = np.where(mask[:, np.newaxis, :], rewards, 0) #(N, 1, 256)
        
    # NOTE: robocasa's first 8 outputs is dummy (3 : "Action :", 5 : base_motion + control_mode ?)
    if strategy == "sum":
        scores = masked_rewards.sum(axis=-1).squeeze(axis=-1) # (N,)
    elif strategy == "avg":
        num_tokens = np.clip(action_eos - 8, 1, T)
        scores = masked_rewards.sum(axis=-1).squeeze(axis=-1) / num_tokens # (N,)
    elif strategy == "first10":
        scores = masked_rewards[..., 8:18].sum(-1).squeeze(-1) # (N,)
    elif strategy == "first5":
        scores = masked_rewards[..., 8:13].sum(-1).squeeze(-1) # (N,)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    
    best_index = int(np.argmax(scores))
    
    #actions : (N, B, H, D)

    # Check if the selected action is zero and find the next non-zero action
    while np.all(actions[best_index] == 0):
        scores[best_index] = -np.inf
        if np.all(np.isneginf(scores)):
            logging.warning("All sampled actions are zero. Using a zero action.")
            best_action = np.zeros_like(actions[0])
            break
        best_index = int(np.argmax(scores))
    else: #no break in while block
        best_action = actions[best_index]

    return best_action, best_index, scores

def action_to_dict(action_chunk: np.ndarray) -> Dict[str, np.ndarray]:
    """Action chunk를 action dictionary로 변환"""
    action_horizon, action_dim = action_chunk.shape
    assert action_dim == 12  # action dimension은 고정
    
    action_dict = {
        "action.base_motion": action_chunk[..., :4],
        "action.control_mode": (action_chunk[..., 4:5] > 0.5).astype(action_chunk.dtype),
        "action.end_effector_position": action_chunk[..., 5:8],
        "action.end_effector_rotation": action_chunk[..., 8:11],
        "action.gripper_close": (action_chunk[..., 11:12] > 0.5).astype(action_chunk.dtype),
    }
    
    return postprocess_action(action_dict)


@dataclasses.dataclass
class Args:
    
    # Model/Policy
    port: int = 8000
    host: str = "0.0.0.0"
    
    seed: int = 7

    # Setup
    env_name: str = "<ENV_NAME>" # Name of the environment to run. -> Task Name
    video_dir: str = "/virtual_lab/sjw_alinlab/suhyeok/openpi/output/robocasa/videos"
    max_episode_steps: int = 750

    # Robocasa env parameters
    robots: List[str] = dataclasses.field(default_factory=lambda: ["PandaOmron"]) # Which robot(s) to use in the env
    layout: List[int] = dataclasses.field(default_factory=lambda: [-1])
    style: List[int] = dataclasses.field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 11])
    
    generative_textures: bool = True # Use generative textures for the environment

    # Data collection parameters
    collect_data: bool = False # Whether to collect data

    # BoN
    reward_strategy: str = "log_likelihood" # "log_likelihood", "negative_entropy"
    best_of_n_strategy: str = "avg" # "sum", "avg", "first10", "first5"

    
    
def eval_robocasa(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    
    # logging 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # # Load Model from Server
    # client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    # available_policies = client.get_available_policies()
    # action_horizon = client.get_action_horizon()
    # logging.info(f"Available policies: {available_policies}")
    # logging.info(f"Action horizon: {action_horizon}")
        
    # env = load_robocasa_gym_env(
    #     args.env_name,
    #     seed=args.seed,
    #     # robosuite-related configs
    #     robots=args.robots,
    #     camera_widths=256,
    #     camera_heights=256,
    #     render_onscreen=False,
    #     # robocasa-related configs
    #     obj_instance_split="A",
    #     generative_textures="100p" if args.generative_textures else None,
    #     randomize_cameras=False,
    #     layout_ids=args.layout,
    #     style_ids=args.style,
    #     # data collection configs
    #     collect_data=args.collect_data,
    # )
    # logging.info(f"Environment {args.env_name} loaded successfully.")

    # env = RoboCasaWrapper(env)
    
    
    # #logging is_success
    # stats = defaultdict(list)
    # evaluated_episodes = set()
    # if os.path.exists(f"{args.video_dir}/prediction.txt"):
    #     with open(f"{args.video_dir}/prediction.txt", "r") as f:
    #         for line in f:
    #             if line.startswith("episode"):
    #                 try:
    #                     ep_num = int(line.split()[1])
    #                     evaluated_episodes.add(ep_num)
    #                     # is_success 값도 파싱해서 stats에 추가
    #                     is_success_str = line.strip().split(":")[-1].strip()
    #                     # [True] 또는 [False] 형태 처리
    #                     if is_success_str.startswith("[") and is_success_str.endswith("]"):
    #                         is_success_val = is_success_str[1:-1].strip()
    #                         if is_success_val.lower() == "true":
    #                             is_success = True
    #                         elif is_success_val.lower() == "false":
    #                             is_success = False
    #                         else:
    #                             is_success = False
    #                     else:
    #                         # True/False 문자열 처리
    #                         is_success = is_success_str.lower() == "true"
    #                     add_to(stats, flatten({"is_success": is_success}))
    #                 except Exception as e:
    #                     logging.warning(f"prediction.txt 파싱 에러: {e}, line: {line}")

    
    # # 1단계 비디오 저장 설정
    # video_dir_eval = f"{args.video_dir}/eval_output"
    # os.makedirs(video_dir_eval, exist_ok=True)
    
    # # Best-of-N 정보를 저장할 TSV 파일 설정 (reward_strategy가 있을 때만)
    # bon_tsv_path = None
    # bon_headers = None
    # if args.reward_strategy is not None:
    #     bon_tsv_path = f"{video_dir_eval}/best_of_n_data.tsv"
    #     bon_headers = ["episode_index", "step", "scores_list", "action_list", "best_index", "best_score", "best_action"]
        
    #     # TSV 파일에서 기존 데이터 읽기 (완성된 episode만 유지)
    #     existing_tsv_data = []
    #     if os.path.exists(bon_tsv_path):
    #         with open(bon_tsv_path, 'r', encoding='utf-8') as f:
    #             reader = csv.reader(f, delimiter='\t')
    #             try:
    #                 header = next(reader)  # 헤더 저장
    #                 for row in reader:
    #                     if len(row) >= 2:
    #                         episode_idx = int(row[0])
    #                         # 완성된 episode의 데이터만 유지, 중간에 끊긴 episode의 데이터는 제거
    #                         if episode_idx in evaluated_episodes:
    #                             existing_tsv_data.append(row)
    #             except Exception as e:
    #                 logging.warning(f"TSV 파일 읽기 에러: {e}")
        
    #     # TSV 파일을 새로 작성 (완성된 episode 데이터만 유지)
    #     if existing_tsv_data:
    #         with open(bon_tsv_path, 'w', newline='', encoding='utf-8') as f:
    #             writer = csv.writer(f, delimiter='\t')
    #             writer.writerow(bon_headers)
    #             writer.writerows(existing_tsv_data)
    #         logging.info(f"완성된 {len(evaluated_episodes)}개 episode의 TSV 데이터를 유지하고, 중간에 끊긴 episode 데이터는 제거했습니다.")
    
    # def write_bon_data_row(path, row_data, args):
    #     """Best-of-N 데이터를 TSV 파일에 저장하는 함수 (bridge_eval 스타일)"""
    #     # reward_strategy가 없으면 TSV 파일을 작성하지 않음
    #     if args.reward_strategy is None:
    #         return
            
    #     # 파일이 존재하지 않거나 비어있으면 헤더를 먼저 작성
    #     write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        
    #     with open(path, 'a', newline='', encoding='utf-8') as f:
    #         writer = csv.writer(f, delimiter='\t')
    #         if write_header:
    #             writer.writerow(bon_headers)
            
    #         # 데이터 변환하여 저장
    #         row_to_write = list(row_data)
    #         for i, item in enumerate(row_to_write):
    #             if isinstance(item, np.ndarray):
    #                 # numpy array의 경우 dtype과 함께 저장
    #                 array_data = {
    #                     'data': item.tolist(),
    #                     'dtype': str(item.dtype),
    #                     'shape': item.shape
    #                 }
    #                 row_to_write[i] = json.dumps(array_data, ensure_ascii=False)
    #             elif isinstance(item, list):
    #                 # list의 경우 그대로 저장
    #                 row_to_write[i] = json.dumps(item, ensure_ascii=False)
    #             elif isinstance(item, (int, float)):
    #                 # 숫자의 경우 그대로 저장
    #                 row_to_write[i] = str(item)
    #             elif isinstance(item, str) and item.endswith('.npy'):
    #                 # .npy 파일 경로는 그대로 저장
    #                 row_to_write[i] = item
            
    #         writer.writerow(row_to_write)
            
            
    # def episode_trigger(episode_id):
    #     # episode_id는 실제 episode 번호 (0-49)
    #     # 모든 episode에 대해 비디오 생성 (n_save=1이므로)
    #     return True
    
    # env = RecordVideo(
    #     env,
    #     Path(video_dir_eval),
    #     disable_logger=True,
    #     episode_trigger=episode_trigger,
    #     fps=20,
    #     name_prefix=f"{args.env_name}_eval",
    # )

    # env = MultiStepWrapper(
    #     env,
    #     video_delta_indices=np.arange(1), #t
    #     state_delta_indices=np.arange(1), #t
    #     n_action_steps=action_horizon,
    #     max_episode_steps=args.max_episode_steps,
    # )


    # logging.info(f"Starting evaluation for {args.env_name} with episodes 6, 7, 8...")
    # # main evaluation loop - episode 6, 7, 8 실행 (첫 번째 성공 시 종료)
    # successful_episode = None
    
    # for episode_num in [3]:        

    #     pbar = tqdm(
    #         total=args.max_episode_steps, desc=f"Episode {episode_num} / {env.unwrapped.get_ep_meta()['lang']}", leave=False
    #     )
        
    #     # 1. Reset Environment and Get Observation
    #     # episode_num번 reset을 호출하여 해당 episode로 이동
    #     for reset_count in range(episode_num + 1):
    #         obs, info = env.reset()
    #         if reset_count == episode_num:
    #             # 원하는 episode에 도달했을 때만 실행
    #             break
                
    #     # RecordVideo wrapper에 episode 번호 설정
    #     if hasattr(env, 'set_episode_number'):
    #         env.set_episode_number(episode_num)
            
    #     # Episode 실행을 위한 변수 초기화
    #     done = False
    #     step = 0
            
    #     while not done:
            
    #         # NOTE: The observation is vertically flipped, uint8
    #         obs["video.left_view"] = np.flip(obs["video.left_view"], axis=1)
    #         obs["video.right_view"] = np.flip(obs["video.right_view"], axis=1)
    #         obs["video.wrist_view"] = np.flip(obs["video.wrist_view"], axis=1)
            
    #         # TODO : Train때 state 전체를 받는 것을 가정했어서 eval할때 concat해서 넘겨줘야함..
    #         state = np.concatenate([obs["state.base_position"], obs["state.base_rotation"], \
    #                                 obs["state.end_effector_position_absolute"], obs["state.end_effector_position_relative"], \
    #                                 obs["state.end_effector_rotation_absolute"], obs["state.end_effector_rotation_relative"], \
    #                                 obs["state.gripper_qpos"], obs["state.gripper_qvel"], \
    #                                 obs["state.joint_position"], obs["state.joint_position_cos"], obs["state.joint_position_sin"], \
    #                                 obs["state.joint_velocity"]], axis=1)
            
    #         assert state.shape == (1,53)
        
            
    #         element ={
    #             "video.left_view" : obs["video.left_view"][0],
    #             "video.right_view" : obs["video.right_view"][0],
    #             "video.wrist_view" : obs["video.wrist_view"][0],
    #             "state" : state[0],
    #             "annotation.human.action.task_description" : obs["annotation.human.action.task_description"][0],
    #         }
                            
    #         # 2. Forward Model
    #         outputs = client.infer(element, policy_name="default")
            
    #         # Best-of-N 관련 변수들을 저장할 변수들
    #         bon_scores = None
    #         bon_actions = None
    #         bon_best_index = None
    #         bon_best_score = None
    #         bon_best_action = None
            
    #         if args.reward_strategy is not None:
    #             if args.reward_strategy == "negative_entropy":
    #                 if "entropies" not in outputs:
    #                     raise ValueError("To use 'negative_entropy' with the default policy, server must be run with --policy.reward_strategy=\"negative_entropy\"")
                    
    #                 action_tokens = outputs["action_tokens"]
    #                 actions = outputs["actions"]
    #                 entropies = outputs["entropies"]
    #                 rewards = negative_entropy_reward(entropies)
                    
    #             elif args.reward_strategy == "log_likelihood":
    #                 action_tokens = outputs["action_tokens"]
    #                 actions = outputs["actions"]
    #                 log_probs = outputs["log_probs"]
    #                 rewards = log_probs
                    
    #             else:
    #                 raise NotImplementedError(f"Unknown reward strategy: {args.reward_strategy}")
                    
    #             action_chunk, best_index, scores = best_of_n_action_selection(action_tokens, actions, rewards, strategy=args.best_of_n_strategy)
                
    #             if action_chunk.ndim == 3: # (1, H, D), openpi transform changed
    #                 actions = np.squeeze(actions, axis=1)
    #                 action_chunk = action_chunk[0] # (H, D)
                
    #             # actions와 action_chunk의 shape 확인
    #             assert actions.shape[0] == len(scores), f"actions shape {actions.shape}와 scores length {len(scores)}가 일치하지 않습니다"
    #             assert action_chunk.shape == (action_horizon, 12), f"action_chunk shape {action_chunk.shape}가 예상과 다릅니다. 예상: ({action_horizon}, 12)"
                
    #             # Best-of-N 데이터를 변수에 저장
    #             bon_scores = scores
    #             bon_actions = actions
    #             bon_best_index = best_index
    #             bon_best_score = scores[best_index]
    #             bon_best_action = action_chunk
                
    #         else: #greedy decoding
    #             action_chunk = outputs["actions"]
                
    #         assert action_chunk.shape == (action_horizon, 12) #(16,12)
            
    #         action_dict = {
    #             "action.base_motion" : action_chunk[..., :4],
    #             "action.control_mode" : (action_chunk[..., 4:5] > 0.5).astype(action_chunk.dtype), #should be binary
    #             "action.end_effector_position" : action_chunk[..., 5:8],
    #             "action.end_effector_rotation" : action_chunk[..., 8:11],
    #             "action.gripper_close" : (action_chunk[..., 11:12] > 0.5).astype(action_chunk.dtype), #should be binary
    #         }
            
    #         post_action = postprocess_action(action_dict)

    #         # Best-of-N 데이터 저장 (action 수행 전 step으로 저장) - reward_strategy가 있을 때만
    #         if args.reward_strategy is not None and bon_scores is not None and bon_tsv_path is not None:
    #             # numpy array를 .npy 파일로 저장
    #             episode_dir = f"{video_dir_eval}/numpy_data/episode_{episode_num}"
    #             os.makedirs(episode_dir, exist_ok=True)
                
    #             # bon_actions와 bon_best_action을 .npy 파일로 저장
    #             bon_actions_path = f"{episode_dir}/step_{step}_actions.npy"
    #             bon_best_action_path = f"{episode_dir}/step_{step}_best_action.npy"
                
    #             np.save(bon_actions_path, bon_actions)
    #             np.save(bon_best_action_path, bon_best_action)
                
    #             # TSV에는 파일 경로만 저장
    #             write_bon_data_row(bon_tsv_path, [
    #                 episode_num, step, bon_scores, bon_actions_path, bon_best_index, bon_best_score, bon_best_action_path
    #             ], args)
            
    #         # 3. Execute action in environment
    #         obs, reward, terminated, truncated, info = env.step(post_action)
    #         done = terminated or truncated
    #         step += action_horizon
            
    #         pbar.update(action_horizon)
            
    #     add_to(stats, flatten({"is_success": info["is_success"]}))
    #     with open(f"{args.video_dir}/prediction.txt", "a") as f:
    #         f.write(f"episode {episode_num} is_success: {info['is_success']} \n")
    #     pbar.close()
        
    #     # 성공한 episode가 나오면 바로 종료
    #     if info["is_success"]:
    #         successful_episode = episode_num
    #         logging.info(f"Episode {episode_num} 성공! 다른 episode는 건너뛰고 rollout으로 넘어갑니다.")
    #         break

    # # 4. Close Environment
    # env.close()
    
    # # 성공한 episode가 없으면 종료
    # # if successful_episode is None:
    # #     logging.error("모든 episode가 실패했습니다. rollout을 진행할 수 없습니다.")
    # #     return
    
    # # 2단계: TSV에서 성공한 episode의 best action만 실행
    # logging.info("=" * 60)
    # logging.info(f"2단계: TSV에서 성공한 episode {successful_episode}의 best action만 실행")
    # logging.info("=" * 60)
    
    # TSV 데이터 로드
    # tsv_path = f"{video_dir_eval}/best_of_n_data.tsv"
    tsv_path = "/virtual_lab/sjw_alinlab/suhyeok/openpi/output_for_analysis/robocasa/PnPCounterToStove/pi0_fast_robocasa_240demos_base/pi0-fast-robocasa-240demos-base-batch-64/10000/best_of_n:4_temp:0.7_reward:log_likelihood_selec:first5/best_of_n_data.tsv"
    if not os.path.exists(tsv_path):
        logging.error(f"TSV 파일을 찾을 수 없습니다: {tsv_path}")
        return
    
    episode_data = load_tsv_data(tsv_path)
    logging.info(f"로드된 episode 수: {len(episode_data)}")
    
    # TSV에서 action_horizon 추출
    action_horizon = 16  # 기본값
    if episode_data:
        first_episode = list(episode_data.values())[0]
        if first_episode:
            first_step = first_episode[0]
            if first_step['action_list'] is not None and len(first_step['action_list']) > 0:
                first_action = first_step['action_list'][0]
                if hasattr(first_action, 'shape') and len(first_action.shape) >= 2:
                    action_horizon = first_action.shape[0]
                    logging.info(f"TSV에서 추출한 action_horizon: {action_horizon}")
    
    # 성공한 episode 확인
    # episode_idx = successful_episode
    episode_idx = 3
    if episode_idx not in episode_data:
        logging.error(f"Episode {episode_idx} 데이터가 없습니다.")
        return
        
    episode_steps = episode_data[episode_idx]
    if not episode_steps:
        logging.error(f"Episode {episode_idx}에 step 데이터가 없습니다.")
        return
    
    logging.info(f"Episode {episode_idx}의 모든 step ({len(episode_steps)}개)에 대해 best action만 실행")
    logging.info("참고: TSV의 step은 action 수행 전 시점으로 저장됨")
    
    # 새로운 환경 생성 (두 번째 환경) - 첫 번째 환경과 동일한 설정
    env2 = load_robocasa_gym_env(
        args.env_name,
        seed=args.seed,
        # robosuite-related configs
        robots=args.robots,
        camera_widths=256,
        camera_heights=256,
        render_onscreen=False,
        # robocasa-related configs
        obj_instance_split="A",
        generative_textures="100p" if args.generative_textures else None,
        randomize_cameras=False,
        layout_ids=args.layout,
        style_ids=args.style,
        # data collection configs
        collect_data=False,  # rollout에서는 데이터 수집 안함
    )
    env2 = RoboCasaWrapper(env2)
    
    # 비디오 저장 설정
    video_dir_rollout = f"{args.video_dir}/rollout_output"
    os.makedirs(video_dir_rollout, exist_ok=True)
    
    def episode_trigger_rollout(episode_id):
        return episode_id == episode_idx
    
    env2 = RecordVideo(
        env2,
        Path(video_dir_rollout),
        disable_logger=True,
        episode_trigger=episode_trigger_rollout,
        fps=20,
        name_prefix=f"{args.env_name}_rollout_ep{episode_idx}",
    )
    
    env2 = MultiStepWrapper(
        env2,
        video_delta_indices=np.arange(1),
        state_delta_indices=np.arange(1),
        n_action_steps=action_horizon,
        max_episode_steps=args.max_episode_steps,
    )
    
    # Episode 실행
    logging.info(f"Episode {episode_idx} 시작...")
    
    # reset: episode_idx+1번 반복, 마지막에 rollout
    for reset_episode_idx in range(episode_idx + 1):
        obs, info = env2.reset()
        if reset_episode_idx == episode_idx:
            if hasattr(env2, 'set_episode_number'):
                env2.set_episode_number(episode_idx)
            done = False
            step = 0
    
    # 모든 step에 대해 best action만 실행
    current_env_step = 0
    for step_idx, step_data in enumerate(episode_steps):
        logging.info(f"Step {step_idx} 실행 중...")
        
        # step-1까지 best action만 따라가기
        for prev_step_idx in range(step_idx):
            prev_step = episode_steps[prev_step_idx]
            target_step = prev_step['step']  # step이 action 수행 전 시점이므로 그대로 사용
            best_action = prev_step['best_action']
            while current_env_step < target_step and not done:
                action_dict = action_to_dict(best_action)
                obs, reward, terminated, truncated, info = env2.step(action_dict)
                obs["video.left_view"] = np.flip(obs["video.left_view"], axis=1)
                obs["video.right_view"] = np.flip(obs["video.right_view"], axis=1)
                obs["video.wrist_view"] = np.flip(obs["video.wrist_view"], axis=1)
                done = terminated or truncated
                current_env_step += action_horizon
        
        # 현재 step에서 best action 실행
        best_action = step_data['best_action']
        action_dict = action_to_dict(best_action)
        obs, reward, terminated, truncated, info = env2.step(action_dict)
        obs["video.left_view"] = np.flip(obs["video.left_view"], axis=1)
        obs["video.right_view"] = np.flip(obs["video.right_view"], axis=1)
        obs["video.wrist_view"] = np.flip(obs["video.wrist_view"], axis=1)
        done = terminated or truncated
        current_env_step += action_horizon
        
        if done:
            break
    
    # 환경 종료
    env2.close()
    
    logging.info("=" * 60)
    logging.info("모든 비디오 생성 완료!")
    logging.info("=" * 60)
    logging.info(f"1단계 비디오 (eval): {video_dir_eval}")
    logging.info(f"2단계 비디오 (rollout): {video_dir_rollout}")
    logging.info(f"TSV 파일: {tsv_path}")
    logging.info(f"성공한 episode: {successful_episode}")
    logging.info(f"선택된 episode: {episode_idx}")
    logging.info("두 비디오를 비교하여 정확히 동일한지 확인하세요.")


if __name__ == "__main__":
    tyro.cli(eval_robocasa)

