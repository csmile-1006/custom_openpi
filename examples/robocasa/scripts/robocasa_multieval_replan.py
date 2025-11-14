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
from pathlib import Path
from tqdm import tqdm
import numpy as np
import dataclasses
from typing import List, Optional
import tyro
import logging
import time
import warnings
import glob

from openpi_client import websocket_client_policy as _websocket_client_policy

import robocasa
import robosuite

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gr00t.eval.wrappers.robocasa_wrapper_multieval import load_robocasa_gym_env

warnings.simplefilter("ignore", category=FutureWarning)

PALIGEMMA_ACTION_END_TOKEN = 235371

def calculate_episode_ranges(n_episodes: int, n_envs: int) -> list[tuple[int, int]]:
    """각 환경별 episode 범위를 계산"""
    episodes_per_env = n_episodes // n_envs
    assert n_episodes % n_envs == 0, "n_episodes must be divisible by n_envs"
    
    ranges = []
    for env_idx in range(n_envs):
        start_episode = env_idx * episodes_per_env
        end_episode = start_episode + episodes_per_env
        ranges.append((start_episode, end_episode))
    
    return ranges

def negative_entropy_reward(entropies: np.ndarray) -> np.ndarray:
    return -entropies

def best_of_n_action_selection(action_tokens: np.ndarray, actions: np.ndarray, rewards: np.ndarray, strategy: str):
    #action tokens : (N, B, 256) or (B, 256)
    while action_tokens.ndim < 3:
        action_tokens = action_tokens[None, ...]
    while actions.ndim < 3:
        actions = actions[None, ...]
    while rewards.ndim < 3:
        rewards = rewards[None, ...]
    
    # action_eos: (N, B) - 각 N, B 조합에 대해 EOS 토큰 위치 찾기
    action_eos = np.argmax(action_tokens == PALIGEMMA_ACTION_END_TOKEN, axis=-1)  # (N, B)
    has_action_eos = np.any(action_tokens == PALIGEMMA_ACTION_END_TOKEN, axis=-1)  # (N, B)
    action_eos = np.where(has_action_eos, action_eos, action_tokens.shape[-1])  # (N, B)
    
    T = action_tokens.shape[-1]
    
    # Create a mask for valid tokens for each item in the batch
    #NOTE : we should execute tokens after the first 3 tokens (Action: ) & before ACTION_EOS_TOKEN ("|") + 5 : base_motion + control_mode
    indices = np.arange(T)  # (256,)
    # mask: (N, B, 256)
    mask = (indices[None, None, :] >= 8) & (indices[None, None, :] < action_eos[..., None])
    masked_rewards = np.where(mask, rewards, 0)  # (N, B, 256)
        
    # NOTE: robocasa's first 8 outputs is dummy (3 : "Action :", 5 : base_motion + control_mode)
    if strategy == "sum":
        scores = masked_rewards.sum(axis=-1)  # (N, B)
    elif strategy == "avg":
        num_tokens = np.clip(action_eos - 8, 1, T)  # (N, B)
        scores = masked_rewards.sum(axis=-1) / num_tokens  # (N, B)
    elif strategy == "first10":
        scores = masked_rewards[..., 8:18].sum(-1)  # (N, B)
    elif strategy == "first5":
        scores = masked_rewards[..., 8:13].sum(-1)  # (N, B)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Select best actions for each batch
    best_actions = []
    best_indices = []
    
    for b in range(scores.shape[1]):  # 각 배치에 대해
        batch_scores = scores[:, b]  # (N,)
        best_index = int(np.argmax(batch_scores))
        
        # Check if the selected action is zero and find the next non-zero action
        while np.all(actions[best_index, b] == 0):
            batch_scores[best_index] = -np.inf
            if np.all(np.isneginf(batch_scores)):
                logging.warning(f"All sampled actions are zero for batch {b}. Using a zero action.")
                best_action = np.zeros_like(actions[0, b])
                break
            best_index = int(np.argmax(batch_scores))
        else: #no break in while block
            best_action = actions[best_index, b]
        
        best_actions.append(best_action)
        best_indices.append(best_index)
    
    # (B,) 형태로 반환
    best_actions = np.array(best_actions)  # (B, action_dim)
    best_indices = np.array(best_indices)  # (B,)
    
    return best_actions, best_indices, scores

def analyze_existing_videos(video_path: str, env_name: str, n_envs: int, n_episodes: int) -> tuple[list[int], dict[int, bool]]:
    """기존 결과를 분석하고 불일치하는 파일들을 정리"""
    skip_episodes_per_env = [0] * n_envs
    existing_episode_results = {}  # {episode_number: success}
    
    prediction_file = os.path.join(video_path, "prediction.txt")
    
    # 1. prediction.txt 분석
    if os.path.exists(prediction_file):
        logging.info("Found existing prediction.txt, analyzing...")
        with open(prediction_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("episode ") and "is_success:" in line:
                    try:
                        parts = line.split()
                        episode_num = int(parts[1])
                        success = parts[3] == "True"
                        existing_episode_results[episode_num] = success
                    except (ValueError, IndexError) as e:
                        logging.warning(f"Could not parse prediction line: {line}, error: {e}")
    
    # 2. 비디오 파일 분석 및 orphan 파일 정리
    video_episodes = set()
    video_files_to_delete = []
    
    for env_idx in range(n_envs):
        pattern = f"{env_name}-env{env_idx}-episode_*.mp4"
        video_files = glob.glob(os.path.join(video_path, pattern))
        
        for video_file in video_files:
            filename = os.path.basename(video_file)
            try:
                episode_number = int(filename.split("episode_")[1].split(".mp4")[0])
                if episode_number in existing_episode_results:
                    video_episodes.add(episode_number)
                    logging.info(f"Episode {episode_number}: Video and prediction match")
                else:
                    video_files_to_delete.append(video_file)
                    logging.warning(f"Episode {episode_number}: Video exists but no prediction, will delete")
            except (IndexError, ValueError) as e:
                logging.warning(f"Could not parse episode number from {filename}: {e}")
                video_files_to_delete.append(video_file)
    
    # 3. prediction.txt에서 orphan episode들 제거
    episodes_to_remove = [ep for ep in existing_episode_results.keys() if ep not in video_episodes]
    if episodes_to_remove:
        logging.warning(f"Episodes {episodes_to_remove}: Prediction exists but no video, will remove from prediction")
    
    # 4. 실제 파일 삭제 및 prediction.txt 정리
    if video_files_to_delete:
        logging.info(f"Deleting {len(video_files_to_delete)} orphaned video files...")
        for video_file in video_files_to_delete:
            try:
                os.remove(video_file)
                logging.info(f"Deleted: {video_file}")
            except OSError as e:
                logging.error(f"Failed to delete {video_file}: {e}")
    
    if episodes_to_remove:
        logging.info(f"Removing {len(episodes_to_remove)} episodes from prediction.txt...")
        if os.path.exists(prediction_file):
            with open(prediction_file, 'r') as f:
                lines = f.readlines()
            
            success_rate_line = None
            episode_lines = []
            for line in lines:
                if line.startswith("is_success:"):
                    success_rate_line = line
                elif line.startswith("episode "):
                    try:
                        episode_num = int(line.split()[1])
                        if episode_num not in episodes_to_remove:
                            episode_lines.append(line)
                    except (ValueError, IndexError):
                        continue
            
            with open(prediction_file, "w") as f:
                for line in sorted(episode_lines, key=lambda x: int(x.split()[1])):
                    f.write(line)
                if success_rate_line:
                    f.write(success_rate_line)
    
    # 5. 최종 검증된 결과로 skip_episodes_per_env 계산
    if existing_episode_results:
        # 실제 남아있는 비디오 파일들 재확인
        actual_video_episodes = set()
        for env_idx in range(n_envs):
            pattern = f"{env_name}-env{env_idx}-episode_*.mp4"
            video_files = glob.glob(os.path.join(video_path, pattern))
            
            for video_file in video_files:
                try:
                    episode_number = int(os.path.basename(video_file).split("episode_")[1].split(".mp4")[0])
                    actual_video_episodes.add(episode_number)
                except (IndexError, ValueError):
                    continue
        
        # 공통 함수를 사용하여 episode 범위 계산
        episode_ranges = calculate_episode_ranges(n_episodes, n_envs)
        
        # 각 환경별로 할당된 episode 범위 내에서 existing episode 개수 계산
        for env_idx in range(n_envs):
            env_start_episode, env_end_episode = episode_ranges[env_idx]
            
            # 해당 범위 내에서 prediction.txt와 비디오가 모두 있는 episode 개수
            env_existing_count = 0
            for episode_num in range(env_start_episode, env_end_episode):
                if episode_num in existing_episode_results and episode_num in actual_video_episodes:
                    env_existing_count += 1
            
            skip_episodes_per_env[env_idx] = env_existing_count
            logging.info(f"Environment {env_idx}: {skip_episodes_per_env[env_idx]} valid episodes (range: {env_start_episode}-{env_end_episode-1})")
        
        # existing_episode_results도 실제 비디오가 있는 것만으로 업데이트
        existing_episode_results = {
            ep: existing_episode_results[ep] 
            for ep in existing_episode_results.keys() 
            if ep in actual_video_episodes
        }
    else:
        logging.info("No valid episodes found, starting from scratch")
    
    return skip_episodes_per_env, existing_episode_results

def save_episode_result(prediction_file: str, episode_num: int, success: bool):
    """개별 episode 결과를 즉시 파일에 추가"""
    try:
        with open(prediction_file, "a") as f:
            f.write(f"episode {episode_num} is_success: {success}\n")
    except IOError as e:
        logging.error(f"Failed to save episode result: {e}")

def finalize_prediction_file(prediction_file: str, episode_results: dict, n_episodes: int):
    """최종 성공률 계산 및 파일 정리 - 모든 episode가 완료된 경우에만 is_success 작성"""
    success_count = sum(episode_results.values())
    total_count = len(episode_results)
    
    # 기존 파일 읽기
    existing_lines = []
    if os.path.exists(prediction_file):
        with open(prediction_file, 'r') as f:
            existing_lines = f.readlines()
    
    # episode 결과만 필터링 (성공률 라인 제외)
    episode_lines = [line for line in existing_lines if line.startswith("episode ")]
    
    # 중복 제거 및 정렬
    seen_episodes = set()
    unique_lines = []
    for line in episode_lines:
        try:
            episode_num = int(line.split()[1])
            if episode_num not in seen_episodes:
                seen_episodes.add(episode_num)
                unique_lines.append(line)
        except (ValueError, IndexError):
            continue
    
    # 정렬하여 재작성
    with open(prediction_file, "w") as f:
        for line in sorted(unique_lines, key=lambda x: int(x.split()[1])):
            f.write(line)
        
        # 모든 episode가 완료된 경우에만 is_success 작성
        if total_count == n_episodes:
            success_rate = success_count / total_count
            f.write(f"is_success: {success_rate:.2f} ({success_count}/{total_count})\n")
            logging.info(f"All {n_episodes} episodes completed. Final success rate: {success_rate:.2f}")
        else:
            logging.info(f"Only {total_count}/{n_episodes} episodes completed. Skipping is_success calculation.")

@dataclasses.dataclass
class Args:
    
    # Model/Policy
    port: int = 8000
    host: str = "0.0.0.0"
    
    seed: int = 7

    # Setup
    env_name: str = "<ENV_NAME>" # Name of the environment to run. -> Task Name
    video_path: str = "/virtual_lab/sjw_alinlab/suhyeok/openpi/output/robocasa/videos"
    n_episodes: int = 50
    n_envs: int = 1
    
    # Planning
    replan_steps: int = 5  # Length of truncated action chunk to execute before replanning

    # Robocasa env parameters
    controller: Optional[str] = None # Choice of controller. Can be, eg. 'NONE' or 'WHOLE_BODY_IK', etc. Or path to controller json file"
    robots: List[str] = dataclasses.field(default_factory=lambda: ["PandaOmron"]) # Which robot(s) to use in the env
    config: str = "single-arm-opposed" # Specified environment configuration if necessary
    arm: str = "right"
    obj_groups: Optional[List[str]] = None # In kitchen environments, either the name of a group to sample object from or path to an .xml file"
    layout: List[int] = dataclasses.field(default_factory=lambda: [-1])
    style: List[int] = dataclasses.field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 11])
    
    generative_textures: bool = False # Use generative textures for the environment

    # Data collection parameters
    collect_data: bool = False # Whether to collect data
    data_collection_path: str = "" # Path to save the data collection
    
    # BoN
    reward_strategy : Optional[str] = None # "log_likelihood", "negative_entropy"
    best_of_n_strategy : str = "avg" # "sum", "avg", "first10", "first5"

    
    
def eval_robocasa(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    
    # logging 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load Model from Server
    try:
        client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        available_policies = client.get_available_policies()
        action_horizon = client.get_action_horizon()
        logging.info(f"Available policies: {available_policies}")
        logging.info(f"Action horizon: {action_horizon}")
    except Exception as e:
        logging.error(f"Failed to connect to server at {args.host}:{args.port}: {e}")
        raise

    # Validate replan steps
    if args.replan_steps < 1:
        raise ValueError(f"replan_steps must be >= 1, got {args.replan_steps}")
    if args.replan_steps > action_horizon:
        raise ValueError(f"replan_steps ({args.replan_steps}) cannot exceed action_horizon ({action_horizon})")

    env_name = args.env_name
    logging.info(f"Environment name: {env_name}")
    logging.info(f"Using replan_steps = {args.replan_steps}")

    # 재시작을 위한 skip episodes 정보 분석 (기존 결과도 함께 반환)
    skip_episodes_per_env, existing_episode_results = analyze_existing_videos(args.video_path, args.env_name, args.n_envs, args.n_episodes)

    env = load_robocasa_gym_env(
        args.env_name,
        seed=args.seed,
        n_envs=args.n_envs,
        n_episodes=args.n_episodes,  # episodes_per_env 계산을 위해 전달
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
        collect_data=args.collect_data,
        collect_directory=Path(args.data_collection_path) if args.collect_data else None,
        # video configs
        video_path=args.video_path,
        # multi-step configs
        action_horizon=args.replan_steps,
        video_delta_indices=np.array([0]), #np.arange(1)
        state_delta_indices=np.array([0]),
        # resume configs
        skip_episodes_per_env=skip_episodes_per_env,
    )
    logging.info(f"Environment {args.env_name} loaded successfully.")
            
    # main evaluation loop
    start_time = time.time()
    
    # Initialize tracking variables
    current_rewards = [0] * args.n_envs
    current_lengths = [0] * args.n_envs
    completed_episodes = sum(skip_episodes_per_env)
    current_successes = [False] * args.n_envs

    # Episode 번호 추적 - 공통 함수 사용
    episode_ranges = calculate_episode_ranges(args.n_episodes, args.n_envs)
    episode_numbers = []
    for i in range(args.n_envs):
        # 각 환경의 시작 episode 번호 = 기본 시작점 + skip된 episode 수
        start_episode = episode_ranges[i][0] + skip_episodes_per_env[i]
        episode_numbers.append(start_episode)
    
    # 기존 결과를 episode_results에 추가
    episode_results = existing_episode_results.copy()
    
    # Initial environment reset
    obs, _ = env.reset(seed=args.seed)
    pbar = tqdm(
        total=args.n_episodes - completed_episodes,
        desc=f"Evaluating {args.n_episodes - completed_episodes} episodes",
        leave=False,
    )

    # Main simulation loop
    while completed_episodes < args.n_episodes:
                    
        ### observation shape
        # state.base_position : (num_envs, 1, 3)

        # NOTE : Train때 state 전체를 받는 것을 가정했어서 eval할때 concat해서 넘겨줘야함..
        state = np.concatenate([obs["state.base_position"], obs["state.base_rotation"], \
                                obs["state.end_effector_position_absolute"], obs["state.end_effector_position_relative"], \
                                obs["state.end_effector_rotation_absolute"], obs["state.end_effector_rotation_relative"], \
                                obs["state.gripper_qpos"], obs["state.gripper_qvel"], \
                                obs["state.joint_position"], obs["state.joint_position_cos"], obs["state.joint_position_sin"], \
                                obs["state.joint_velocity"]], axis=2)
    
        assert state.shape == (args.n_envs, 1, 53)
        
        # Element
        # video.left_view: shape=(num_envs, 256, 256, 3), dtype=uint8
        # video.right_view: shape=(num_envs, 256, 256, 3), dtype=uint8
        # video.wrist_view: shape=(num_envs, 256, 256, 3), dtype=uint8
        # state : shape=(num_envs, 53)
        # prompt : list[str] 
        
        element ={
            "video.left_view" : np.squeeze(obs["video.left_view"], axis=1),
            "video.right_view" : np.squeeze(obs["video.right_view"], axis=1),
            "video.wrist_view" : np.squeeze(obs["video.wrist_view"], axis=1),
            "state" : np.squeeze(state, axis=1),
            "annotation.human.action.task_description" : [obs["annotation.human.action.task_description"][i][0] for i in range(args.n_envs)],
        }
                                    
        # 2. Forward Model
        # Best-of-N =1 -> action_chunk : (B, 16, 12), action_tokens, log_probs : (B, 256)
        # Best-of-N > 1 -> action_chunk : (N, B, 16, 12), action_tokens, log_probs : (N, B, 256)
        outputs = client.infer(element, policy_name="default")
                        
        if args.reward_strategy is not None:
            if args.reward_strategy == "negative_entropy":
                if "entropies" not in outputs:
                    raise ValueError("To use 'negative_entropy' with the default policy, server must be run with --policy.reward_strategy=\"negative_entropy\"")
                
                action_tokens = outputs["action_tokens"]
                actions = outputs["actions"]
                entropies = outputs["entropies"]
                rewards = negative_entropy_reward(entropies)
                
            elif args.reward_strategy == "log_likelihood":
                action_tokens = outputs["action_tokens"]
                actions = outputs["actions"]
                log_probs = outputs["log_probs"]
                rewards = log_probs
                
            elif args.reward_strategy == "self_certainty" or args.reward_strategy == "kl_div":
                action_tokens = outputs["action_tokens"]
                actions = outputs["actions"]
                kl_divs = outputs["kl_divs"]
                rewards = kl_divs #bigger means predicted distribution deviated from randomness
                
            else:
                raise NotImplementedError(f"Unknown reward strategy: {args.reward_strategy}")
                
            action_chunk, best_index, scores = best_of_n_action_selection(action_tokens, actions, rewards, strategy=args.best_of_n_strategy)
            
        else: #Greedy Decoding or N=1
            #action_chunk : (B, 16, 12)
            action_chunk = outputs["actions"]
            
        # import pdb; pdb.set_trace()
                        
        # assert action_chunk.shape == (args.n_envs, action_horizon, 12)

        # Truncate the action chunk to replan_steps and discard the rest
        if action_chunk.shape[1] < args.replan_steps:
            raise ValueError(f"Predicted action chunk is shorter ({action_chunk.shape[1]}) than replan_steps ({args.replan_steps})")
        action_chunk = action_chunk[:, :args.replan_steps, :]
            
        action_dict = {
            "action.base_motion" : action_chunk[..., :4], # (B, replan_steps, 4)
            "action.control_mode" : (action_chunk[..., 4:5] > 0.5).astype(action_chunk.dtype), #should be binary
            "action.end_effector_position" : action_chunk[..., 5:8],
            "action.end_effector_rotation" : action_chunk[..., 8:11],
            "action.gripper_close" : (action_chunk[..., 11:12] > 0.5).astype(action_chunk.dtype), #should be binary
        }
        
        # 3. Execute action in environment
        next_obs, rewards, terminations, truncations, env_infos = env.step(action_dict)
                                
        # Update episode tracking
        for env_idx in range(args.n_envs):
            # Dummy 응답인지 확인 (reward = -999.0)
            if rewards[env_idx] == -999.0:
                continue  # Dummy 응답은 완전히 제외
                        
            try:
                success_info = env_infos["success"][env_idx]
                if isinstance(success_info, (list, tuple, np.ndarray)):
                    if len(success_info) > 0:
                        current_successes[env_idx] |= bool(success_info[0])
                    else:
                        current_successes[env_idx] |= False
                else: #scalar value
                    # 스칼라 값인 경우 (True, False, None 등)
                    current_successes[env_idx] |= bool(success_info)
            except Exception as e:
                logging.warning(f"Error processing success info for env {env_idx}: {e}")
                current_successes[env_idx] |= False
                
            current_rewards[env_idx] += rewards[env_idx]
            current_lengths[env_idx] += 1 #execution 횟수를 말하는 것
            
            # If episode ended, store results
            if terminations[env_idx] or truncations[env_idx]:
                # Episode별 결과를 딕셔너리에 저장
                episode_results[episode_numbers[env_idx]] = current_successes[env_idx]
                
                # 즉시 파일에 저장 (재시작 대비)
                save_episode_result(f"{args.video_path}/prediction.txt", 
                                  episode_numbers[env_idx], 
                                  current_successes[env_idx])
                
                # Reset trackers for this environment
                current_successes[env_idx] = False
                current_rewards[env_idx] = 0
                current_lengths[env_idx] = 0
                completed_episodes += 1 
                episode_numbers[env_idx] += 1  # 해당 환경의 episode 번호 증가
                pbar.update(1)
                
        obs = next_obs
            
    pbar.close()
    env.close()
    
    print(f"Collecting {args.n_episodes} episodes took {time.time() - start_time:.2f} seconds")
    assert completed_episodes >= args.n_episodes, (
        f"Expected {args.n_episodes} episodes, got {completed_episodes}"
    )

    if args.collect_data:
        raise NotImplementedError("Collect data is not implemented yet")

    # 최종 정리
    finalize_prediction_file(f"{args.video_path}/prediction.txt", episode_results, args.n_episodes)

    exit()
    
if __name__ == "__main__":
    tyro.cli(eval_robocasa)

