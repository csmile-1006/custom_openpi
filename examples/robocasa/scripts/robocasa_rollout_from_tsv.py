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
import json
from collections import defaultdict
from pathlib import Path
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

from gr00t.eval.wrappers.record_video import RecordVideo
from gr00t.eval.wrappers.robocasa_wrapper import RoboCasaWrapper, load_robocasa_gym_env
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper


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


def postprocess_action(action):
    """Postprocess function of action, to handle the case where number of dimensions are not the same"""
    new_action = {}
    for k, v in action.items():
        if v.ndim == 1:
            new_action[k] = v[..., None]
        else:
            new_action[k] = v
    return new_action


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
            best_index = int(row['best_index']) if 'best_index' in row and row['best_index'] != '' else None
            best_score = float(row['best_score']) if 'best_score' in row and row['best_score'] != '' else None
            best_action_path = row.get('best_action', '')  # .npy 파일 경로 또는 비어있을 수 있음
            bon_info_raw = row.get('bon_info', None)
            is_success_raw = row.get('is_success', None)
            
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
                action_list = json.loads(action_list_path)  # compare_rollout_and_eval.py와 완전히 동일하게
                        
            # best_action을 .npy 파일에서 로드 (있을 때만)
            best_action = None
            if best_action_path:
                if best_action_path.endswith('.npy'):
                    best_action = np.load(best_action_path)
                else:
                    # 기존 JSON 방식 (호환성 유지)
                    try:
                        best_action_data = json.loads(best_action_path)
                        if isinstance(best_action_data, dict) and 'data' in best_action_data:
                            best_action = np.array(best_action_data['data'], dtype=best_action_data['dtype'])
                        else:
                            best_action = np.array(best_action_data)
                    except Exception:
                        best_action = None
            
            # action_list가 (N, B, 16, 12) 형태인 경우 B 차원을 squeeze
            if hasattr(action_list, 'shape') and len(action_list.shape) == 4 and action_list.shape[1] == 1:
                action_list = np.squeeze(action_list, axis=1)  # (N, 16, 12)
            
            # bon_info 복원
            bon_info = None
            if bon_info_raw is not None and bon_info_raw != '':
                try:
                    bon_info = json.loads(bon_info_raw)
                    # 주요 키들을 numpy array로 변환
                    for k in [
                        'likelihood_sum', 'likelihood_first5', 'likelihood_first5_element',
                        'entropy_sum', 'entropy_first5', 'entropy_first5_element',
                        'kl_div_sum', 'kl_div_first5', 'kl_div_first5_element',
                        'real_action_tokens', 'real_action_tokens_first5',
                    ]:
                        if k in bon_info and isinstance(bon_info[k], list):
                            bon_info[k] = np.array(bon_info[k])
                    # top-k 디버그 정보 복원: JSON에서 dict 키가 문자열이므로 int로 변환
                    if 'topk_per_candidate_per_step' in bon_info and isinstance(bon_info['topk_per_candidate_per_step'], list):
                        restored_topk = []
                        for cand_steps in bon_info['topk_per_candidate_per_step']:
                            step_list = []
                            for step_dict in cand_steps:
                                if isinstance(step_dict, dict):
                                    new_d = {int(k): float(v) for k, v in step_dict.items()}
                                    step_list.append(new_d)
                                else:
                                    step_list.append(step_dict)
                            restored_topk.append(step_list)
                        bon_info['topk_per_candidate_per_step'] = restored_topk
                except Exception:
                    bon_info = None
            
            # is_success 복원
            is_success = None
            if is_success_raw is not None and is_success_raw != '':
                if isinstance(is_success_raw, str):
                    is_success = is_success_raw.strip().lower() in ['true', '1', 'yes']
                else:
                    is_success = bool(is_success_raw)
            
            # 로드한 데이터의 shape 확인 (best_action 로딩 이후로 이동)
            assert len(action_list.shape) == 3, f"action_list shape {action_list.shape}가 예상과 다릅니다. 예상: (N, H, 12)"
            assert action_list.shape[-1] == 12, f"action_list shape {action_list.shape}의 마지막 차원이 예상과 다릅니다. 예상: 12"
            assert len(scores_list.shape) == 1, f"scores_list shape {scores_list.shape}가 예상과 다릅니다. 예상: (N,)"
            if best_action is not None:
                assert len(best_action.shape) == 2, f"best_action shape {best_action.shape}가 예상과 다릅니다. 예상: (H, 12)"
                assert best_action.shape[1] == 12 and best_action.shape[0] == action_list.shape[1], f"best_action shape {best_action.shape}가 예상과 다릅니다. 예상: ({action_list.shape[1]}, 12)"
            assert action_list.shape[0] == len(scores_list), f"action_list의 candidate 수 {action_list.shape[0]}와 scores_list 길이 {len(scores_list)}가 일치하지 않습니다"
            
            step_data = {
                'step': step,
                'scores_list': scores_list, #(N,)
                'action_list': action_list, #(N,16,12)
                'best_index': best_index, # Optional[int]
                'best_score': best_score, # Optional[float]
                'best_action': best_action,  #(16,12) or None
                'bon_info': bon_info,
                'is_success': is_success,
            }
            
            episode_data[episode_idx].append(step_data)
    
    # 각 episode 내에서 step 순으로 정렬
    for episode_idx in episode_data:
        episode_data[episode_idx].sort(key=lambda x: x['step'])
    
    return episode_data


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


def choose_best_index_from_bon(step_data: Dict[str, Any], confidence_type: str) -> Optional[int]:
    """bon_info와 confidence_type에 따라 best index 선택"""
    bon_info = step_data.get('bon_info', None)
    if bon_info is None:
        return step_data.get('best_index', None)
    if confidence_type == 'log_likelihood':
        arr = bon_info.get('likelihood_first5', None)
        if arr is None:
            return step_data.get('best_index', None)
        return int(np.argmax(arr))
    elif confidence_type == 'negative_entropy':
        arr = bon_info.get('entropy_first5', None)
        if arr is None:
            return step_data.get('best_index', None)
        # maximize -entropy -> minimize entropy
        return int(np.argmin(arr))
    elif confidence_type == 'kl_div':
        arr = bon_info.get('kl_div_first5', None)
        if arr is None:
            return step_data.get('best_index', None)
        return int(np.argmax(arr))
    else:
        return step_data.get('best_index', None)


@dataclasses.dataclass
class Args:
    
    # Setup
    env_name: str = "<ENV_NAME>"  # Name of the environment to run. -> Task Name
    video_dir: str = "/virtual_lab/sjw_alinlab/suhyeok/openpi/output/robocasa/videos"
    tsv_path: str = ""  # Path to best_of_n_data.tsv file
    episode_indices: List[int] = dataclasses.field(default_factory=list)  # 실행할 episode 번호들 (비어있으면 전체)
    step_indices: List[int] = dataclasses.field(default_factory=lambda: [])  # 추가: 특정 step만 실행
    seed: int = 7
    max_episode_steps: int = 750

    # 모드 및 선택 기준
    mode: str = "candidate"  # "best" 또는 "candidate"
    confidence_type: str = "log_likelihood"  # "log_likelihood", "negative_entropy", "kl_div"
    fail_episode_only: bool = False  # True면 실패한 episode만 실행

    # Robocasa env parameters
    controller: Optional[str] = None
    robots: List[str] = dataclasses.field(default_factory=lambda: ["PandaOmron"])
    config: str = "single-arm-opposed"
    arm: str = "right"
    obj_groups: Optional[List[str]] = None
    layout: List[int] = dataclasses.field(default_factory=lambda: [-1])
    style: List[int] = dataclasses.field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 11])
    
    generative_textures: bool = True

    # 추가: 실행할 candidate 인덱스 지정 (비어있으면 모든 candidate 실행)
    candidate_indices: List[int] = dataclasses.field(default_factory=lambda: [])


def create_rollout_video(args: Args) -> None:
    # Set random seed (함수 시작 시 한 번만)
    np.random.seed(args.seed)
    
    # logging 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # TSV 데이터 로드
    if not os.path.exists(args.tsv_path):
        raise FileNotFoundError(f"TSV 파일을 찾을 수 없습니다: {args.tsv_path}")

    episode_data = load_tsv_data(args.tsv_path)
    logging.info(f"로드된 episode 수: {len(episode_data)}")

    # TSV에서 action_horizon 추출
    action_horizon = 16  # 기본값
    if episode_data:
        first_episode = list(episode_data.values())[0]
        if first_episode:
            first_step = first_episode[0]
            if first_step['action_list'] is not None and len(first_step['action_list']) > 0:
                # action_list는 이미 numpy array로 복원되었으므로 그대로 사용
                first_action = first_step['action_list'][0]
                
                if hasattr(first_action, 'shape') and len(first_action.shape) >= 2:
                    action_horizon = first_action.shape[0]
                    logging.info(f"TSV에서 추출한 action_horizon: {action_horizon}")

    # 실행할 에피소드 인덱스 확정 (비어있거나 -1 포함 시 전체)
    if args.episode_indices is None or len(args.episode_indices) == 0 or (len(args.episode_indices) == 1 and args.episode_indices[0] == -1):
        target_episode_indices = sorted(list(episode_data.keys()))
    else:
        target_episode_indices = args.episode_indices

    # 통계 저장용 (episode별로 저장)
    all_episode_stats = {}
    all_rollout_results = {}

    logging.info(f"Starting rollout video generation for {args.env_name}...")

    for episode_idx in target_episode_indices:
        if episode_idx not in episode_data:
            logging.warning(f"Episode {episode_idx} 데이터가 없습니다. 건너뜁니다.")
            continue
        episode_steps = episode_data[episode_idx]
        if not episode_steps:
            logging.warning(f"Episode {episode_idx}에 step 데이터가 없습니다. 건너뜁니다.")
            continue
        # 실패 에피소드만 실행 옵션 처리 (첫 step만 확인)
        if args.fail_episode_only:
            first_step_success = episode_steps[0].get('is_success', None)
            if first_step_success is True:
                logging.info(f"Episode {episode_idx}는 성공 에피소드이므로 skip (fail_episode_only=True)")
                continue
        
        # step_indices가 지정된 경우 해당 step만 처리
        if args.step_indices:
            target_step_indices = [i for i in args.step_indices if i < len(episode_steps)]
            logging.info(f"Episode {episode_idx} - 지정된 step들: {target_step_indices}")
        else:
            target_step_indices = list(range(len(episode_steps)))
            logging.info(f"Episode {episode_idx} - 모든 step 실행")
        episode_stats = defaultdict(list)
        episode_rollout_results = {}
        
        if args.mode == 'best':
            # 전체 에피소드를 bon 기반 best로 실행
            base_dir = Path(args.video_dir) / args.mode / args.confidence_type
            video_save_dir = base_dir / f"ep{episode_idx}"
            name_prefix = "best"
            env = load_robocasa_gym_env(
                args.env_name,
                seed=args.seed,
                robots=args.robots,
                camera_widths=256,
                camera_heights=256,
                render_onscreen=False,
                obj_instance_split="A",
                generative_textures="100p" if args.generative_textures else None,
                randomize_cameras=False,
                layout_ids=args.layout,
                style_ids=args.style,
                collect_data=False,
            )
            env = RoboCasaWrapper(env)
            env = RecordVideo(
                env,
                video_save_dir,
                disable_logger=True,
                episode_trigger=lambda eid: eid == episode_idx,
                fps=20,
                name_prefix=name_prefix,
            )
            env = MultiStepWrapper(
                env,
                video_delta_indices=np.arange(1),
                state_delta_indices=np.arange(1),
                n_action_steps=action_horizon,
                max_episode_steps=args.max_episode_steps,
            )
            # reset: episode_idx+1번 반복, 마지막에 rollout
            for reset_episode_idx in range(episode_idx + 1):
                obs, info = env.reset()
                if reset_episode_idx == episode_idx:
                    if hasattr(env, 'set_episode_number'):
                        env.set_episode_number(episode_idx)
                    done = False
                    step = 0
            # rollout: 모든 step을 bon 기반 best로 실행
            for rollout_step_idx in range(len(episode_steps)):
                step_data = episode_steps[rollout_step_idx]
                best_idx = choose_best_index_from_bon(step_data, args.confidence_type)
                if best_idx is None:
                    logging.warning(f"Episode {episode_idx} Step {rollout_step_idx}: best index를 찾지 못해 0으로 대체")
                    best_idx = 0
                action = step_data['action_list'][best_idx]
                action_dict = action_to_dict(action)
                obs, reward, terminated, truncated, info = env.step(action_dict)
                obs["video.left_view"] = np.flip(obs["video.left_view"], axis=1)
                obs["video.right_view"] = np.flip(obs["video.right_view"], axis=1)
                obs["video.wrist_view"] = np.flip(obs["video.wrist_view"], axis=1)
                done = terminated or truncated
                step += action_horizon
                if done:
                    break
            final_success = info.get('is_success', False)
            episode_rollout_results[f"episode_{episode_idx}_best_{args.confidence_type}"] = {
                'episode_idx': episode_idx,
                'mode': 'best',
                'success': final_success,
                'total_steps': len(episode_steps)
            }
            add_to(episode_stats, flatten({"is_success": final_success}))
            env.close()
            logging.info(f"Episode {episode_idx} (best mode) 완료")
        else:
            # candidate 모드: 기존 기능 유지 (target step 전까지는 best_action, target step에서만 candidate 실행)
            for step_idx in target_step_indices:
                step_data_cur = episode_steps[step_idx]
                action_list = step_data_cur['action_list']
                num_candidates = len(action_list)
                # 실행할 candidate 목록 확정
                if args.candidate_indices:
                    candidate_indices_to_run = [ci for ci in args.candidate_indices if 0 <= ci < num_candidates]
                    logging.info(f"Step {step_idx}에서 지정된 {len(candidate_indices_to_run)}/{num_candidates}개 candidate 실행 예정: {candidate_indices_to_run}")
                    if len(candidate_indices_to_run) == 0:
                        logging.warning(f"Step {step_idx}: 유효한 candidate 인덱스가 없습니다. 건너뜁니다.")
                        continue
                else:
                    candidate_indices_to_run = list(range(num_candidates))
                    logging.info(f"Step {step_idx}에서 모든 {num_candidates}개 candidate 실행 예정")
                for candidate_idx in candidate_indices_to_run:
                    logging.info(f"Episode {episode_idx}, Step {step_idx}, Candidate {candidate_idx} 시작")
                    # score 계산
                    candidate_score = step_data_cur['scores_list'][candidate_idx] if 'scores_list' in step_data_cur and step_data_cur['scores_list'] is not None else None
                    # best candidate 여부 판단
                    is_best = False
                    if step_data_cur.get('best_index', None) is not None:
                        is_best = (candidate_idx == step_data_cur['best_index'])
                    # likelihood 기반 best 여부 판단
                    is_likelihood_best = False
                    bon_info = step_data_cur.get('bon_info', None)
                    if bon_info is not None:
                        arr = bon_info.get('likelihood_first5', None)
                        if arr is not None:
                            try:
                                likelihood_best_idx = int(np.argmax(arr))
                                is_likelihood_best = (candidate_idx == likelihood_best_idx)
                            except Exception:
                                is_likelihood_best = False
                    # 하위 폴더 구조 생성 및 파일명 구성
                    base_dir = Path(args.video_dir) / "candidate"
                    subdir = f"ep{episode_idx}/step{step_idx}"
                    cand_name = f"cand{candidate_idx}-score"
                    if candidate_score is not None:
                        cand_name += f"{float(candidate_score):.4f}"
                    if is_best and is_likelihood_best:
                        cand_name += "_best_likelihood_best"
                    elif is_best:
                        cand_name += "_best"
                    elif is_likelihood_best:
                        cand_name += "_likelihood_best"
                    video_save_dir = base_dir / subdir
                    name_prefix = cand_name
                    env = load_robocasa_gym_env(
                        args.env_name,
                        seed=args.seed,
                        robots=args.robots,
                        camera_widths=256,
                        camera_heights=256,
                        render_onscreen=False,
                        obj_instance_split="A",
                        generative_textures="100p" if args.generative_textures else None,
                        randomize_cameras=False,
                        layout_ids=args.layout,
                        style_ids=args.style,
                        collect_data=False,
                    )
                    env = RoboCasaWrapper(env)
                    env = RecordVideo(
                        env,
                        video_save_dir,
                        disable_logger=True,
                        episode_trigger=lambda eid: eid == episode_idx,
                        fps=20,
                        name_prefix=name_prefix,
                    )
                    env = MultiStepWrapper(
                        env,
                        video_delta_indices=np.arange(1),
                        state_delta_indices=np.arange(1),
                        n_action_steps=action_horizon,
                        max_episode_steps=args.max_episode_steps,
                    )
                    # reset: episode_idx+1번 반복, 마지막에 rollout
                    for reset_episode_idx in range(episode_idx + 1):
                        obs, info = env.reset()
                        if reset_episode_idx == episode_idx:
                            if hasattr(env, 'set_episode_number'):
                                env.set_episode_number(episode_idx)
                            done = False
                            step = 0
                    # rollout: target step까지 best action, target step에서만 candidate action
                    for rollout_step_idx, rollout_step_data in enumerate(episode_steps):
                        if rollout_step_idx < step_idx:
                            # best_action 우선 사용, 없으면 bon 기반 선택, 그것도 없으면 best_index 기반
                            prev_action = rollout_step_data.get('best_action', None)
                            if prev_action is None:
                                best_idx_prev = choose_best_index_from_bon(rollout_step_data, args.confidence_type)
                                if best_idx_prev is None:
                                    best_idx_prev = rollout_step_data.get('best_index', 0) or 0
                                prev_action = rollout_step_data['action_list'][best_idx_prev]
                            action = prev_action
                        elif rollout_step_idx == step_idx:
                            action = rollout_step_data['action_list'][candidate_idx]
                        else:
                            break
                        action_dict = action_to_dict(action)
                        obs, reward, terminated, truncated, info = env.step(action_dict)
                        obs["video.left_view"] = np.flip(obs["video.left_view"], axis=1)
                        obs["video.right_view"] = np.flip(obs["video.right_view"], axis=1)
                        obs["video.wrist_view"] = np.flip(obs["video.wrist_view"], axis=1)
                        done = terminated or truncated
                        step += action_horizon
                        if done:
                            break
                    final_success = info.get('is_success', False)
                    key_name = f"episode_{episode_idx}_step_{step_idx}_candidate_{candidate_idx}"
                    episode_rollout_results[key_name] = {
                        'episode_idx': episode_idx,
                        'step_idx': step_idx,
                        'candidate_idx': candidate_idx,
                        'success': final_success,
                        'total_steps': len(episode_steps)
                    }
                    if candidate_score is not None:
                        episode_rollout_results[key_name]['score'] = float(candidate_score)
                    add_to(episode_stats, flatten({"is_success": final_success}))
                    env.close()
                    logging.info(f"Episode {episode_idx}, Step {step_idx}, Candidate {candidate_idx} 완료")
        # episode별 통계 계산 및 저장
        keys_to_process = list(episode_stats.keys())
        additional_stats = {}
        for k in keys_to_process:
            try:
                if k == "is_success":
                    success_count = sum(1 for x in episode_stats[k] if x)
                    total_count = len(episode_stats[k])
                    if total_count > 0:
                        episode_stats[k] = success_count / total_count
                        additional_stats[f"{k}_count"] = f"{success_count}/{total_count}"
                    else:
                        episode_stats[k] = 0.0
                        additional_stats[f"{k}_count"] = "0/0"
                else:
                    if len(episode_stats[k]) > 0:
                        episode_stats[k] = float(np.mean(episode_stats[k]))
                    else:
                        episode_stats[k] = 0.0
            except Exception as e:
                logging.warning(f"Episode {episode_idx} 통계 계산 에러 for {k}: {e}")
                episode_stats[k] = 0.0
        episode_stats.update(additional_stats)
        
        # episode별 통계 저장
        all_episode_stats[episode_idx] = episode_stats
        all_rollout_results[episode_idx] = episode_rollout_results
        
        # episode별 summary 출력
        success_rate = episode_stats.get('is_success', 0.0)
        success_count = additional_stats.get('is_success_count', '0/0')
        avg_score = episode_stats.get('scores', 0.0)
        
        logging.info(f"Episode {episode_idx} 완료!")
        logging.info(f"Success Rate: {success_rate:.2f} ({success_count})")
        logging.info(f"Average Score: {avg_score:.4f}")

        # 각 episode별로 summary 파일에 저장 (append 모드)
        with open(f"{args.video_dir}/rollout_summary.txt", "a") as f:
            f.write(f"\nRollout Summary for Episode {episode_idx}:\n")
            f.write("=" * 50 + "\n")
            f.write(f"Episode {episode_idx} Success Rate: {success_rate:.2f} ({success_count})\n")
            f.write(f"Episode {episode_idx} Average Score: {avg_score:.4f}\n\n")
            
            f.write(f"Episode {episode_idx} Details:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Success Rate: {success_rate:.2f} ({success_count})\n")
            f.write(f"Average Score: {avg_score:.4f}\n\n")
            
            f.write("Individual Results:\n")
            for key, result in episode_rollout_results.items():
                if 'score' in result:
                    f.write(f"  {key}: Success={result['success']}, Score={result['score']:.4f}\n")
                else:
                    f.write(f"  {key}: Success={result['success']}\n")
            f.write("\n" + "="*50 + "\n")

    logging.info(f"전체 Rollout 완료!")
    logging.info(f"결과가 {args.video_dir}에 저장되었습니다.")


if __name__ == "__main__":
    tyro.cli(create_rollout_video) 