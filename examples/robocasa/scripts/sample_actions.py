import dataclasses
import logging
import platform
import time
import random
import numpy as np
import pandas as pd
import os
import json
import shutil
from typing import Dict, List, Tuple
import tyro
import jax
import jax.numpy as jnp
import tqdm_loggable.auto as tqdm
import functools
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import openpi.policies.policy as _policy
import openpi.policies.policy_config as _policy_config
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding

@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""
    config: str
    dir: str
    best_of_n: int = 2  # DPO용으로 2개 action 생성
    temperature: float = 0.7
    seed: int = 7

@dataclasses.dataclass
class Args:
    # Policy parameters
    policy: Checkpoint
    
    # Dataset parameters
    source_dataset_path: str = "/virtual_lab/sjw_alinlab/RDG/datasets/lerobot_robocasa/robocasa_kitchen_24tasks_60demos"
    output_dataset_path: str = "/virtual_lab/sjw_alinlab/RDG/datasets/lerobot_robocasa/robocasa_kitchen_24tasks_60demos_dpo"
    
    # Sampling parameters
    num_samples: int = 10000  # 랜덤 샘플링할 개수
    
    # Data loading parameters
    num_workers: int = 16
    batch_size: int = 1
    
    # If provided, will be used in case the "prompt" key is not present in the data
    default_prompt: str | None = None


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    return _policy_config.create_trained_policy(
        _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt,
        sample_kwargs={"temperature": args.policy.temperature, "best_of_n": args.policy.best_of_n},
        seed=args.policy.seed,
    )

def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def sample_random_frames(all_frames: List[Dict], num_samples: int, seed: int) -> List[Dict]:
    """전체 프레임에서 랜덤하게 샘플링합니다."""
    random.seed(seed)
    
    # episode별로 그룹화하여 action_horizon=16을 고려한 샘플링
    episode_groups = {}
    for frame in all_frames:
        episode_idx = frame['episode_index']
        if episode_idx not in episode_groups:
            episode_groups[episode_idx] = []
        episode_groups[episode_idx].append(frame)
    
    # 각 episode에서 마지막 15개 프레임 제외 (16개 연속 프레임이 필요하므로)
    valid_frames = []
    for episode_frames in episode_groups.values():
        if len(episode_frames) > 15:
            valid_frames.extend(episode_frames[:-15])
        else:
            valid_frames.extend(episode_frames)
    
    # 랜덤 샘플링
    sampled_indices = random.sample(range(len(valid_frames)), min(num_samples, len(valid_frames)))
    sampled_frames = [valid_frames[i] for i in sampled_indices]
    
    return sampled_frames


def run_model_inference(model: _policy.Policy, sampled_frames: List[Dict], 
                       all_frames: List[Dict]) -> pd.DataFrame:
    """모델 추론을 실행하여 negative actions를 생성합니다."""
    results = []
    
    for frame_info in tqdm.tqdm(sampled_frames, desc="Running model inference"):
        try:
            obs_dict = frame_info['observation']
            episode_idx = frame_info['episode_index']
            frame_idx = frame_info['frame_index']
            
            # 모델 추론 실행 (best_of_n=2)
            outputs = model.infer(obs=obs_dict, action_tokens=None)
            
            pred_actions = outputs['actions']  # (2, 16, 12) 형태
            
            action_horizon = pred_actions.shape[1]
            
            assert pred_actions.shape == (2, action_horizon, 12)
            
            # negative_action_1, negative_action_2로 저장 (16개 timestep의 action)
            negative_action_1 = pred_actions[0, :, :].tolist()  # (16, 12)
            negative_action_2 = pred_actions[1, :, :].tolist()  # (16, 12)
            
            # t~t+action_horizon 시점의 모든 frame에 대해 결과 저장
            for t in range(action_horizon):
                target_frame_idx = frame_idx + t
                
                # 같은 episode에서 target frame 찾기
                target_frame = None
                for frame in all_frames:
                    if frame['episode_index'] == episode_idx and frame['frame_index'] == target_frame_idx:
                        target_frame = frame
                        break
                
                if target_frame is None:
                    continue  # episode 범위를 벗어나면 스킵
                
                # 각 timestep에 해당하는 action 저장
                result = {
                    'timestamp': target_frame['timestamp'],
                    'frame_index': target_frame['frame_index'],
                    'episode_index': target_frame['episode_index'],
                    'index': target_frame['index'],
                    'task_index': target_frame['task_index'],
                    'negative_action_1': negative_action_1[t],  # t번째 timestep의 action
                    'negative_action_2': negative_action_2[t],  # t번째 timestep의 action
                    'action_timestep': t  # 0~15 중 어느 timestep인지 표시
                }
                
                results.append(result)
            
        except Exception as e:
            logging.error(f"Error in inference for frame {frame_info['frame_index']} in episode {frame_info['episode_index']}: {e}")
            continue
    
    return pd.DataFrame(results)


def merge_with_full_dataset(model_results: pd.DataFrame, all_frames: List[Dict]) -> List[Dict]:
    """모델 결과와 전체 데이터셋을 병합합니다."""
    # 모델 결과를 (episode_index, frame_index)로 인덱싱
    model_results_dict = {}
    for _, row in model_results.iterrows():
        key = (row['episode_index'], row['frame_index'])
        if key not in model_results_dict:
            model_results_dict[key] = []
        model_results_dict[key].append(row.to_dict())
    
    # 전체 데이터셋과 병합
    merged_data = []
    for frame_info in all_frames:
        key = (frame_info['episode_index'], frame_info['frame_index'])
        if key in model_results_dict:
            # 모델 결과가 있는 프레임만 포함
            for result in model_results_dict[key]:
                merged_frame = {
                    'timestamp': frame_info['timestamp'],
                    'frame_index': frame_info['frame_index'],
                    'episode_index': frame_info['episode_index'],
                    'index': frame_info['index'],
                    'task_index': frame_info['task_index'],
                    'observation': frame_info['observation'],
                    'positive_action': frame_info['gt_actions'],  # GT action을 positive_action으로
                    'negative_action_1': result['negative_action_1'],
                    'negative_action_2': result['negative_action_2'],
                    'action_timestep': result['action_timestep']
                }
                merged_data.append(merged_frame)
    
    return merged_data


def reindex_episodes_and_tasks(merged_data: List[Dict]) -> Tuple[List[Dict], Dict, Dict]:
    """episode_index와 task_index를 0부터 순차적으로 재정렬합니다."""
    # episode_index 재정렬
    unique_episodes = sorted(set(frame['episode_index'] for frame in merged_data))
    episode_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_episodes)}
    
    # task_index 재정렬
    unique_tasks = sorted(set(frame['task_index'] for frame in merged_data))
    task_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_tasks)}
    
    # 데이터 재정렬
    reindexed_data = []
    for frame in merged_data:
        new_frame = frame.copy()
        new_frame['episode_index'] = episode_mapping[frame['episode_index']]
        new_frame['task_index'] = task_mapping[frame['task_index']]
        reindexed_data.append(new_frame)
    
    return reindexed_data, episode_mapping, task_mapping


def save_episode_parquets(merged_data: List[Dict], output_path: str):
    """DPO 데이터를 episode별 parquet 파일로 저장합니다."""
    data_dir = os.path.join(output_path, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # episode별로 그룹화
    episode_groups = {}
    for frame in merged_data:
        episode_idx = frame['episode_index']
        if episode_idx not in episode_groups:
            episode_groups[episode_idx] = []
        episode_groups[episode_idx].append(frame)
    
    # 각 episode별로 parquet 파일 저장
    for episode_idx, episode_frames in tqdm.tqdm(episode_groups.items(), desc="Saving episode parquets"):
        chunk_idx = episode_idx // 1000
        chunk_path = os.path.join(data_dir, f"chunk-{chunk_idx:03d}")
        os.makedirs(chunk_path, exist_ok=True)
        
        # DataFrame으로 변환
        df_data = []
        for frame in episode_frames:
            row = {
                'timestamp': frame['timestamp'],
                'frame_index': frame['frame_index'],
                'episode_index': frame['episode_index'],
                'index': frame['index'],
                'task_index': frame['task_index'],
                'positive_action': frame['positive_action'],
                'negative_action_1': frame['negative_action_1'],
                'negative_action_2': frame['negative_action_2'],
                'action_timestep': frame['action_timestep']
            }
            # observation의 모든 키들을 추가
            for key, value in frame['observation'].items():
                row[f'observation.{key}'] = value
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # episode별 parquet 파일 저장
        episode_file = os.path.join(chunk_path, f"episode_{episode_idx:06d}.parquet")
        df.to_parquet(episode_file, index=False)
    
    logging.info(f"Saved {len(episode_groups)} episode parquet files")


def update_meta_files(output_path: str, merged_data: List[Dict]):
    """DPO 데이터셋 메타 정보를 생성합니다."""
    meta_dir = os.path.join(output_path, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    
    # DPO 데이터셋 정보 생성
    info_data = {
        "dataset_type": "dpo",
        "total_frames": len(merged_data),
        "features": {
            "positive_action": {
                "dtype": "float64",
                "shape": [12],
                "description": "Ground truth action"
            },
            "negative_action_1": {
                "dtype": "float64", 
                "shape": [12],
                "description": "Model generated negative action 1"
            },
            "negative_action_2": {
                "dtype": "float64",
                "shape": [12], 
                "description": "Model generated negative action 2"
            }
        }
    }
    
    # info.json 저장
    output_info_file = os.path.join(meta_dir, "info.json")
    with open(output_info_file, 'w') as f:
        json.dump(info_data, f, indent=2)
    
    logging.info(f"Created DPO dataset meta info with {len(merged_data)} frames")





def process_sample(i, dataset):
    sample = dataset[i]
    row = {}
    for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        val = sample.get(key, None)
        if val is not None:
            if key == "timestamp":
                row[key] = np.array(val, dtype=np.float32).reshape(-1)
            else:
                row[key] = np.array(val, dtype=np.int64).reshape(-1)
    row["observation.state"] = np.array(sample.get("observation.state", np.zeros(53)), dtype=np.float64).reshape(53)
    row["action"] = np.array(sample.get("action"), dtype=np.float64)[0,:].reshape(12)
    return row


def main(args: Args):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    
    # Seed 설정
    random.seed(args.policy.seed)
    np.random.seed(args.policy.seed)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dataset_path, exist_ok=True)
    
    # 1. 모델 로드
    logging.info("Loading policy model...")
    model = create_policy(args)
    logging.info("Policy model loaded!")
    
    # 2. Config 로드
    logging.info("Loading config...")
    config = _config.get_config(args.policy.config)
    logging.info("Config loaded!")
    
    # 3. 전체 데이터셋 로드 (DataLoader로 순회하여 json 저장)
    logging.info("Loading full dataset with DataLoader and saving as JSON...")
    all_frames = get_all_frames_data_from_dataloader(config, args)
    json_path = os.path.join(args.output_dataset_path, "full_dataset.json")
    with open(json_path, "w") as f:
        import json
        json.dump(all_frames, f)
    logging.info(f"Saved full dataset as JSON: {json_path} (rows: {len(all_frames)})")

    # 4. 모델 추론용 Dataloader 생성 (shuffle, subset_size=10000)
    logging.info("Creating dataloader for model inference")
    
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    
    
    config = dataclasses.replace(config, 
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            subset_size=args.num_samples)  # 1000개 샘플만 사용
    
    dataloader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=True,
        subset_size=config.subset_size,
        num_batches=config.subset_size // args.batch_size,
    )
    
    data_size = len(dataloader._data_loader._dataset)
    logging.info(f"Data loader created with {data_size} samples")

    # 5. 모델 추론 및 결과 저장
    results = []
    for batch in tqdm.tqdm(dataloader, desc="Model inference loop"):
        observation, gt_actions, timestamp, frame_index, episode_index, index, task_index = batch
        # observation dict로 변환
        if hasattr(observation, 'to_dict'):
            obs_dict = observation.to_dict()
        else:
            obs_dict = observation
            
        import pdb; pdb.set_trace()
        
        # 모델 추론
        outputs = model.infer(obs=obs_dict, action_tokens=None)
        pred_actions = outputs['actions']  # (2, 16, 12) 형태
        action_horizon = pred_actions.shape[1]
        assert pred_actions.shape == (2, action_horizon, 12)
        negative_action_1 = pred_actions[0, :, :].tolist()
        negative_action_2 = pred_actions[1, :, :].tolist()
        # 각 timestep별로 저장
        for t in range(action_horizon):
            results.append({
                'timestamp': timestamp,
                'frame_index': frame_index,
                'episode_index': episode_index,
                'index': index,
                'task_index': task_index,
                'positive_action': gt_actions,
                'negative_action_1': negative_action_1[t],
                'negative_action_2': negative_action_2[t],
                'action_timestep': t,
            })
            
    # DataFrame 변환 및 parquet 저장
    results_df = pd.DataFrame(results)
    results_df.to_parquet(os.path.join(args.output_dataset_path, "dpo_inference_results.parquet"), index=False)
    logging.info(f"Saved DPO inference results with {len(results_df)} rows")

    # 6. 메타 파일 생성
    update_meta_files(args.output_dataset_path, results)
    logging.info(f"DPO dataset created successfully at: {args.output_dataset_path}")
    logging.info(f"Final dataset statistics:")
    logging.info(f"  - Total frames: {len(results_df)}")


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args) 