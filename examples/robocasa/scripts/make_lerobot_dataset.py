import os
import json
import pandas as pd
from glob import glob
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import tqdm

# 1. 모든 json chunk 파일 불러오기
json_dir = "/virtual_lab/sjw_alinlab/RDG/datasets/lerobot_robocasa/robocasa_kitchen_24tasks_60demos_dpo/json/preference_dataset_sample:1.0"
json_files = sorted(glob(os.path.join(json_dir, "preference_dataset_chunk_*.json")))

all_frames = []
for file in json_files:
    with open(file, "r") as f:
        chunk = json.load(f)
        all_frames.extend(chunk)

# info.json과 dtype/shape 매칭을 위한 가공 함수
def fix_types_and_shapes(df):
    # float32 (1,) → float32
    df['timestamp'] = df['timestamp'].apply(lambda x: np.array(x, dtype=np.float32).reshape(1))
    # int64 (1,) → int64
    for col in ['frame_index', 'episode_index', 'index', 'task_index']:
        df[col] = df[col].apply(lambda x: np.array(x, dtype=np.int64).reshape(1))
    # observation.state: float64 (53,)
    df['observation.state'] = df['observation.state'].apply(lambda x: np.array(x, dtype=np.float64).reshape(53))
    # positive_action: float64 (12,)
    df['positive_action'] = df['positive_action'].apply(lambda x: np.array(x, dtype=np.float64).reshape(12))
    # negative_action: float64 (12,) or None → None이면 positive_action 복사
    def fix_neg(row):
        x = row['negative_action']
        #NOTE : to remove preference, set negative_action to positive_action
        if x is None:
            return np.array(row['positive_action'], dtype=np.float64).reshape(12)
        return np.array(x, dtype=np.float64).reshape(12)
    df['negative_action'] = df.apply(fix_neg, axis=1)
    return df

# 2. DataFrame으로 변환
# nan(null) 값이 있는 행은 모두 제외

df = pd.DataFrame(all_frames)
print(f"초기 데이터셋 샘플 수: {len(df)}")
nan_rows = df.isnull().any(axis=1).sum()
print(f"nan(null) 값이 있는 행의 수: {nan_rows}")
# df = df.dropna().reset_index(drop=True)  # <- 이 줄을 주석처리하여 결측치 row를 살림
print(f"nan drop 후 최종 남은 데이터셋 샘플 수: {len(df)}")

# episode_index를 int로 변환 (hashable하게, nunique/unique 에러 방지)
df['episode_index'] = df['episode_index'].apply(lambda x: int(x[0]) if isinstance(x, (np.ndarray, list)) else int(x))
episode_count = df['episode_index'].nunique()
print(f"전체 episode 개수: {episode_count} (생성될 parquet 파일 개수)")

df = fix_types_and_shapes(df)

# timestamp, frame_index, episode_index, index, task_index 모두 List/array 제거 (스칼라로 변환)
for col, typ in zip(['timestamp', 'frame_index', 'episode_index', 'index', 'task_index'], [float, int, int, int, int]):
    df[col] = df[col].apply(lambda x: typ(x[0]) if isinstance(x, (np.ndarray, list)) else typ(x))

df['timestamp'] = df['timestamp'].astype(np.float32)

# 3. "index" 컬럼 기준으로 정렬
df = df.sort_values("index").reset_index(drop=True)

# 원하는 column 순서로 재정렬
ordered_cols = [
    'observation.state',
    'positive_action',
    'negative_action',
    'timestamp',
    'frame_index',
    'episode_index',
    'index',
    'task_index',
]
df = df[ordered_cols]

# episode_index, frame_index 기준으로 정렬 후 index 재할당
df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
df["index"] = np.arange(len(df))

# 원하는 column 순서로 다시 맞춤 (index 값이 바뀌었으므로)
df = df[ordered_cols]

# 전체 데이터셋 row 개수 출력
print(f"전체 데이터셋 샘플 수: {len(df)}")

# 4. episode_index별로 그룹화 및 parquet 저장 (진행률 표시)
chunk_size = 1000  # 한 chunk에 들어갈 episode 개수

base_out_dir = "/virtual_lab/sjw_alinlab/RDG/datasets/lerobot_robocasa/robocasa_kitchen_24tasks_60demos_dpo/data"
episode_indices = sorted(df['episode_index'].unique())
for i, (episode_idx, group) in enumerate(tqdm.tqdm(df.groupby("episode_index"), total=len(episode_indices), desc="Saving parquet by episode")):
    chunk_id = i // chunk_size
    chunk_dir = os.path.join(base_out_dir, f"chunk-{chunk_id:03d}")
    os.makedirs(chunk_dir, exist_ok=True)
    episode_str = f"{int(episode_idx):06d}"
    out_path = os.path.join(chunk_dir, f"episode_{episode_str}.parquet")
    table = pa.Table.from_pandas(group.reset_index(drop=True))
    pq.write_table(table, out_path)

# === info.json 자동 갱신 ===
meta_dir = os.path.join(os.path.dirname(base_out_dir), "meta")
info_path = os.path.join(meta_dir, "info.json")

# 1. parquet/비디오 파일 정보
parquet_files = glob(os.path.join(base_out_dir, "chunk-*", "episode_*.parquet"))
total_episodes = len(parquet_files)
total_chunks = len(glob(os.path.join(base_out_dir, "chunk-*")))
video_dir = os.path.join(os.path.dirname(base_out_dir), "videos")
video_files = glob(os.path.join(video_dir, "chunk-*", "*", "episode_*.mp4"))
total_videos = len(video_files)

# 2. 전체 프레임 수
frame_count = 0
task_indices = set()
for f in tqdm.tqdm(parquet_files, desc="Counting total frames & unique tasks"):
    df_tmp = pd.read_parquet(f, columns=["timestamp", "task_index"])
    frame_count += len(df_tmp)
    if "task_index" in df_tmp:
        task_indices.update(df_tmp["task_index"].unique())
total_frames = frame_count
total_tasks = len(task_indices)

# 3. splits
splits = {"train": f"0:{total_episodes}"}

# 4. features
sample_parquet = parquet_files[0]
df_sample = pd.read_parquet(sample_parquet)
parquet_cols = list(df_sample.columns)

reference_info_path = "/virtual_lab/sjw_alinlab/RDG/datasets/lerobot_robocasa/robocasa_kitchen_24tasks_60demos/meta/info.json"

with open(reference_info_path, "r") as f:
    info = json.load(f)
old_features = info.get("features", {})
features = {}

# observation.state (기존 부가정보 유지)
old = old_features.get("observation.state", {})
features["observation.state"] = dict(old)

sample = df_sample["observation.state"].iloc[0]
features["observation.state"]["dtype"] = str(sample.dtype)
features["observation.state"]["shape"] = list(sample.shape)

# positive_action, negative_action
for act in ["positive_action", "negative_action"]:
    old = old_features.get("action", {})
    features[act] = dict(old)
    sample = df_sample[act].iloc[0]
    features[act]["dtype"] = str(sample.dtype)
    features[act]["shape"] = list(sample.shape)
    
features.update({k: v for k, v in old_features.items() if k.startswith("observation.images")})
        
# 기타 컬럼
for col in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
    old = old_features.get(col, {})
    features[col] = dict(old)
    features[col]["dtype"] = str(df_sample[col].dtype)
    features[col]["shape"] = [1]
    
        
# info.json 갱신
info.update({
    "total_episodes": total_episodes,
    "total_frames": total_frames,
    "total_videos": total_videos,
    "total_chunks": total_chunks,
    "total_tasks": total_tasks,
    "splits": splits,
    "features": features
})

with open(info_path, "w") as f:
    json.dump(info, f, indent=2)
print(f"info.json 갱신 완료: {info_path}")