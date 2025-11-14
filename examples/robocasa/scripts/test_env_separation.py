#!/usr/bin/env python3

import os
import numpy as np

import robocasa
import robosuite
from robosuite.controllers import load_composite_controller_config

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gr00t.eval.wrappers.robocasa_wrapper_dev import load_robocasa_gym_env


def test_env_separation():
    """각 환경이 실제로 독립적인지 테스트"""
    
    print("=== 환경 분리 테스트 시작 ===")
    
    # 3개의 환경 생성
    n_envs = 3
    env = load_robocasa_gym_env(
        env_name="CloseDoubleDoor",
        n_envs=n_envs,
        seed=42,
        video_path="/virtual_lab/sjw_alinlab/suhyeok/openpi/output/test_videos",  # 테스트용 비디오 경로
    )
    
    print(f"생성된 환경 수: {n_envs}")
    
    # 초기화
    obs, _ = env.reset()
    print(f"초기 obs shape: {obs['video.left_view'].shape}")
    
    # 각 환경의 첫 번째 프레임을 캡처
    print("\n=== 각 환경의 첫 번째 프레임 캡처 ===")
    
    # 간단한 액션으로 몇 스텝 진행
    for step in range(5):
        print(f"\n--- Step {step} ---")
        
        # 랜덤 액션 생성
        action = env.action_space.sample()
        
        # 환경 스텝 실행
        obs, rewards, terminations, truncations, infos = env.step(action)
        
        # 각 환경의 상태 출력
        for env_idx in range(n_envs):
            success = infos["success"][env_idx][0] if "success" in infos else False
            reward = rewards[env_idx]
            print(f"  Env {env_idx}: reward={reward:.3f}, success={success}")
    
    env.close()
    print("\n=== 테스트 완료 ===")
    print("생성된 비디오 파일들을 확인하여 각 환경이 다른 내용을 보여주는지 확인하세요.")

if __name__ == "__main__":
    test_env_separation() 