import os
import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gr00t.eval.wrappers.robocasa_wrapper import load_robocasa_gym_env, RoboCasaWrapper

def extract_prompts(
    env_name: str,
    n_episodes: int = 50,
    seed: int = 7,
    robots = ["PandaOmron"],
    layout = [-1],
    style = [0,1,2,3,4,5,6,7,8,11],
):
    # 환경 생성
    env = load_robocasa_gym_env(
        env_name,
        seed=seed,
        robots=robots,
        camera_widths=256,
        camera_heights=256,
        render_onscreen=False,
        obj_instance_split="A",
        generative_textures="100p",
        randomize_cameras=False,
        layout_ids=layout,
        style_ids=style,
        collect_data=False,
    )
    env = RoboCasaWrapper(env)

    prompts = []
    for i in range(n_episodes):
        obs, info = env.reset()
        # prompt 추출 (annotation.human.action.task_description)
        prompt = obs["annotation.human.action.task_description"][0]
        prompts.append(prompt)
        print(f"Episode {i}: {prompt}")

    env.close()
    return prompts

if __name__ == "__main__":
    # 예시 실행
    prompts = extract_prompts(
        env_name="PnPCounterToSink",  # 원하는 환경 이름으로 변경
        n_episodes=50,
        seed=7,
    )

    with open("/virtual_lab/sjw_alinlab/suhyeok/openpi/examples/robocasa/scripts/prompts.txt", "w") as f:
        for p in prompts:
            f.write(p + "\n")