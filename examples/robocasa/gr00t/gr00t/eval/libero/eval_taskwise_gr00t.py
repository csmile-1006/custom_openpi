import os
import collections
import dataclasses
import logging
import math
import pathlib
import imageio
import numpy as np
import tqdm
import tyro

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    task_idx: int = -1  # Task index to run

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}, idx: {args.task_idx}")
    logging.info(f"Number of tasks in suite: {num_tasks_in_suite}")
    logging.info(f"Number of trials per task: {args.num_trials_per_task}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial" or args.task_suite_name == "libero_spatial_new" or args.task_suite_name == "libero_spatial_ood":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object" or args.task_suite_name == "libero_object_ood":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal" or args.task_suite_name == "libero_goal_new" or args.task_suite_name == "libero_goal_ood":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = None

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        if args.task_idx != -1 and task_id != args.task_idx:
            continue
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")
            task_segment = task_description.replace(" ", "_")
            if os.path.exists(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx}_failure.mp4"
            ):
                logging.info(f"Video already exists, skipping episode {episode_idx}...")
                total_episodes += 1
                task_episodes += 1
                continue
            elif os.path.exists(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx}_success.mp4"
            ):
                logging.info(f"Video already exists, skipping episode {episode_idx}...")
                total_episodes += 1
                task_episodes += 1
                total_successes += 1
                task_successes += 1
                continue    
            if client is None:
                client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])


            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    
                    # No need to resize for GR00T 
                    # img = image_tools.convert_to_uint8(
                    #     image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    # )
                    # wrist_img = image_tools.convert_to_uint8(
                    #     image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    # )

                    #"""  # Save images for debugging
                    if t == args.num_steps_wait and episode_idx == 0:
                        task_segment = task_description.replace(" ", "_")
                        image_prefix = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_ep{episode_idx:02d}"
                        image_prefix.parent.mkdir(parents=True, exist_ok=True)
                        imageio.imwrite(f"{image_prefix}_img.png", img)
                        imageio.imwrite(f"{image_prefix}_wrist.png", wrist_img)
                    #"""

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict

                        # element = {
                        #     "observation/image": img,
                        #     "observation/wrist_image": wrist_img,
                        #     "observation/state": np.concatenate(
                        #         (
                        #             obs["robot0_eef_pos"],
                        #             _quat2axisangle(obs["robot0_eef_quat"]),
                        #             obs["robot0_gripper_qpos"],
                        #         )
                        #     ),
                        #     **({"previous_actions": prev_actions_vec,
                        #         "previous_state": prev_state_vec,
                        #         "observation/previous_image": prev_img,
                        #         "observation/previous_wrist_image": prev_wrist_img} if args.use_previous_info else {}),
                        #     "prompt": str(task_description),
                        # }
                        element = {
                            "video.front_view": np.array([img]),
                            "video.left_wrist_view": np.array([wrist_img]),
                            "state.eef_pos_absolute": obs["robot0_eef_pos"], # GR00T requries [horizon, state_dim]
                            "state.eef_rot_absolute": _quat2axisangle(obs["robot0_eef_quat"]),
                            "state.gripper_close": obs["robot0_gripper_qpos"],
                            "annotation.human.action.task_description": [str(task_description)],
                        }
                        # Query model to get action
                        action_chunk = client.infer(element)
                        action_chunk = np.concatenate([
                            action_chunk['action.eef_pos_delta'],
                            action_chunk['action.eef_rot_delta'],
                            action_chunk['action.gripper_close'].reshape(-1, 1)
                        ], axis=-1)
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    assert False
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=30,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    result_save_path = pathlib.Path(args.video_out_path) / f"{args.task_idx}_results.txt"
    with open(result_save_path, "w") as f:
        f.write(f"Total success rate: {float(total_successes) / float(total_episodes)}\n")
        f.write(f"Total episodes: {total_episodes}")
    logging.info(f"Results saved to {result_save_path}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
