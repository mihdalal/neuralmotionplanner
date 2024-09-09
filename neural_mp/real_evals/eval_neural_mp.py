"""
Neural MP evaluation
"""

import argparse

import numpy as np

from neural_mp.envs.franka_real_env import FrankaRealEnvManimo
from neural_mp.real_evals.eval_base import rw_eval
from neural_mp.real_utils.neural_motion_planner import NeuralMP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mdl_url",
        type=str,
        default="mihdalal/NeuralMP",
        help="hugging face url to load the neural_mp model",
    )
    parser.add_argument(
        "--cfg-set",
        type=str,
        default="scene1_l1",
        help="Specify the config set for testing",
    )
    parser.add_argument(
        "--cache-name",
        type=str,
        default="scene1_single_blcok",
        help="Specify the scene cache file with pcd and rgb data",
    )
    parser.add_argument(
        "--cam-only",
        action="store_true",
        help=(
            "If set, franka_env will launch in cam_only mode, where the robot will not be connected"
        ),
    )
    parser.add_argument(
        "--arm-only",
        action="store_true",
        help=(
            "If set, robot will launch in arm_only mode, where the gripper will not be connected"
        ),
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help=("If set, will use pre-stored point clouds"),
    )
    parser.add_argument(
        "--debug-raw-pcd",
        action="store_true",
        help=("If set, will show visualization of raw pcd from each camera"),
    )
    parser.add_argument(
        "--debug-combined-pcd",
        action="store_true",
        help=("If set, will show visualization of the combined pcd"),
    )
    parser.add_argument(
        "--denoise-pcd",
        action="store_true",
        help=("If set, will apply denoising to the pcds"),
    )
    parser.add_argument(
        "--save-pcd",
        action="store_true",
        help=("If set, will save the combined pcd"),
    )
    parser.add_argument(
        "-l",
        "--log-name",
        type=str,
        default="rw_eval_neural_mp",
        help="Specify the file name for logging eval info",
    )
    parser.add_argument(
        "--train-mode", action="store_true", help=("If set, will eval with policy in training mode")
    )
    parser.add_argument(
        "--tto", action="store_true", help=("If set, will apply test time optimization")
    )
    parser.add_argument(
        "--in-hand", action="store_true", help=("If set, will enable in hand mode for eval")
    )
    parser.add_argument(
        "--in-hand-param",
        nargs="+",
        type=float,
        default=[0.26, 0.08, 0.2, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 1.0],
        help="Specify the bounding box of the in hand object. 10 params in total [size(xyz), pos(xyz), ori(xyzw)] 3+3+4.",
    )

    args = parser.parse_args()

    env = FrankaRealEnvManimo(cam_only=args.cam_only, arm_only=args.arm_only)
    eval_agent = NeuralMP(
        env=env,
        model_url=args.mdl_url,
        train_mode=args.train_mode,
        in_hand=args.in_hand,
        in_hand_params=args.in_hand_param,
        visualize=True,
    )

    points, colors = eval_agent.get_scene_pcd(
        use_cache=args.use_cache,
        cache_name=args.cache_name,
        debug_raw_pcd=args.debug_raw_pcd,
        debug_combined_pcd=args.debug_combined_pcd,
        save_pcd=args.save_pcd,
        save_file_name=args.log_name,
        denoise=args.denoise_pcd,
    )

    config_set = np.load("real_world_test_set/collected_configs/" + args.cfg_set + ".npy")

    rw_eval(
        eval_agent=eval_agent,
        env=env,
        points=points,
        colors=colors[:, [2, 1, 0]],
        config_set=config_set,
        arm_only=args.arm_only,
        args=args,
        file_path="./real_world_test_set/evals/" + args.log_name + ".csv",
    )
