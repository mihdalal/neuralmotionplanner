import argparse

import numpy as np

from neural_mp.envs.franka_real_env import FrankaRealEnvManimo
from neural_mp.real_utils.neural_motion_planner import NeuralMP

if __name__ == "__main__":
    """
    example of motion planning with Manimo
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mdl_url",
        type=str,
        default="mihdalal/NeuralMP",
        help="hugging face url to load the neural_mp model",
    )
    parser.add_argument(
        "--cache-name",
        type=str,
        default="scene1_single_blcok",
        help="Specify the scene cache file with pcd and rgb data",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help=("If set, will use pre-stored point clouds"),
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
        "--train-mode", action="store_true", help=("If set, will eval with policy in training mode")
    )
    parser.add_argument(
        "--tto", action="store_true", help=("If set, will apply test time optimization")
    )
    parser.add_argument(
        "--in-hand", action="store_true", help=("If set, will enable in hand mode for eval")
    )
    parser.add_argument(
        "--in-hand-params",
        nargs="+",
        type=float,
        default=[0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
        help="Specify the bounding box of the in hand object. 10 params in total [size(xyz), pos(xyz), ori(xyzw)] 3+3+4.",
    )

    args = parser.parse_args()
    env = FrankaRealEnvManimo()
    neural_mp = NeuralMP(
        env=env,
        model_url=args.mdl_url,
        train_mode=args.train_mode,
        in_hand=args.in_hand,
        in_hand_params=args.in_hand_params,
        visualize=True,
    )

    points, colors = neural_mp.get_scene_pcd(
        use_cache=args.use_cache,
        cache_name=args.cache_name,
        debug_combined_pcd=args.debug_combined_pcd,
        denoise=args.denoise_pcd,
    )

    # specify start and goal configurations
    start_config = np.array([-0.538, 0.628, -0.061, -1.750, 0.126, 2.418, 1.610])
    goal_config = np.array([1.067, 0.847, -0.591, -1.627, 0.623, 2.295, 2.580])

    if args.tto:
        trajectory = neural_mp.motion_plan_with_tto(
            start_config=start_config,
            goal_config=goal_config,
            points=points,
            colors=colors,
        )
    else:
        trajectory = neural_mp.motion_plan(
            start_config=start_config,
            goal_config=goal_config,
            points=points,
            colors=colors,
        )

    success, joint_error = neural_mp.execute_motion_plan(trajectory, speed=0.2)
