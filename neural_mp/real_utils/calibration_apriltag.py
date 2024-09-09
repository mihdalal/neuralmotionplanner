"""
Calibrating camera extrinsics autonomously using Apriltag
"""

import argparse
import faulthandler
import os
import pickle

import cv2
import hydra
import numpy as np
import pupil_apriltags as apriltag
from manimo.actuators.arms.franka_arm import FrankaArm
from manimo.environments.single_arm_env import SingleArmEnv
from omegaconf import OmegaConf

from neural_mp.real_utils.homography_utils import (
    get_cam_constants,
    get_tag_pose_in_camera_frame,
    label_tag_detection,
)


def get_args():
    """
    Get arguments from the command line.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cam_idx", required=True, help="camera id, single int")
    parser.add_argument(
        "--flip",
        action="store_true",
        help="flip the last joint of the robot by 180deg so cameras can get better views of the eef.",
    )
    args = parser.parse_args()

    return args


def move_robot_to_joint_state(env, joint_state: np.ndarray, time_to_go: float = 4):
    """
    Set the joint state of the robot using a controller with lower Kp and Kd gains.

    Args:
        env: Real world deployment environment to control the robot.
        joint_state (np.ndarray): Target joint state (7D).
        time_to_go (float): Time to execute the motion (in seconds).
    """

    if joint_state is not None:
        assert (
            type(env.actuators[0]) == FrankaArm
        )  # should always make sure you are calling the right class
        frankarm = env.actuators[0]
        frankarm.soft_ctrl(joint_state, time_to_go)


def get_img_frame_3d_coords(pixel, d_im, fx, fy, cx, cy, ds):
    """
    Get 3D coordinates from a 2D image frame.

    Args:
        pixel (tuple): Pixel coordinates (x, y).
        d_im (np.ndarray): Depth image.
        fx (float): Focal length in the x-axis.
        fy (float): Focal length in the y-axis.
        cx (float): Optical center in the x-axis.
        cy (float): Optical center in the y-axis.
        ds (float): Depth scaling factor.

    Returns:
        np.ndarray: 3D coordinates corresponding to the 2D pixel.
    """
    y, x = np.round(pixel).astype(int)
    Z = d_im[x, y] * ds
    X = Z / fx * (x - cx)
    Y = Z / fy * (y - cy)
    return np.array([X, Y, Z])


def compute_homography(eef_pose, depth, tag_center_pixel, out_file, fx, fy, cx, cy, ds):
    """
    Compute the homography transformation.

    Args:
        eef_pose (np.ndarray): End effector poses.
        depth (np.ndarray): Depth images.
        tag_center_pixel (list): Center pixel coordinates of tags.
        out_file (str): Output file name for saving the homography.
        fx (float): Focal length in the x-axis.
        fy (float): Focal length in the y-axis.
        cx (float): Optical center in the x-axis.
        cy (float): Optical center in the y-axis.
        ds (float): Depth scaling factor.
    """
    A = np.array(
        [
            get_img_frame_3d_coords(tag_center_pixel[i], depth[i], fx, fy, cx, cy, ds)
            for i in range(len(depth))
        ]
    )
    A = np.hstack((A, np.ones((len(A), 1))))

    B = eef_pose

    # split A and B into val:
    train_indices = np.random.choice(A.shape[0], (int(0.8 * A.shape[0]),))
    A_train, B_train = A[train_indices], B[train_indices]
    val_indices = np.delete(
        np.arange(A.shape[0]), np.random.choice(A.shape[0], (int(0.8 * A.shape[0]),))
    )
    A_val, B_val = A[~val_indices], B[~val_indices]

    res, resi = np.linalg.lstsq(A_train, B_train)[:2]
    print("train diff", np.mean((A_train @ res - B_train) ** 2))
    print("val_diff", np.mean((A_val @ res - B_val) ** 2))
    pickle.dump(res, open("homography_data/homography_transforms/" + out_file + ".pkl", "wb"))


if __name__ == "__main__":
    # initialize franka_env with arm and camera supports
    hydra.initialize(config_path="../../manimo/manimo/conf/", job_name="collect_demos_test")
    env_cfg = hydra.compose(config_name="env")
    actuators_cfg = hydra.compose(config_name="actuators_cam_calibration")
    sensors_cfg = hydra.compose(config_name="sensors")

    faulthandler.enable()
    env = SingleArmEnv(sensors_cfg, actuators_cfg, env_cfg)

    args = get_args()

    config = OmegaConf.load(
        os.path.join(os.path.dirname(__file__), "..", "configs", "calibration_apriltag.yaml")
    )["calibrate_extrinsics"]

    # initialize apriltag detector
    detector = apriltag.Detector(
        families=config.tag.type, quad_decimate=1.0, quad_sigma=0.0, decode_sharpening=0.25
    )
    home_joint_angles = config.robot.home_joint_angles
    if args.flip:
        home_joint_angles[6] -= np.pi
    move_robot_to_joint_state(env, home_joint_angles)

    steps = 3
    camera_name = args.cam_idx
    camera = f"cam{camera_name}"
    tag_length = config.tag.length
    tag_active_pixel_ratio = config.tag.active_pixel_ratio

    calibration_dir = f"homography_data/homography_transforms/"
    os.makedirs(calibration_dir, exist_ok=True)

    if args.flip:
        ori_target = np.array([0.383, 0.924, 0, 0])  # keep it fixed the whole time
    else:
        ori_target = np.array([0.924, -0.383, 0, 0])  # keep it fixed the whole time

    fx, fy, cx, cy, ds, pcs = get_cam_constants(cam_cfg=sensors_cfg["camera"][camera]["camera_cfg"])
    intrinsics = {
        "cx": cx,
        "cy": cy,
        "fx": fx,
        "fy": fy,
    }

    collected_eef_pos = []
    collected_depth = []
    collected_tag_center_pixel = []
    detected_tag_num = 0

    for idx, delta in enumerate(
        [
            np.array([0, 0, -1]),
            np.array([1.5, 0, 0]),
            np.array([0, 0, 1]),
            np.array([0, 1.0, 0]),
            np.array([0, 0, -1]),
            np.array([-1.0, 0, 0]),
            np.array([0, 0, 1]),
            np.array([0, -1.0, 0]),
        ]
    ):
        obs = env.get_obs()
        for i in range(steps):
            delta_action = delta * 0.05

            pos_target = obs["eef_pos"] + delta_action
            env.step([np.array([*pos_target, *ori_target])])
            obs = env.get_obs()

            image = obs[f"{camera}"][0]
            depth = obs[f"{camera}_depth"][0]

            (
                is_tag_detected,
                tag_pose_t,
                tag_pose_r,
                tag_center_pixel,
                tag_corner_pixels,
                tag_family,
            ) = get_tag_pose_in_camera_frame(
                detector=detector,
                image=image,
                intrinsics=intrinsics,
                tag_length=tag_length,
                tag_active_pixel_ratio=tag_active_pixel_ratio,
            )
            if config.tag_detection.display_images:
                cv2.namedWindow(winname="RGB Output", flags=cv2.WINDOW_AUTOSIZE)
                cv2.imshow(winname="RGB Output", mat=image)
                cv2.waitKey(delay=1000)
                cv2.destroyAllWindows()

            if is_tag_detected:
                collected_eef_pos.append(obs["eef_pos"])
                collected_depth.append(depth)
                collected_tag_center_pixel.append(tag_center_pixel)
                detected_tag_num += 1
                print(f"Has detected {detected_tag_num} tags")

                image_labeled = label_tag_detection(
                    image=image, tag_corner_pixels=tag_corner_pixels, tag_family=tag_family
                )

                if config.tag_detection.display_images:
                    cv2.imshow("Tag Detection", image_labeled)
                    cv2.waitKey(delay=1000)
                    cv2.destroyAllWindows()

    collected_eef_pos = np.array(collected_eef_pos)
    collected_depth = np.array(collected_depth)
    collected_tag_center_pixel = np.array(collected_tag_center_pixel)
    outfile_name = f"img{camera_name}_hom"
    compute_homography(
        collected_eef_pos,
        collected_depth,
        collected_tag_center_pixel,
        outfile_name,
        fx,
        fy,
        cx,
        cy,
        ds,
    )
    env.reset()
