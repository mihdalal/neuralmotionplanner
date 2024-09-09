"""
Get intrinsics matrix of the selected camera
"""

import argparse

import numpy as np
import pyrealsense2 as rs


def get_camera_intrinsics(serial_number):
    """
    Print out the intrinsics matrix of the selected camera

    Args:
        serial_number (int): serial ID of the selected camera.
    """
    # Initialize the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable depth and color streams with the specified serial number
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start the pipeline
    pipeline.start(config)

    try:
        # Wait for the first frame
        frames = pipeline.wait_for_frames()

        # Get the depth and color intrinsics
        depth_intrinsics = frames.get_depth_frame().profile.as_video_stream_profile().intrinsics
        color_intrinsics = frames.get_color_frame().profile.as_video_stream_profile().intrinsics

        # Print the camera intrinsics
        print("Depth Intrinsics:")
        print(f"Width: {depth_intrinsics.width}, Height: {depth_intrinsics.height}")
        print(f"Fx: {depth_intrinsics.fx}, Fy: {depth_intrinsics.fy}")
        print(f"Cx: {depth_intrinsics.ppx}, Cy: {depth_intrinsics.ppy}")

        print("\nColor Intrinsics:")
        print(f"Width: {color_intrinsics.width}, Height: {color_intrinsics.height}")
        print(f"Fx: {color_intrinsics.fx}, Fy: {color_intrinsics.fy}")
        print(f"Cx: {color_intrinsics.ppx}, Cy: {color_intrinsics.ppy}")

        # Construct the camera intrinsics matrix
        intrinsics_matrix = np.array(
            [
                [depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                [0, 0, 1],
            ]
        )

        # Print the intrinsics matrix
        print("\nCamera Intrinsics Matrix:")
        print(intrinsics_matrix)

    finally:
        # Stop the pipeline and release resources
        pipeline.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--serial-number",
        required=True,
        help="serial number of the camera",
    )
    args = parser.parse_args()
    get_camera_intrinsics(args.serial_number)
