import pickle

import cv2
import imageio
import numpy as np
import open3d as o3d
import quaternion
from matplotlib import pyplot as plt
from moviepy.editor import *
from pyquaternion import Quaternion


def compose_poses(pos1, quat1, pos2, quat2):
    """
    Compose two poses.

    Args:
        pos1 (np.ndarray): Position 1.
        quat1 (np.ndarray): Quaternion 1.
        pos2 (np.ndarray): Position 2.
        quat2 (np.ndarray): Quaternion 2.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Composed position and quaternion.
    """
    q1 = Quaternion(quat1[[3, 0, 1, 2]])
    q2 = Quaternion(quat2[[3, 0, 1, 2]])
    return q2.rotate(pos1) + pos2, (q2 * q1).elements[[1, 2, 3, 0]]


def new_quat_from_delta_rpy(quat, delta_rpy):
    """
    Generate a new quaternion from a delta roll-pitch-yaw.

    Args:
        quat (np.ndarray): Original quaternion.
        delta_rpy (np.ndarray): Delta roll-pitch-yaw.

    Returns:
        np.ndarray: New quaternion.
    """
    rpy = quat_to_rpy(quat)
    rpy += delta_rpy
    return convert_xyzw_to_wxyz(rpy_to_quat(rpy))


def convert_xyzw_to_wxyz(q):
    """
    Convert a quaternion from XYZW format to WXYZ format.

    Args:
        q (np.ndarray): Quaternion in XYZW format.

    Returns:
        np.ndarray: Quaternion in WXYZ format.
    """
    return np.array([q[3], q[0], q[1], q[2]])


def quat_to_rpy(q):
    """
    Convert a quaternion to roll-pitch-yaw.

    Args:
        q (np.ndarray): Quaternion.

    Returns:
        np.ndarray: Roll-pitch-yaw angles.
    """
    q = quaternion.quaternion(q[0], q[1], q[2], q[3])
    return quaternion.as_euler_angles(q)


def rpy_to_quat(rpy):
    """
    Convert roll-pitch-yaw angles to a quaternion.

    Args:
        rpy (np.ndarray): Roll-pitch-yaw angles.

    Returns:
        np.ndarray: Quaternion.
    """
    q = quaternion.from_euler_angles(rpy)
    return np.array([q.x, q.y, q.z, q.w])


def quat_to_axis_angle(_quat):
    """
    Convert a quaternion to axis-angle representation.

    Args:
        _quat (np.ndarray): Quaternion.

    Returns:
        Tuple[float, float, float, float]: Axis-angle representation.
    """
    if _quat[0] == 1:
        return (0, 1, 0, 0)
    else:
        theta = 2 * np.arccos(_quat[0])
        x = _quat[1] / np.sin(theta / 2)
        y = _quat[2] / np.sin(theta / 2)
        z = _quat[3] / np.sin(theta / 2)
        return (theta, x, y, z)


def axis_angle_to_quat(_axis_angle):
    """
    Convert axis-angle representation to a quaternion.

    Args:
        _axis_angle (Tuple[float, float, float, float]): Axis-angle representation.

    Returns:
        np.ndarray: Quaternion.
    """
    theta, x, y, z = _axis_angle
    q0 = np.cos(theta / 2)
    q1 = x * np.sin(theta / 2)
    q2 = y * np.sin(theta / 2)
    q3 = z * np.sin(theta / 2)
    return (q0, q1, q2, q3)


def get_connected_devices():
    """
    Get a list of connected RealSense devices.

    Returns:
        list: List of connected device serial numbers.
    """
    import pyrealsense2 as rs

    realsense_ctx = rs.context()
    connected_devices = []
    for i in range(len(realsense_ctx.devices)):
        detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
        connected_devices.append(detected_camera)
    return connected_devices


def slerp(quat1, quat2, interp_gap):
    """
    Spherical linear interpolation (SLERP) between two quaternions.

    Args:
        quat1 (np.ndarray): Start quaternion.
        quat2 (np.ndarray): End quaternion.
        interp_gap (float): Interpolation gap.

    Returns:
        np.ndarray: Interpolated quaternions.
    """
    q1 = Quaternion(quat1[[3, 0, 1, 2]])
    q2 = Quaternion(quat2[[3, 0, 1, 2]])

    return np.array(
        [
            Quaternion.slerp(q1, q2, w).elements[[1, 2, 3, 0]]
            for w in np.linspace(0, 1, int(interp_gap))
        ]
    )


def interpolate(p1, p2, interp_gap):
    """
    Interpolate between two points.

    Args:
        p1 (np.ndarray): Start point.
        p2 (np.ndarray): End point.
        interp_gap (float): Interpolation gap.

    Returns:
        np.ndarray: Interpolated points.
    """
    length = int(np.linalg.norm(p1 - p2) / interp_gap) + 1
    w = np.linspace(0, 1, length)
    w = np.tile(w, (p1.shape[0], 1)).T
    p1_tile = np.tile(p1, (length, 1))
    p2_tile = np.tile(p2, (length, 1))
    return p1_tile * (1 - w) + p2 * w


def interpolate_between_waypoints(path, interp_gap):
    """
    Interpolate between waypoints in a path.

    Args:
        path (list): List of waypoints.
        interp_gap (float): Interpolation gap.

    Returns:
        np.ndarray: Interpolated path.
    """
    refined_path = [path[0]]
    for i in range(1, len(path)):
        if np.linalg.norm(refined_path[-1] - path[i]) < interp_gap:
            pass
        else:
            interp_points = interpolate(refined_path[-1], path[i], interp_gap)
            refined_path.extend(interp_points[1:])

    return np.array(refined_path)


def filter_point_cloud_box_around_core(_points, _colors, core, box, excl_radius=0.05):
    """
    Filter points in a point cloud within a box around a core point, excluding a radius around the core.

    Args:
        _points (np.ndarray): Point cloud points.
        _colors (np.ndarray): Point cloud colors.
        core (np.ndarray): Core point.
        box (np.ndarray): Box dimensions.
        excl_radius (float): Exclusion radius around the core.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Filtered points and colors.
    """
    excl_ball = np.linalg.norm(_points - core, axis=1) < excl_radius
    incl_box = np.prod((core - box < _points) * (_points < core + box), axis=1)

    idxs = np.argwhere(incl_box * (1 - excl_ball)).flatten()
    return _points[idxs], _colors[idxs]


def point_in_workspace(point):
    """
    Check if a point is within the workspace.

    Args:
        point (np.ndarray): Point to check.

    Returns:
        bool: True if the point is within the workspace, False otherwise.
    """
    return point[0] > 0.2 and point[0] < 0.8


def get_cam_constants(cam_cfg):
    """
    Get camera constants from the camera configuration.

    Args:
        cam_cfg (dict): Camera configuration.

    Returns:
        Tuple: Camera constants (fx, fy, cx, cy, ds, pcs).
    """
    K1 = np.array(cam_cfg["intrinsics"])
    mv_shift = np.array(cam_cfg["mv_shift"])

    fx, fy, cx, cy = K1[0, 0], K1[1, 1], K1[0, -1], K1[1, -1]
    ds = 0.0010000000474974513
    pcs = mv_shift
    return fx, fy, cx, cy, ds, pcs


def save_video(img_list, _file, frame_duration=0.01):
    """
    Save a list of images as a video.

    Args:
        img_list (list): List of images.
        _file (str): Output file name.
        frame_duration (float): Duration of each frame.
    """
    clips = [ImageClip(m).set_duration(frame_duration) for m in img_list]
    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(_file, fps=24)


def load_video(_file):
    """
    Load a video file.

    Args:
        _file (str): Input file name.

    Returns:
        np.ndarray: Loaded video frames.
    """
    reader = imageio.get_reader(_file)
    frames = []
    for cur_frame in reader:
        frames.append(cur_frame)
    return np.array(frames)


def pad_vid(vid, _len):
    """
    Pad a video to a specified length.

    Args:
        vid (np.ndarray): Video frames.
        _len (int): Desired length.

    Returns:
        np.ndarray: Padded video.
    """
    assert len(vid) <= _len
    if len(vid) == _len:
        return vid
    else:
        addend = np.repeat([vid[-1]], _len - len(vid), axis=0)
        return np.concatenate([vid, addend], axis=0)


def pad_vid_list(_vid_list):
    """
    Pad a list of videos to the length of the longest video.

    Args:
        _vid_list (list): List of video frames.

    Returns:
        np.ndarray: Padded list of videos.
    """
    max_frames = max([vid.shape[0] for vid in _vid_list])
    return np.array([pad_vid(vid, max_frames) for vid in _vid_list])


def save_grid(vid_list, _out_file, frame_duration, w, h, spacing_gap=5):
    """
    Save a grid of videos.

    Args:
        vid_list (np.ndarray): List of videos.
        _out_file (str): Output file name.
        frame_duration (float): Duration of each frame.
        w (int): Grid width.
        h (int): Grid height.
        spacing_gap (int): Spacing gap between videos.
    """
    assert vid_list.shape[0] == w * h
    _, n, h_res, w_res, _ = vid_list.shape
    vid_list = vid_list.reshape((h, w) + vid_list.shape[1:])
    vid_list = np.transpose(vid_list, (2, 0, 3, 1, 4, 5))
    s = vid_list.shape
    vid_grid = np.reshape(vid_list, (s[0], s[1] * s[2], s[3] * s[4], s[5]))

    for i in range(h):
        vid_grid[:, i * h_res : i * h_res + 5, :, :] = 0

    for j in range(w):
        vid_grid[:, :, j * w_res : j * w_res + 5, :] = 0

    save_video(vid_grid, _out_file, frame_duration=frame_duration)


def save_pointcloud_sequential(_file, _points):
    """
    Save a point cloud sequentially.

    Args:
        _file (str): Output file name.
        _points (np.ndarray): Point cloud points.
    """
    red_weights = np.linspace(0, 1, len(_points)).reshape(len(_points), 1)
    colors = [1, 0, 0] * (np.ones((1500, 3)) * red_weights) + [0, 0, 1] * (
        np.ones((1500, 3)) * red_weights[::-1]
    )
    save_pointcloud(_file, _points, colors)


def plot_pointcloud(pts, _colors, marked_points=[]):
    """
    Plot a point cloud.

    Args:
        pts (np.ndarray): Point cloud points.
        _colors (np.ndarray): Point cloud colors.
        marked_points (list): List of marked points.
    """
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection="3d")
    ax.set_xlabel("X, [m]")
    ax.set_ylabel("Y, [m]")
    ax.set_zlabel("Z, [m]")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 0.9)

    for point in marked_points:
        ax.scatter3D(point[0], point[1], point[2], color="red", s=200)

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=_colors)
    plt.show()


def save_pointcloud(_file, _points, _colors):
    """
    Save a point cloud to a file.

    Args:
        _file (str): Output file name.
        _points (np.ndarray): Point cloud points.
        _colors (np.ndarray): Point cloud colors.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(_points)
    pcd.colors = o3d.utility.Vector3dVector(_colors)
    o3d.io.write_point_cloud(_file, pcd)


def shift_pointcloud(_file, shift):
    """
    Shift a point cloud.

    Args:
        _file (str): Input file name.
        shift (np.ndarray): Shift vector.
    """
    pcd = o3d.io.read_point_cloud(_file + ".ply")
    _points = np.asarray(pcd.points)
    _points += shift
    save_pointcloud(_file + "_shift.ply", _points, np.asarray(pcd.colors))


def label_tag_detection(image, tag_corner_pixels, tag_family):
    """
    Label a tag detection on an image.

    Args:
        image (np.ndarray): Input image.
        tag_corner_pixels (np.ndarray): Corner pixels of the tag.
        tag_family (str): Tag family.

    Returns:
        np.ndarray: Labeled image.
    """
    image_labeled = image.copy()

    corner_a = (int(tag_corner_pixels[0][0]), int(tag_corner_pixels[0][1]))
    corner_b = (int(tag_corner_pixels[1][0]), int(tag_corner_pixels[1][1]))
    corner_c = (int(tag_corner_pixels[2][0]), int(tag_corner_pixels[2][1]))
    corner_d = (int(tag_corner_pixels[3][0]), int(tag_corner_pixels[3][1]))

    # Draw oriented box on image
    cv2.line(img=image_labeled, pt1=corner_a, pt2=corner_b, color=(0, 255, 0), thickness=2)
    cv2.line(img=image_labeled, pt1=corner_b, pt2=corner_c, color=(0, 255, 0), thickness=2)
    cv2.line(img=image_labeled, pt1=corner_c, pt2=corner_d, color=(0, 255, 0), thickness=2)
    cv2.line(img=image_labeled, pt1=corner_d, pt2=corner_a, color=(0, 255, 0), thickness=2)

    # Draw tag family on image
    cv2.putText(
        img=image_labeled,
        text=tag_family.decode("utf-8"),
        org=(corner_a[0], corner_c[1] - 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(255, 0, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    return image_labeled


def get_tag_pose_in_camera_frame(detector, image, intrinsics, tag_length, tag_active_pixel_ratio):
    """
    Detect an AprilTag in an image and get its pose in the camera frame.

    Args:
        detector: AprilTag detector.
        image (np.ndarray): Input image.
        intrinsics (dict): Camera intrinsics.
        tag_length (float): Tag length.
        tag_active_pixel_ratio (float): Active pixel ratio of the tag.

    Returns:
        Tuple: Detection status, position, orientation matrix, center pixel, corner pixels, and tag family.
    """
    gray_image = cv2.cvtColor(src=image.astype(np.uint8), code=cv2.COLOR_BGR2GRAY)
    tag_active_length = tag_length * tag_active_pixel_ratio
    detection = detector.detect(
        img=gray_image,
        estimate_tag_pose=True,
        camera_params=[intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]],
        tag_size=tag_active_length,
    )

    if detection:
        is_detected = True
        pos = detection[0].pose_t.copy().squeeze()  # (3, )
        ori_mat = detection[0].pose_R.copy()
        center_pixel = detection[0].center
        corner_pixels = detection[0].corners
        family = detection[0].tag_family

    else:
        is_detected = False
        pos, ori_mat, center_pixel, corner_pixels, family = None, None, None, None, None

    return is_detected, pos, ori_mat, center_pixel, corner_pixels, family


class HomographyTransform:
    def __init__(self, key, transform_file, cam_cfg):
        """
        Initialize the HomographyTransform class.

        Args:
            key (str): Key for the transformation file.
            transform_file (str): Transformation file name.
            cam_cfg (dict): Camera configuration.
        """
        transform_file = key + "_" + transform_file + ".pkl"
        self.transform_file = transform_file
        self.transform_matrix = pickle.load(
            open("homography_data/homography_transforms/" + transform_file, "rb")
        )
        self.fx, self.fy, self.cx, self.cy, self.ds, self.pcs = get_cam_constants(cam_cfg)
        self.workspace_min = np.array(cam_cfg["workspace_min"])
        self.workspace_max = np.array(cam_cfg["workspace_max"])
        self.filter_pc = True
        self.cam_cfg = cam_cfg

    def get_img_frame_3d_coords(self, pixel, d_im):
        """
        Get 3D coordinates from image frame coordinates.

        Args:
            pixel (tuple): Pixel coordinates.
            d_im (np.ndarray): Depth image.

        Returns:
            np.ndarray: 3D coordinates.
        """
        y, x = pixel
        Z = d_im[x, y] * self.ds
        X = Z / self.fx * (x - self.cx)
        Y = Z / self.fy * (y - self.cy)
        return np.array([X, Y, Z])

    def get_robot_coords(self, pixel, d_im):
        """
        Get robot coordinates from image frame coordinates.

        Args:
            pixel (tuple): Pixel coordinates.
            d_im (np.ndarray): Depth image.

        Returns:
            np.ndarray: Robot coordinates.
        """
        X, Y, Z = self.get_img_frame_3d_coords(pixel, d_im)
        return (np.array([X, Y, Z, 1]) @ self.transform_matrix) + self.pcs

    def get_robot_coords_vectorized(self, px_arr, depth_im):
        """
        Get robot coordinates from image frame coordinates in a vectorized manner.

        Args:
            px_arr (np.ndarray): Pixel array.
            depth_im (np.ndarray): Depth image.

        Returns:
            np.ndarray: Robot coordinates.
        """
        X, Y = px_arr
        Z = depth_im * self.ds
        X = Z / self.fx * (X - self.cx)
        Y = Z / self.fy * (Y - self.cy)

        img_frame_coords = np.transpose(np.array([X, Y, Z]), (1, 2, 0))
        ones = np.ones(img_frame_coords.shape[:2] + (1,))
        return np.concatenate([img_frame_coords, ones], axis=-1) @ self.transform_matrix + self.pcs

    def get_filtered_pc(self, _points, _colors=None):
        """
        Get filtered point cloud.

        Args:
            _points (np.ndarray): Point cloud points.
            _colors (np.ndarray, optional): Point cloud colors.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Filtered points and colors.
        """
        mask = np.all(_points > self.workspace_min, axis=1) * np.all(
            _points < self.workspace_max, axis=1
        )
        if _colors is None:
            return _points[mask]
        else:
            return _points[mask], _colors[mask]

    def denoise_pc(self, points, colors=None):
        """
        Denoise a point cloud.

        Args:
            points (np.ndarray): Point cloud points.
            colors (np.ndarray, optional): Point cloud colors.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Denoised points and colors.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        denoised_pc = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1)[0]
        if colors is None:
            return np.asarray(denoised_pc.points)
        else:
            return np.asarray(denoised_pc.points), np.asarray(denoised_pc.colors)

    def mask_dilation(self, masks, radius=1, iteration=1, debug=False):
        """
        Dilate a mask.

        Args:
            masks (np.ndarray): Masks to be dilated.
            radius (int): Radius of dilation.
            iteration (int): Number of iterations.
            debug (bool): Enable debug mode.

        Returns:
            np.ndarray: Dilated masks.
        """
        from scipy.ndimage import binary_dilation

        structure_element = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=bool)
        for i in range(2 * radius + 1):
            for j in range(2 * radius + 1):
                if (i - radius) ** 2 + (j - radius) ** 2 < (radius + np.sqrt(0.5)) ** 2:
                    structure_element[i, j] = True

        if debug:
            print("structure element: ", structure_element)
        enlarged_mask = binary_dilation(masks, structure=structure_element)
        return enlarged_mask

    def get_pointcloud(self, depth, image=None):
        """
        Get a point cloud from a depth image.

        Args:
            depth (np.ndarray): Depth image.
            image (np.ndarray, optional): RGB image.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Point cloud points and colors if image is provided, otherwise just points.
        """
        h, w = depth.shape

        mg = np.meshgrid(np.arange(0, 640, 640 / w), np.arange(0, 480, 480 / h))
        grid = np.concatenate([np.expand_dims(mg[1], 0), np.expand_dims(mg[0], 0)], axis=0)
        points = self.get_robot_coords_vectorized(grid, depth)

        if image is None:
            return points
        else:
            colors = image / 255.0
            return points, colors
