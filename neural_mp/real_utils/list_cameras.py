"""
List device ID of all the available cameras
"""

import pyrealsense2 as rs

device_ls = []
for cam in rs.context().query_devices():
    device_ls.append(cam.get_info(rs.camera_info(1)))
device_ls.sort()

for i, device in enumerate(device_ls):
    print(f"Device {i}: {device}")
