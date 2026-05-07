#!/usr/bin/env python3

import json
import pickle
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image


def _to_seconds(timestamps):
    ts = np.array(timestamps, dtype=np.float64)
    if ts.size == 0:
        return ts
    if np.nanmax(ts) > 1e12:
        ts = ts * 1e-9
    return ts


def _rot6d_to_rotmat(rot6d):
    a1 = np.array(rot6d[:3], dtype=np.float64)
    a2 = np.array(rot6d[3:6], dtype=np.float64)
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def _quaternion_from_rotation_matrix(rot):
    trace = np.trace(rot)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    else:
        if rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif rot[1, 1] > rot[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s

    return [float(x), float(y), float(z), float(w)]


def _quat_to_euler(q):
    x, y, z, w = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp) if abs(sinp) <= 1 else np.sign(sinp) * np.pi / 2

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def _extract_pose_series(data_list, key):
    positions = []
    eulers = []
    for d in data_list:
        if not isinstance(d, dict) or key not in d:
            positions.append([np.nan, np.nan, np.nan])
            eulers.append([np.nan, np.nan, np.nan])
            continue
        pose = d.get(key)
        if not isinstance(pose, (list, tuple)):
            positions.append([np.nan, np.nan, np.nan])
            eulers.append([np.nan, np.nan, np.nan])
            continue
        if len(pose) == 7:
            p = pose[:3]
            q = pose[3:7]
        elif len(pose) == 9:
            p = pose[:3]
            rot6d = pose[3:9]
            rot = _rot6d_to_rotmat(rot6d)
            q = _quaternion_from_rotation_matrix(rot)
        else:
            positions.append([np.nan, np.nan, np.nan])
            eulers.append([np.nan, np.nan, np.nan])
            continue
        positions.append(p)
        eulers.append(_quat_to_euler(q))
    eulers_arr = np.array(eulers, dtype=np.float64)
    eulers_arr = _unwrap_euler_series(eulers_arr, max_jump_deg=0.5)
    return np.array(positions, dtype=np.float64), eulers_arr


def _unwrap_euler_series(eulers, max_jump_deg=0.5):
    if eulers.size == 0:
        return eulers
    max_jump = np.deg2rad(max_jump_deg)
    unwrapped = eulers.copy()
    for axis in range(unwrapped.shape[1]):
        series = unwrapped[:, axis]
        valid = np.isfinite(series)
        if not np.any(valid):
            continue
        idxs = np.where(valid)[0]
        prev = series[idxs[0]]
        for idx in idxs[1:]:
            val = series[idx]
            candidates = np.array([val, val + 2.0 * np.pi, val - 2.0 * np.pi], dtype=np.float64)
            deltas = np.abs(candidates - prev)
            best = int(np.argmin(deltas))
            val = candidates[best]
            series[idx] = val
            prev = val
        unwrapped[:, axis] = series
    return unwrapped


def _apply_ee_rotation_wrap_correction(
    euler_series_list,
    enabled=True,
    near_2pi_tol_rad=np.deg2rad(60.0),
    near_zero_tol_rad=np.deg2rad(60.0),
    near_pi_tol_rad=np.deg2rad(45.0),
):
    if not enabled:
        return euler_series_list

    corrected = []
    for arr in euler_series_list:
        if isinstance(arr, np.ndarray):
            corrected.append(arr.copy())
        else:
            corrected.append(arr)

    if not corrected:
        return corrected

    n_axes = 3
    for axis in range(n_axes):
        medians = []
        for arr in corrected:
            if not isinstance(arr, np.ndarray) or arr.size == 0:
                medians.append(np.nan)
                continue
            vals = arr[:, axis]
            valid = vals[np.isfinite(vals)]
            medians.append(float(np.nanmedian(valid)) if valid.size > 0 else np.nan)

        finite_medians = [m for m in medians if np.isfinite(m)]
        if not finite_medians:
            continue

        near_zero_medians = [m for m in finite_medians if abs(m) <= near_zero_tol_rad]
        target = float(np.nanmedian(np.array(near_zero_medians, dtype=np.float64))) if near_zero_medians else 0.0

        for idx, med in enumerate(medians):
            if not np.isfinite(med):
                continue
            delta = med - target
            if abs(abs(delta) - 2.0 * np.pi) <= near_2pi_tol_rad:
                corrected[idx][:, axis] = corrected[idx][:, axis] - np.sign(delta) * 2.0 * np.pi

    # Fallback for single-series plotting: use the other rotation axes as reference.
    if len(corrected) == 1 and isinstance(corrected[0], np.ndarray) and corrected[0].size > 0:
        arr = corrected[0]
        axis_medians = []
        for axis in range(n_axes):
            vals = arr[:, axis]
            valid = vals[np.isfinite(vals)]
            axis_medians.append(float(np.nanmedian(valid)) if valid.size > 0 else np.nan)
        for axis, med in enumerate(axis_medians):
            if not np.isfinite(med):
                continue
            other_medians = [m for i, m in enumerate(axis_medians) if i != axis and np.isfinite(m)]
            if not other_medians:
                continue
            if not any(abs(m) <= near_zero_tol_rad for m in other_medians):
                continue
            if abs(abs(med) - 2.0 * np.pi) <= near_2pi_tol_rad:
                arr[:, axis] = arr[:, axis] - np.sign(med) * 2.0 * np.pi
        corrected[0] = arr

    # Handle the common plotting case where one axis sits near +/-pi
    # while the other axes are near zero.
    for idx, arr in enumerate(corrected):
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            continue
        axis_medians = []
        for axis in range(n_axes):
            vals = arr[:, axis]
            valid = vals[np.isfinite(vals)]
            axis_medians.append(float(np.nanmedian(valid)) if valid.size > 0 else np.nan)

        near_zero_axes = [
            axis for axis, med in enumerate(axis_medians) if np.isfinite(med) and abs(med) <= near_zero_tol_rad
        ]
        near_pi_axes = [
            axis
            for axis, med in enumerate(axis_medians)
            if np.isfinite(med) and abs(abs(med) - np.pi) <= near_pi_tol_rad
        ]
        if len(near_pi_axes) == 1 and len(near_zero_axes) >= 1:
            axis = near_pi_axes[0]
            med = axis_medians[axis]
            arr[:, axis] = arr[:, axis] - np.sign(med) * np.pi
            corrected[idx] = arr

    return corrected


def _extract_scalar_from_jointstate(data_list, key, index=0):
    vals = []
    for d in data_list:
        if isinstance(d, dict) and key in d and len(d[key]) > index:
            vals.append(d[key][index])
        else:
            vals.append(np.nan)
    return np.array(vals, dtype=np.float64)


def _extract_vector_series(data_list, key="data"):
    series = []
    for d in data_list:
        if isinstance(d, dict) and key in d:
            series.append(d[key])
        else:
            series.append([])
    try:
        return np.array(series, dtype=np.float64)
    except Exception:
        return np.array(series, dtype=object)


def load_data(data_path):
    """Load data from pkl or json file"""
    data_path = Path(data_path)

    if data_path.suffix == ".pkl":
        with open(data_path, "rb") as f:
            return pickle.load(f)
    elif data_path.suffix == ".json":
        with open(data_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")


def extract_topic_data(pkl_data, topic_name):
    """Extract data for a specific topic from pkl format"""
    if topic_name not in pkl_data["data"]:
        return None, None

    data = pkl_data["data"][topic_name]
    timestamps = pkl_data["timestamps"][topic_name]

    return data, timestamps


def create_image_gif(
    pkl_data, output_path, topic_name="/realsense/front/color/image_raw", fps=30, video_downsample_factor=1
):
    """Create GIF from image data"""
    print(f"Creating MP4 from topic {topic_name}...")

    image_data, timestamps = extract_topic_data(pkl_data, topic_name)

    if image_data is None:
        print(f"No data found for topic: {topic_name}")
        return

    images = []
    for i, data in enumerate(image_data):
        if "data" in data:
            # Convert ROS Image message data to numpy array
            width = data["width"]
            height = data["height"]
            encoding = data["encoding"]
            raw_data = data["data"]

            if encoding == "rgb8":
                # RGB image
                img_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width, 3)
            elif encoding == "jpeg":
                img_buffer = np.frombuffer(raw_data, dtype=np.uint8)
                decoded = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
                if decoded is None:
                    print("Unsupported encoding: jpeg decode failed")
                    continue
                img_array = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            elif encoding == "32FC1":
                # Depth image
                img_array = np.frombuffer(raw_data, dtype=np.float32).reshape(height, width)
                # Normalize depth for visualization
                img_array = (img_array - np.nanmin(img_array)) / (np.nanmax(img_array) - np.nanmin(img_array))
                img_array = (img_array * 255).astype(np.uint8)
                img_array = cv2.applyColorMap(img_array, cv2.COLORMAP_JET)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            else:
                print(f"Unsupported encoding: {encoding}")
                continue

            # Add step number text
            img_pil = Image.fromarray(img_array)
            images.append(img_pil)

            # if i % 100 == 0:
            #     print(f"  Processed {i+1}/{len(image_data)} frames")

            # if i > 300 and i < 350 and topic_name == '/realsense/front/color/image_raw':
            #     # Save individual frame as an image
            #     out_path = output_path.parent / "frames"
            #     out_path.mkdir(parents=True, exist_ok=True)
            #     frame_path = out_path / f"frame_{i:04d}.png"
            #     img_pil.save(frame_path)
            #     print(f"  Saved frame {i} to {frame_path}")

    if images:
        resized_images = [
            img.resize((img.width // video_downsample_factor, img.height // video_downsample_factor)) for img in images
        ]
        imageio.mimsave(output_path, resized_images, fps=fps)
        # images[0].save(
        #     output_path,
        #     save_all=True,
        #     append_images=images[1:],
        #     duration=int(1000/fps),
        #     loop=0
        # )
        print(f"✅ MP4 with {len(images)} frames saved to {output_path}")
    else:
        print("❌ No images to save")


def ros_image_to_numpy(img_msg):
    width = img_msg["width"]
    height = img_msg["height"]
    encoding = img_msg["encoding"]
    raw_data = img_msg["data"]

    if encoding == "rgb8":
        img = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width, 3)
    elif encoding == "jpeg":
        img_buffer = np.frombuffer(raw_data, dtype=np.uint8)
        decoded = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Unsupported encoding: jpeg decode failed")
        img = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    elif encoding == "32FC1":
        img = np.frombuffer(raw_data, dtype=np.float32).reshape(height, width)
        img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img) + 1e-6)
        img = (img * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    return img


def plot_joint_data(pkl_data, output_dir, fix_ee_rotation_wrap=True):
    """Plot all data recorded from cartesian recording"""
    print("Creating plots...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data for each topic
    topics = {
        "external_torques": "/franka_robot_state_broadcaster/external_joint_torques",
        "measured_joints": "/franka_robot_state_broadcaster/measured_joint_states",
        "gripper_state": "/robotiq_gripper/gripper_state",
        "gripper_cmd": "/factr_teleop/gripper_pos_cmd",
        "tracking_error": "/cartesian_impedance_controller/tracking_error",
        "ee_velocity": "/cartesian_impedance_controller/ee_velocity",
        "pose_command": "/cartesian_impedance_controller/pose_command",
        "external_wrench": "/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame",
        "robot_state": "/franka_robot_state_broadcaster/robot_state",
        "image": "/realsense/front/color/image_raw",
        # Optional inference topics (if present in pkl)
        "predictions": "/inference/ensembled_predictions",
        "raw_predictions": "/inference/raw_predictions",
        "attention": "/inference/attention",
        "impedance_cmd": "/joint_impedance_dynamic_gain_controller/joint_impedance_command",
    }

    data_dict = {}
    global_t0 = None
    first_timestamps = []
    for key, topic in topics.items():
        data, timestamps = extract_topic_data(pkl_data, topic)
        if data is not None and len(data) > 0:
            ts_sec = _to_seconds(timestamps)
            data_dict[key] = {"data": data, "timestamps": ts_sec}
            if ts_sec.size > 0:
                first_timestamps.append(ts_sec[0])

    if first_timestamps:
        global_t0 = float(np.nanmin(np.array(first_timestamps, dtype=np.float64)))
        for key in data_dict:
            data_dict[key]["timestamps"] = data_dict[key]["timestamps"] - global_t0
    else:
        global_t0 = 0.0

    # 1. Plot measured joint positions, velocities, efforts
    if "measured_joints" in data_dict:
        print("  Plotting measured joint states...")
        measured = data_dict["measured_joints"]["data"]
        timestamps = data_dict["measured_joints"]["timestamps"]

        positions = np.array([d.get("position", [np.nan] * 7) for d in measured])
        velocities = np.array([d.get("velocity", [np.nan] * 7) for d in measured])
        efforts = np.array([d.get("effort", [np.nan] * 7) for d in measured])

        fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
        fig.suptitle("Measured Joint Positions", fontsize=16)
        for i in range(7):
            axes[i].plot(timestamps, positions[:, i], linewidth=1.2)
            axes[i].set_ylabel(f"Joint {i + 1} [rad/s]", fontsize=10)
            axes[i].grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time [s]", fontsize=10)
        plt.tight_layout()
        out = output_dir / "joint_positions.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Saved to {out}")

        fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
        fig.suptitle("Measured Joint Velocities", fontsize=16)
        for i in range(7):
            axes[i].plot(timestamps, velocities[:, i], linewidth=1.2)
            axes[i].set_ylabel(f"Joint {i + 1} [rad/s]", fontsize=10)
            axes[i].grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time [s]", fontsize=10)
        plt.tight_layout()
        out = output_dir / "joint_velocities.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Saved to {out}")

        # fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
        # fig.suptitle("Measured Joint Efforts", fontsize=16)
        # for i in range(7):
        #     axes[i].plot(timestamps, efforts[:, i], linewidth=1.2)
        #     axes[i].set_ylabel(f"Joint {i+1} [Nm]", fontsize=10)
        #     axes[i].grid(True, alpha=0.3)
        # axes[-1].set_xlabel("Time [s]", fontsize=10)
        # plt.tight_layout()
        # out = output_dir / "joint_efforts.png"
        # plt.savefig(out, dpi=150, bbox_inches="tight")
        # plt.close()
        # print(f"  ✅ Saved to {out}")

    # 2. Plot External Joint Torques
    if "external_torques" in data_dict:
        # User-configurable options for this plot
        ext_x_axis_limit = None  # e.g. (5.0, 20.0) or None
        ext_plot_joints = None  # e.g. [0,2,4] (0-based indices) or None for all joints

        torques = np.array([d["effort"] for d in data_dict["external_torques"]["data"]])
        timestamps = data_dict["external_torques"]["timestamps"]

        # Determine which joints to plot
        if ext_plot_joints is None:
            joints = list(range(7))
        else:
            joints = [j for j in ext_plot_joints if 0 <= j < 7]

        total = len(joints)
        fig, axes = plt.subplots(total, 1, figsize=(12, 2.0 * total))
        fig.suptitle("External Joint Torques", fontsize=16)

        if total == 1:
            axes = np.array([axes])

        # Apply x-axis limit if requested
        if ext_x_axis_limit is not None:
            ex0, ex1 = ext_x_axis_limit
            ts_mask = np.isfinite(timestamps) & (timestamps >= ex0) & (timestamps <= ex1)
            ts_plot = timestamps[ts_mask]
            torques_plot = torques[ts_mask]
        else:
            ts_plot = timestamps
            torques_plot = torques

        for idx, j in enumerate(joints):
            axes[idx].plot(ts_plot, torques_plot[:, j], linewidth=1, color="blue")
            axes[idx].set_ylabel(f"Joint {j + 1} [Nm]", fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            if idx == total - 1:
                axes[idx].set_xlabel("Time [s]", fontsize=10)

        plt.tight_layout()
        output_path = output_dir / "joint_external_torques.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ Saved to {output_path}")
        plt.close()

    # 2b. Plot external wrench (force/torque in stiffness frame)
    if "external_wrench" in data_dict:
        print("  Plotting external wrench in stiffness frame...")
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle("External Wrench in Stiffness Frame", fontsize=16)

        external_wrench = np.array([d["external_wrench"] for d in data_dict["external_wrench"]["data"]])
        ts = data_dict["external_wrench"]["timestamps"]
        # Extract wrench components (force and torque)

        # Force components
        axes[0].plot(ts, external_wrench[:, 0], label="Fx", color="tab:blue")
        axes[0].plot(ts, external_wrench[:, 1], label="Fy", color="tab:orange")
        axes[0].plot(ts, external_wrench[:, 2], label="Fz", color="tab:green")
        axes[0].set_ylabel("Force [N]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8)

        # Torque components
        axes[1].plot(ts, external_wrench[:, 3], label="Tx", color="tab:red")
        axes[1].plot(ts, external_wrench[:, 4], label="Ty", color="tab:purple")
        axes[1].plot(ts, external_wrench[:, 5], label="Tz", color="tab:brown")
        axes[1].set_ylabel("Torque [Nm]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right", fontsize=8)

        # Force norm
        f_norm = np.sqrt(external_wrench[:, 0] ** 2 + external_wrench[:, 1] ** 2 + external_wrench[:, 2] ** 2)
        axes[2].plot(ts, f_norm, label="|F|", color="black")
        axes[2].set_ylabel("Force Norm [N]")
        axes[2].set_xlabel("Time [s]")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        out = output_dir / "ee_external_wrench.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  ✅ Saved to {out}")
        plt.close()

    # 3. Plot gripper state and command
    if "gripper_state" in data_dict or "gripper_cmd" in data_dict:
        print("  Plotting gripper state/command...")
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        fig.suptitle("Gripper Position", fontsize=14)

        if "gripper_state" in data_dict:
            ts = data_dict["gripper_state"]["timestamps"]
            pos = _extract_scalar_from_jointstate(data_dict["gripper_state"]["data"], "position", index=0)
            ax.plot(ts, pos, label="Gripper State", linewidth=1.5, color="tab:blue")

        if "gripper_cmd" in data_dict:
            ts = data_dict["gripper_cmd"]["timestamps"]
            pos = _extract_scalar_from_jointstate(data_dict["gripper_cmd"]["data"], "position", index=0)
            ax.plot(ts, pos, label="Gripper Command", linewidth=1.5, color="tab:red", alpha=0.8)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        out = output_dir / "gripper_position.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Saved to {out}")

    # 4. Plot tracking error
    if "tracking_error" in data_dict:
        print("  Plotting tracking error...")
        ts = data_dict["tracking_error"]["timestamps"]
        err = _extract_vector_series(data_dict["tracking_error"]["data"], key="data")
        if err.size > 0 and err.ndim == 2 and err.shape[1] >= 6:
            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            fig.suptitle("Cartesian Tracking Error", fontsize=16)

            labels = ["x", "y", "z"]
            for i in range(3):
                axes[0].plot(ts, err[:, i], linewidth=1.2, label=labels[i])
            axes[0].set_ylabel("Translation Error")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(loc="upper right", fontsize=9)

            labels = ["rx", "ry", "rz"]
            for i in range(3):
                axes[1].plot(ts, err[:, i + 3], linewidth=1.2, label=labels[i])
            axes[1].set_ylabel("Rotation Error")
            axes[1].set_xlabel("Time [s]")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc="upper right", fontsize=9)

            out = output_dir / "tracking_error.png"
            plt.tight_layout()
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  ✅ Saved to {out}")

    # 4b. Plot end-effector velocity
    if "ee_velocity" in data_dict:
        print("  Plotting end-effector velocity...")
        ts = data_dict["ee_velocity"]["timestamps"]
        vel = _extract_vector_series(data_dict["ee_velocity"]["data"], key="data")
        if vel.size > 0 and vel.ndim == 2 and vel.shape[1] >= 6:
            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            fig.suptitle("End-Effector Velocity", fontsize=16)

            labels = ["vx", "vy", "vz"]
            for i in range(3):
                axes[0].plot(ts, vel[:, i], linewidth=1.2, label=labels[i])
            axes[0].set_ylabel("Translation Velocity")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(loc="upper right", fontsize=9)

            labels = ["wx", "wy", "wz"]
            for i in range(3):
                axes[1].plot(ts, vel[:, i + 3], linewidth=1.2, label=labels[i])
            axes[1].set_ylabel("Rotation Velocity")
            axes[1].set_xlabel("Time [s]")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc="upper right", fontsize=9)

            out = output_dir / "ee_velocity.png"
            plt.tight_layout()
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  ✅ Saved to {out}")

    # # 4. Combined plot: Measured vs Commanded positions
    # if 'measured_joints' in data_dict and 'impedance_cmd' in data_dict:
    #     fig, axes = plt.subplots(7, 1, figsize=(12, 14))
    #     fig.suptitle('Joint Positions: Measured vs Commanded', fontsize=16)

    #     measured_pos = np.array([d['position'] for d in data_dict['measured_joints']['data']])
    #     measured_ts = data_dict['measured_joints']['timestamps']

    #     commanded_pos = np.array([d['position'] for d in data_dict['impedance_cmd']['data']])
    #     commanded_ts = data_dict['impedance_cmd']['timestamps']

    #     for i in range(7):
    #         axes[i].plot(measured_ts, measured_pos[:, i], linewidth=2, label='Measured', alpha=0.7)
    #         axes[i].plot(commanded_ts, commanded_pos[:, i], linewidth=2, label='Commanded', alpha=0.7, linestyle='--')
    #         axes[i].set_ylabel(f'Joint {i+1} [rad]', fontsize=10)
    #         axes[i].grid(True, alpha=0.3)
    #         axes[i].legend(loc='upper right', fontsize=8)
    #         if i == 6:
    #             axes[i].set_xlabel('Time [s]', fontsize=10)

    #     plt.tight_layout()
    #     output_path = output_dir / 'measured_vs_commanded.png'
    #     plt.savefig(output_path, dpi=150, bbox_inches='tight')
    #     print(f"  ✅ Saved to {output_path}")
    #     plt.close()

    # 8. Plot Predictions (expanded to 25Hz, batches NOT connected)
    if "predictions" in data_dict and "raw_predictions" in data_dict:
        pred_batches = data_dict["predictions"]["data"]
        pred_ts_raw = data_dict["predictions"]["timestamps"]

        pred_batches_raw = data_dict["raw_predictions"]["data"]
        pred_ts_raw_raw = data_dict["raw_predictions"]["timestamps"]

        expanded_predictions = []
        expanded_timestamps = []
        expanded_predictions_raw = []
        expanded_timestamps_raw = []
        last_batch = None  # Keep track of the last batch
        valid_pred_indices = []
        valid_raw_indices = []

        for idx, (ts, batch) in enumerate(zip(pred_ts_raw, pred_batches)):
            if "positions" not in batch:
                continue
            preds = batch["positions"]  # list of 25 preds
            n = len(preds)
            if n == 0 or batch == last_batch:  # Skip if the batch is empty or the same as the last one
                continue
            valid_pred_indices.append(idx)
            dt = 1.0 / 30.0
            # Add the batch
            for i, p in enumerate(preds):
                expanded_predictions.append(p)
                expanded_timestamps.append(ts + i * dt)

            # NaN separator → prevents matplotlib from connecting batches
            expanded_predictions.append([np.nan] * len(preds[0]))
            expanded_timestamps.append(np.nan)

            last_batch = batch  # Update the last batch
        for idx, (ts, batch) in enumerate(zip(pred_ts_raw_raw, pred_batches_raw)):
            if "positions" not in batch:
                continue
            preds = batch["positions"]  # list of 25 preds
            n = len(preds)
            if n == 0 or batch == last_batch:
                continue
            valid_raw_indices.append(idx)
            dt = 1.0 / 30.0
            # Add the batch
            for i, p in enumerate(preds):
                expanded_predictions_raw.append(p)
                expanded_timestamps_raw.append(ts + i * dt)

            # NaN separator → prevents matplotlib from connecting batches
            expanded_predictions_raw.append([np.nan] * len(preds[0]))
            expanded_timestamps_raw.append(np.nan)

            last_batch = batch  # Update the last batch

        expanded_predictions = np.array(expanded_predictions)
        expanded_timestamps = np.array(expanded_timestamps)
        expanded_predictions_raw = np.array(expanded_predictions_raw)
        expanded_timestamps_raw = np.array(expanded_timestamps_raw)

        # fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        # fig.suptitle('Active Inference Predictions (30 Hz)', fontsize=16)

        # for dim in range(expanded_predictions.shape[1]):
        #     ax.plot(expanded_timestamps,
        #             expanded_predictions[:, dim],
        #             linewidth=1.5,
        #             label=f"Dim {dim}")

        # ax.set_xlabel("Time [s]")
        # ax.set_ylabel("Prediction Value")
        # ax.grid(True, alpha=0.3)
        # ax.legend(fontsize=8)
        # out = output_dir / 'predictions_25hz.png'
        # plt.tight_layout()
        # plt.savefig(out, dpi=150)
        # plt.close()
        # print(f"  ✅ Saved to {out}")

    # 9. Add predictions to measured vs commanded joint plot
    if (
        "measured_joints" in data_dict
        and "impedance_cmd" in data_dict
        and "predictions" in data_dict
        and "raw_predictions" in data_dict
    ):
        measured_pos = np.array([d["position"] for d in data_dict["measured_joints"]["data"]])
        meas_ts = data_dict["measured_joints"]["timestamps"]

        commanded_pos = np.array([d["position"] for d in data_dict["impedance_cmd"]["data"]])
        cmd_ts = data_dict["impedance_cmd"]["timestamps"]

        # Use expanded predictions already computed
        pred_pos = expanded_predictions
        pred_ts = expanded_timestamps
        raw_predictions = expanded_predictions_raw
        raw_predictions_ts = expanded_timestamps_raw

        # Check if entropy (8th dim) is available
        entropy_available = False
        try:
            if pred_pos.size > 0 and pred_pos.shape[1] > 7:
                entropy_available = True
        except Exception:
            entropy_available = False

        # Check if gains data exists (impedance_cmd)
        gains_available = False
        pct_mean = None
        gains_ts = None
        try:
            imp_data_local = data_dict["impedance_cmd"]["data"]
            imp_ts_local = data_dict["impedance_cmd"]["timestamps"]
            k_vals_local = []
            d_vals_local = []
            for msg in imp_data_local:
                k = None
                d = None
                if isinstance(msg, dict):
                    k = msg.get("k_gains") or msg.get("k") or msg.get("stiffness")
                    d = msg.get("d_gains") or msg.get("d") or msg.get("damping")
                if k is None or d is None:
                    try:
                        k = k or msg["command"]["k_gains"]
                        d = d or msg["command"]["d_gains"]
                    except Exception:
                        pass
                if k is None or d is None:
                    continue
                k_vals_local.append(np.array(k))
                d_vals_local.append(np.array(d))

            if len(k_vals_local) > 0:
                k_vals_local = np.vstack(k_vals_local)
                d_vals_local = np.vstack(d_vals_local)
                k_soft = np.array([69.89, 86.61, 232.11, 91.75, 32.38, 17.76, 10.49])
                d_soft = np.array([7.66, 7.36, 12.86, 6.61, 2.18, 1.11, 0.99])
                k_stiff = np.array([305.62, 303.85, 449.42, 309.65, 200.14, 110.16, 105.80])
                d_stiff = np.array([37.15, 36.63, 46.69, 33.14, 17.06, 9.16, 9.41])
                pct_k = (k_vals_local - k_soft) / (k_stiff - k_soft) * 100.0
                pct_d = (d_vals_local - d_soft) / (d_stiff - d_soft) * 100.0
                pct_k = np.clip(pct_k, 0.0, 100.0)
                pct_d = np.clip(pct_d, 0.0, 100.0)
                pct_k_mean = np.nanmean(pct_k, axis=1)
                pct_d_mean = np.nanmean(pct_d, axis=1)
                pct_mean = 0.5 * (pct_k_mean + pct_d_mean)
                gains_available = True
                gains_ts = imp_ts_local
        except Exception:
            gains_available = False

        ## FOR VIDEO DATA
        video_data, video_ts = extract_topic_data(pkl_data, "/realsense/front/color/image_raw")

        video_available = video_data is not None and len(video_data) > 0

        ########################### User-configurable plotting options:
        # - Set `x_axis_limit` to a (start, end) tuple in seconds to plot only that time window.
        #   Set to `None` to plot the full range.
        # - Set `exclude_joints` to a list of 0-based joint indices to skip plotting those joints.
        # Normal plot:
        x_axis_limit = None  # e.g. (5.0, 20.0) or None
        exclude_joints = []  # e.g. [2, 4]
        width_of_plot = 16  # 16
        # Give the whole figure a bit more vertical space when the
        # video row is present so that the images are not squished.
        additional_height_plot = 2 if video_available else 0
        # Zoomed plot:
        # x_axis_limit = (7.0, 15.0)  # e.g. (5.0, 20.0) or None
        # exclude_joints = [0, 1, 2, 3, 4, 5] # e.g. [2, 4]
        # width_of_plot = 8 # 16
        # additional_height_plot = 1

        extra_rows = 0
        if entropy_available:
            extra_rows += 1
        if gains_available:
            extra_rows += 1
        if video_available:
            extra_rows += 1

        # Build the list of joints to plot (0-based indices)
        joints_to_plot = [j for j in range(7) if j not in exclude_joints]
        total_rows = len(joints_to_plot) + extra_rows
        fig, axes = plt.subplots(
            total_rows, 1, figsize=(width_of_plot, 2.2 * total_rows + additional_height_plot), sharex=True
        )
        fig.suptitle("Measured, Commanded, Predictions and Entropy/Gains", fontsize=16, y=0.99)

        # Ensure axes is indexable
        if total_rows == 1:
            axes = np.array([axes])

        # Optionally limit time window for all series
        if x_axis_limit is not None:
            x0, x1 = x_axis_limit
            # measured
            mask = np.isfinite(meas_ts) & (meas_ts >= x0) & (meas_ts <= x1)
            meas_ts_plot = meas_ts[mask]
            measured_pos_plot = measured_pos[mask]
            # commanded
            mask = np.isfinite(cmd_ts) & (cmd_ts >= x0) & (cmd_ts <= x1)
            cmd_ts_plot = cmd_ts[mask]
            commanded_pos_plot = commanded_pos[mask]
            # predictions
            # Preserve NaN separators so matplotlib doesn't connect batches
            pred_mask = np.isnan(pred_ts) | ((pred_ts >= x0) & (pred_ts <= x1))
            pred_ts_plot = pred_ts[pred_mask]
            pred_pos_plot = pred_pos[pred_mask]
            raw_mask = np.isnan(raw_predictions_ts) | ((raw_predictions_ts >= x0) & (raw_predictions_ts <= x1))
            raw_predictions_ts_plot = raw_predictions_ts[raw_mask]
            raw_predictions_plot = raw_predictions[raw_mask]
        else:
            meas_ts_plot = meas_ts
            measured_pos_plot = measured_pos
            cmd_ts_plot = cmd_ts
            commanded_pos_plot = commanded_pos
            pred_ts_plot = pred_ts
            pred_pos_plot = pred_pos
            raw_predictions_ts_plot = raw_predictions_ts
            raw_predictions_plot = raw_predictions

        if video_available:
            # Align video timestamps to the same global time base as the
            # other topics (measured joints, commands, predictions).
            raw_video_ts = np.array(video_ts, dtype=np.float64)

            # Convert from nanoseconds to seconds if needed
            if np.nanmax(raw_video_ts) > 1e12:
                raw_video_ts = raw_video_ts * 1e-9

            # Shift by the same global_t0 that was used for all data_dict topics
            video_ts = raw_video_ts - global_t0

            # Respect x-axis limits
            if x_axis_limit is not None:
                mask = (video_ts >= x0) & (video_ts <= x1)
                video_ts = video_ts[mask]
                video_data = [d for d, m in zip(video_data, mask) if m]

            # Downsample the video for plotting so that frames are displayed every x s.
            # NOTE: We intentionally do NOT keep the very first frame (index 0).
            dt = 2.0
            keep_idx = []
            if len(video_ts) > 0:
                last_t = video_ts[0]
                for i in range(1, len(video_ts)):
                    if video_ts[i] - last_t >= dt:
                        keep_idx.append(i)
                        last_t = video_ts[i]

                # Fallback: if dt is too large and nothing was selected, keep a non-zero frame if possible.
                if len(keep_idx) == 0:
                    if len(video_ts) > 1:
                        keep_idx = [1]
                    else:
                        keep_idx = [0]

            video_ts = video_ts[keep_idx]
            video_frames = [ros_image_to_numpy(video_data[i]) for i in keep_idx]

        for idx, j in enumerate(joints_to_plot):
            ax = axes[idx]
            # Plot measured
            if measured_pos_plot.size > 0:
                ax.plot(
                    meas_ts_plot, measured_pos_plot[:, j], label="Measured", linewidth=1.5, alpha=0.8, color="black"
                )

            # Only plot predictions for matching dimension
            if pred_pos_plot.size > 0 and j < pred_pos_plot.shape[1]:
                ax.plot(
                    pred_ts_plot,
                    pred_pos_plot[:, j],
                    label="Predictions ensembled",
                    linewidth=1.4,
                    alpha=0.7,
                    color="blue",
                )
            if raw_predictions_plot.size > 0 and j < raw_predictions_plot.shape[1]:
                ax.plot(
                    raw_predictions_ts_plot,
                    raw_predictions_plot[:, j],
                    label="Predictions raw",
                    linewidth=1.4,
                    alpha=0.35,
                    color="grey",
                )

            # Add circle markers only for batches that were actually expanded (and within x range if set)
            for k, vidx in enumerate(valid_pred_indices):
                batch = pred_batches[vidx]
                ts_batch = pred_ts_raw[vidx]
                if x_axis_limit is not None:
                    if not (np.isfinite(ts_batch) and (ts_batch >= x0) and (ts_batch <= x1)):
                        continue
                if "positions" in batch and len(batch["positions"]) > 0:
                    ax.plot(
                        ts_batch,
                        batch["positions"][0][j],
                        "o",
                        color="blue",
                        markersize=3,
                        label="Batch Start (ensembled)" if idx == 0 and k == 0 else "",
                    )

            for k, vidx in enumerate(valid_raw_indices):
                batch = pred_batches_raw[vidx]
                ts_batch = pred_ts_raw_raw[vidx]
                if x_axis_limit is not None:
                    if not (np.isfinite(ts_batch) and (ts_batch >= x0) and (ts_batch <= x1)):
                        continue
                if "positions" in batch and len(batch["positions"]) > 0:
                    ax.plot(
                        ts_batch,
                        batch["positions"][0][j],
                        "o",
                        color="grey",
                        markersize=2,
                        label="Batch Start (raw)" if idx == 0 and k == 0 else "",
                    )

            # Commanded
            if commanded_pos_plot.size > 0:
                ax.plot(cmd_ts_plot, commanded_pos_plot[:, j], label="Commanded", linewidth=1.3, alpha=1, color="red")

            ax.set_ylabel(f"Joint {j + 1}")
            ax.grid(True, alpha=0.3)
            if idx == len(joints_to_plot) - 1:
                if not entropy_available and not gains_available:
                    ax.set_xlabel("Time [s]")
            ax.legend(fontsize=8, loc="upper right")

        # Plot entropy and gains in the extra rows, if available
        row_idx = len(joints_to_plot)
        if entropy_available:
            ax_ent = axes[row_idx]
            ent_ensembled = pred_pos[:, 7]
            ent_raw = raw_predictions[:, 7]
            # Apply x-axis limit if configured
            if x_axis_limit is not None:
                # Preserve NaN separators so matplotlib doesn't connect entropy batches
                ent_mask = np.isnan(pred_ts) | ((pred_ts >= x0) & (pred_ts <= x1))
                raw_ent_mask = np.isnan(raw_predictions_ts) | ((raw_predictions_ts >= x0) & (raw_predictions_ts <= x1))
                ent_ts_plot = pred_ts[ent_mask]
                ent_ensembled_plot = ent_ensembled[ent_mask]
                raw_ent_ts_plot = raw_predictions_ts[raw_ent_mask]
                ent_raw_plot = ent_raw[raw_ent_mask]
                ax_ent.plot(
                    ent_ts_plot, ent_ensembled_plot, label="Variance ensembled", color="tab:blue", linewidth=1.2
                )
                ax_ent.plot(raw_ent_ts_plot, ent_raw_plot, label="Variance raw", color="gray", linewidth=1.0, alpha=0.6)
            else:
                ax_ent.plot(pred_ts, ent_ensembled, label="Variance ensembled", color="tab:blue", linewidth=1.2)
                ax_ent.plot(raw_predictions_ts, ent_raw, label="Variance raw", color="gray", linewidth=1.0, alpha=0.6)
            ax_ent.set_ylabel("Variance")
            ax_ent.set_ylim(0, 1)
            ax_ent.grid(True, alpha=0.3)
            ax_ent.legend(fontsize=8, loc="upper right")
            row_idx += 1

        if gains_available:
            ax_g = axes[row_idx]
            # Apply x-axis limit to gains if configured
            if x_axis_limit is not None:
                gains_mask = np.isfinite(gains_ts) & (gains_ts >= x0) & (gains_ts <= x1)
                gains_ts_plot = gains_ts[gains_mask]
                pct_mean_plot = np.array(pct_mean)[gains_mask]
                ax_g.plot(gains_ts_plot, pct_mean_plot, label="Mean Gains Percent", color="tab:red", linewidth=1.4)
            else:
                ax_g.plot(gains_ts, pct_mean, label="Mean Gains Percent", color="tab:red", linewidth=1.4)
            ax_g.set_ylabel("Impedance Gains % (0=soft,100=stiff)")
            ax_g.set_xlabel("Time [s]")
            ax_g.set_ylim(-5, 105)
            ax_g.grid(True, alpha=0.3)
            ax_g.legend(fontsize=8, loc="upper right")
            row_idx += 1

        if video_available:
            ax_vid = axes[row_idx]

            # Use the (possibly downsampled) video timestamps/frames to
            # place images across the whole time range. Optionally cap
            # the maximum number of frames to avoid clutter.
            max_frames = 25
            if len(video_ts) > max_frames:
                idxs = np.linspace(0, len(video_ts) - 1, max_frames).astype(int)
            else:
                idxs = np.arange(len(video_ts))

            # Defer actual drawing of images until after tight_layout(),
            # so we can compute a good pixel-based zoom that fills the
            # subplot height while preserving the image aspect ratio.
            video_ts_plot = [float(video_ts[i]) for i in idxs]
            video_frames_plot = [video_frames[i] for i in idxs]

            # ax_vid.set_facecolor("#f2f2f2")

            ax_vid.set_yticks([])
            ax_vid.set_ylabel("Images")
            ax_vid.set_xlabel("Time [s]")
            ax_vid.set_ylim(0, 1)
            ax_vid.grid(False)

        # Ensure all subplots share a sensible global x-axis range when no
        # explicit x_axis_limit is provided, so that the figure is not
        # unintentionally zoomed to a tiny window.
        if x_axis_limit is None:
            ts_list = []
            for arr in [meas_ts_plot, cmd_ts_plot, pred_ts_plot, raw_predictions_ts_plot]:
                if arr is not None:
                    arr_np = np.asarray(arr, dtype=float)
                    finite = arr_np[np.isfinite(arr_np)]
                    if finite.size > 0:
                        ts_list.append(finite)

            if ts_list:
                all_ts = np.concatenate(ts_list)
                t_min = float(np.nanmin(all_ts))
                t_max = float(np.nanmax(all_ts))
                if t_max <= t_min:
                    t_max = t_min + 1.0
                span = t_max - t_min
                pad = max(0.2, 0.05 * span)
                axes[0].set_xlim(t_min, t_max + pad)

        out = output_dir / "measured_commanded_predictions_entropy_gains.png"
        # Layout first so axis pixel sizes are stable.
        plt.tight_layout()

        # Draw video frames as pixel-sized artists (no stretching).
        if video_available:
            fig.canvas.draw()
            bbox = ax_vid.get_window_extent()
            ax_x0, ax_x1 = ax_vid.get_xlim()
            x_range = max(1e-9, float(ax_x1 - ax_x0))

            # Target: fill most of the subplot height.
            desired_h_px = 0.45 * float(
                bbox.height
            )  ##################### set height and therefore also width of images

            # If frames are dense, cap zoom so they don't overlap too badly.
            if len(video_ts_plot) >= 2:
                ts_sorted = np.sort(np.asarray(video_ts_plot, dtype=float))
                dts = np.diff(ts_sorted)
                dt_min = float(np.min(dts[dts > 0])) if np.any(dts > 0) else None
            else:
                dt_min = None

            for t, frame in zip(video_ts_plot, video_frames_plot):
                # Compute zoom from height (keeps square images square).
                h, w = frame.shape[:2]
                zoom_h = desired_h_px / max(1.0, float(h))

                zoom = zoom_h
                if dt_min is not None:
                    spacing_px = (dt_min / x_range) * float(bbox.width)
                    zoom_w_max = 0.90 * spacing_px / max(1.0, float(w))
                    zoom = min(zoom, zoom_w_max)

                image = OffsetImage(frame, zoom=max(0.01, zoom))

                # If the frame is near the left/right boundary, align
                # it so it doesn't get clipped.
                half_w_px = 0.5 * max(1.0, float(w)) * max(0.01, zoom)
                half_w_data = (half_w_px / max(1.0, float(bbox.width))) * x_range

                if t <= ax_x0 + half_w_data:
                    box_alignment = (0.0, 0.5)  # left-align
                elif t >= ax_x1 - half_w_data:
                    box_alignment = (1.0, 0.5)  # right-align
                else:
                    box_alignment = (0.5, 0.5)  # center

                ab = AnnotationBbox(
                    image,
                    (t, 0.5),
                    xycoords="data",
                    frameon=True,
                    pad=0.15,
                    bboxprops={"edgecolor": "black", "linewidth": 0.5, "alpha": 0.75},
                    box_alignment=box_alignment,
                    zorder=5,
                )
                ax_vid.add_artist(ab)

        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  ✅ Saved to {out}")

    # 3. Entropy (8th prediction dimension) and Gains percent
    if "predictions" in data_dict and "raw_predictions" in data_dict and "impedance_cmd" in data_dict:
        try:
            print("  Plotting entropy (8th prediction dim) and gains percentage...")

            # Attempt to get the 8th dimension (index 7) from expanded predictions
            if expanded_predictions.size == 0 or expanded_predictions_raw.size == 0:
                raise ValueError("Expanded prediction arrays are empty")

            # If predictions have fewer than 8 dims, this will raise and be caught
            entropy_ensembled = expanded_predictions[:, 7]
            entropy_raw = expanded_predictions_raw[:, 7]

            # Timestamps for these series
            entropy_ts_ensembled = expanded_timestamps
            entropy_ts_raw = expanded_timestamps_raw

            # Extract gains from impedance command topic
            imp_data = data_dict["impedance_cmd"]["data"]
            imp_ts = data_dict["impedance_cmd"]["timestamps"]

            k_vals = []
            d_vals = []
            k_t = []
            for msg in imp_data:
                # Robust extraction: look for common keys
                k = None
                d = None
                if isinstance(msg, dict):
                    k = msg.get("k_gains") or msg.get("k") or msg.get("stiffness")
                    d = msg.get("d_gains") or msg.get("d") or msg.get("damping")
                # If still None and msg has nested structures, try to find keys heuristically
                if k is None or d is None:
                    # try attributes-like access
                    try:
                        k = k or msg["command"]["k_gains"]
                        d = d or msg["command"]["d_gains"]
                    except Exception:
                        pass

                if k is None or d is None:
                    # skip if gains not present
                    continue

                k_vals.append(np.array(k))
                d_vals.append(np.array(d))

            if len(k_vals) == 0:
                print("  ❌ No gain messages found in impedance_cmd topic; skipping gains plot")

                ################# Gains Plot #############
            # else:
            #     k_vals = np.vstack(k_vals)
            #     d_vals = np.vstack(d_vals)

            #     # Reference soft/stiff gains (from inference_parameters.yaml)
            #     k_soft = np.array([69.89, 86.61, 232.11, 91.75, 32.38, 17.76, 10.49])
            #     d_soft = np.array([7.66, 7.36, 12.86, 6.61, 2.18, 1.11, 0.99])

            #     k_stiff = np.array([305.62, 303.85, 449.42, 309.65, 200.14, 110.16, 105.80])
            #     d_stiff = np.array([37.15, 36.63, 46.69, 33.14, 17.06, 9.16, 9.41])

            #     # Compute percent [0..100] per joint, clamp
            #     pct_k = (k_vals - k_soft) / (k_stiff - k_soft) * 100.0
            #     pct_d = (d_vals - d_soft) / (d_stiff - d_soft) * 100.0
            #     pct_k = np.clip(pct_k, 0.0, 100.0)
            #     pct_d = np.clip(pct_d, 0.0, 100.0)

            #     # Aggregate across joints (mean percent). You can change to plot per-joint if desired
            #     pct_k_mean = np.nanmean(pct_k, axis=1)
            #     pct_d_mean = np.nanmean(pct_d, axis=1)
            #     pct_mean = 0.5 * (pct_k_mean + pct_d_mean)

            #     # Build timestamps array for gains (imp_ts already in seconds and offset earlier)
            #     gains_ts = imp_ts

            #     # Create figure with two subplots: entropy on top, gains % below
            #     fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            #     fig.suptitle('Entropy (8th pred dim) and Controller Gains Percent (0% soft → 100% stiff)', fontsize=14)

            #     # Entropy plot
            #     axes[0].plot(entropy_ts_ensembled, entropy_ensembled, label='Entropy (ensembled)', color='tab:blue', linewidth=1.5)
            #     axes[0].plot(entropy_ts_raw, entropy_raw, label='Entropy (raw)', color='gray', linewidth=1.0, alpha=0.6)
            #     axes[0].set_ylabel('Entropy value (pred dim 8)')
            #     axes[0].grid(True, alpha=0.3)
            #     axes[0].legend(fontsize=8)

            #     # Gains percent plot
            #     axes[1].plot(gains_ts, pct_mean, label='Mean Gains Percent (k+d)', color='tab:red', linewidth=1.5)
            #     axes[1].set_ylabel('Gains percent [%] (0=soft, 100=stiff)')
            #     axes[1].set_xlabel('Time [s]')
            #     axes[1].set_ylim(-5, 105)
            #     axes[1].grid(True, alpha=0.3)
            #     axes[1].legend(fontsize=8)

            #     # Prefer using the same time range as the "Measured vs Commanded vs Predictions" plot.
            #     # That plot uses `meas_ts` and `cmd_ts` when present. Fall back to gains timestamps.
            #     try:
            #         if 'measured_joints' in data_dict:
            #             mts = np.array(meas_ts)
            #             cts = np.array(cmd_ts)
            #             parts = []
            #             for arr in (mts, cts):
            #                 if isinstance(arr, np.ndarray) and arr.size > 0:
            #                     valid = arr[~np.isnan(arr)]
            #                     if valid.size > 0:
            #                         parts.append(valid)
            #             if parts:
            #                 all_ts = np.concatenate(parts)
            #                 min_t = float(np.nanmin(all_ts))
            #                 max_t = float(np.nanmax(all_ts))
            #                 span = max_t - min_t
            #                 if span <= 0:
            #                     span = 1.0
            #                 pad = max(0.5, span * 0.02)
            #                 axes[0].set_xlim(min_t, max_t + pad)
            #         else:
            #             gts = np.array(gains_ts)
            #             if gts.size > 0:
            #                 valid_gts = gts[~np.isnan(gts)]
            #                 if valid_gts.size > 0:
            #                     min_t = float(np.nanmin(valid_gts))
            #                     max_t = float(np.nanmax(valid_gts))
            #                     span = max_t - min_t
            #                     if span <= 0:
            #                         span = 1.0
            #                     pad = max(0.5, span * 0.02)
            #                     axes[0].set_xlim(min_t, max_t + pad)
            #     except Exception:
            #         pass

            #     out = output_dir / 'entropy_and_gains.png'
            #     plt.tight_layout()
            #     plt.savefig(out, dpi=200, bbox_inches='tight')
            #     plt.close()
            #     print(f"  ✅ Saved to {out}")

        except Exception as e:
            print(f"  ❌ Could not create entropy/gains plot: {e}")

    # 9. Plot Attention
    if "attention" in data_dict:
        att = np.array([d["data"] for d in data_dict["attention"]["data"]])
        ts = data_dict["attention"]["timestamps"]
        # Reshape the attention array to timesteps x 6 x 6
        att_reshaped = np.zeros((att.shape[0], 6, 6))
        for t in range(att.shape[0]):
            att_reshaped[:, :, 0] = att[:, :6]  # First 6 (image) are in 3rd dimension 0
            att_reshaped[:, :, 1] = att[:, 6:]  # Last 6 (torque) are in 3rd dimension 1

        fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
        fig.suptitle("Attention to Force and Image", fontsize=16)

        ax1.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.6)
        for layer_idx in range(att_reshaped.shape[1]):
            linestyle = "-" if layer_idx == 0 else "-" if layer_idx == att_reshaped.shape[1] - 1 else "--"
            colour = (
                "blue" if layer_idx == 0 else "black" if layer_idx == att_reshaped.shape[1] - 1 else f"C{layer_idx}"
            )
            alpha = 1.0 if layer_idx == 0 else 1.0 if layer_idx == att_reshaped.shape[1] - 1 else 0.7
            ax1.plot(
                ts,
                (att_reshaped[:, layer_idx, 1] - att_reshaped[:, layer_idx, 0]),
                label=f"Layer {layer_idx + 1} - (Torque - Image)",
                linewidth=1.8,
                linestyle=linestyle,
                color=colour,
                alpha=alpha,
            )

        ax1.set_ylabel("Mean attention weight")
        ax1.legend(loc="lower right")
        ax1.set_ylim(-1, 1)
        ax1.grid(True, alpha=0.4)

        # Add labels for -1 and 1 limits
        ax1.text(ax1.get_xlim()[0] + 0.1, -0.9, "100% Image", va="center", ha="left", fontsize=12, color="gray")
        ax1.text(ax1.get_xlim()[0] + 0.1, 0.9, "100% Torque", va="center", ha="left", fontsize=12, color="gray")

        ax1.set_xlabel("Time [s]")
        ax1.legend(fontsize=8)

        out = output_dir / "attention.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  ✅ Saved to {out}")

    # 10. Plot end-effector position and orientation (robot_state + pose_command)
    if "robot_state" in data_dict or "pose_command" in data_dict:
        print("  Plotting end-effector pose...")

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle("End-Effector Pose", fontsize=16)

        ee_pose_ts = None
        ee_pos = None
        ee_euler = None
        cmd_ts = None
        cmd_pos = None
        cmd_euler = None

        if "robot_state" in data_dict:
            ee_pose_ts = data_dict["robot_state"]["timestamps"]
            ee_pos, ee_euler = _extract_pose_series(data_dict["robot_state"]["data"], "ee_pose")

        if "pose_command" in data_dict:
            cmd_ts = data_dict["pose_command"]["timestamps"]
            cmd_pos, cmd_euler = _extract_pose_series(data_dict["pose_command"]["data"], "ee_pose_commanded")

        euler_series = []
        if ee_euler is not None:
            euler_series.append(ee_euler)
        if cmd_euler is not None:
            euler_series.append(cmd_euler)
        if euler_series:
            corrected_eulers = _apply_ee_rotation_wrap_correction(
                euler_series,
                enabled=fix_ee_rotation_wrap,
            )
            i = 0
            if ee_euler is not None:
                ee_euler = corrected_eulers[i]
                i += 1
            if cmd_euler is not None:
                cmd_euler = corrected_eulers[i]

        if ee_pose_ts is not None and ee_pos is not None and ee_euler is not None:
            axes[0].plot(ee_pose_ts, ee_pos[:, 0], label="x (state)", color="tab:blue")
            axes[0].plot(ee_pose_ts, ee_pos[:, 1], label="y (state)", color="tab:orange")
            axes[0].plot(ee_pose_ts, ee_pos[:, 2], label="z (state)", color="tab:green")
            axes[1].plot(ee_pose_ts, ee_euler[:, 0], label="roll (state)", color="tab:red")
            axes[1].plot(ee_pose_ts, ee_euler[:, 1], label="pitch (state)", color="tab:purple")
            axes[1].plot(ee_pose_ts, ee_euler[:, 2], label="yaw (state)", color="tab:brown")

        if cmd_ts is not None and cmd_pos is not None and cmd_euler is not None:
            axes[0].plot(cmd_ts, cmd_pos[:, 0], label="x (cmd)", color="tab:blue", alpha=0.6, linestyle="--")
            axes[0].plot(cmd_ts, cmd_pos[:, 1], label="y (cmd)", color="tab:orange", alpha=0.6, linestyle="--")
            axes[0].plot(cmd_ts, cmd_pos[:, 2], label="z (cmd)", color="tab:green", alpha=0.6, linestyle="--")
            axes[1].plot(cmd_ts, cmd_euler[:, 0], label="roll (cmd)", color="tab:red", alpha=0.6, linestyle="--")
            axes[1].plot(cmd_ts, cmd_euler[:, 1], label="pitch (cmd)", color="tab:purple", alpha=0.6, linestyle="--")
            axes[1].plot(cmd_ts, cmd_euler[:, 2], label="yaw (cmd)", color="tab:brown", alpha=0.6, linestyle="--")

        axes[0].set_ylabel("Position [m]")
        axes[0].grid(True, alpha=0.3)
        axes[1].set_ylabel("Orientation [rad]")
        axes[1].set_xlabel("Time [s]")
        axes[1].grid(True, alpha=0.3)

        for ax in axes:
            ax.legend(loc="upper right", fontsize=8)

        out = output_dir / "ee_pose.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Saved to {out}")

    print("All plots created!")


def visualize_data(data_path, output_dir=None, fix_ee_rotation_wrap=True):
    """Main function to visualize collected data"""
    data_path = Path(data_path)

    # Set output directory
    if output_dir is None:
        output_dir = data_path.parent / f"{data_path.stem}_visualizations"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 Visualizing data from: {data_path}")
    # print(f"📁 Output directory: {output_dir}")
    # print("=" * 60)

    # Load pkl data (contains actual image data)
    pkl_path = data_path.with_suffix(".pkl")
    if not pkl_path.exists():
        print(f"❌ PKL file not found: {pkl_path}")
        return

    pkl_data = load_data(pkl_path)

    plot_joint_data(pkl_data, output_dir, fix_ee_rotation_wrap=fix_ee_rotation_wrap)
    create_image_gif(
        pkl_data,
        output_dir / "camera_rgb_front.mp4",
        "/realsense/front/color/image_raw",
        fps=30,
        video_downsample_factor=3,
    )


if __name__ == "__main__":
    import sys

    # Automatically pick the newest date folder
    # Choose the newest folder inside the base directory

    ########################### Chose your base directory here ###########################
    base_dir = Path("~/activeinference/factr/process_data/raw_data/fourgoals_2").expanduser()

    # Sort by folder name (assuming YYYYMMDD format) to get the newest by date, not by mtime
    data_dirs = [d for d in base_dir.glob("*/data") if d.is_dir()]
    if not data_dirs:
        print(f"❌ No data directories found in {base_dir}")
        sys.exit(1)
    # Use parent folder name as date string for sorting
    base_data_dir = max(data_dirs, key=lambda d: d.parent.name)
    print(f"Using newest data directory: {base_data_dir}")
    base_output_dir = base_data_dir.parent / "visualizations"

    base_output_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(base_data_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"❌ No .pkl files found in {base_data_dir}")
        sys.exit(1)

    # Check if we should process all files or only the newest one
    process_all = False  ################## Set to True to process all files
    overwrite_existing = True  ################## Set to True to overwrite existing visualizations
    selected_episode = "ep_42_stiff"  ################## e.g., "ep_01" or None to use the latest
    fix_ee_rotation_wrap = True  ################## Set to False to disable +/-2pi correction in ee pose plot

    if process_all:
        for idx, pkl_path in enumerate(pkl_files, start=0):
            episode_name = pkl_path.stem
            output_dir = base_output_dir / episode_name

            if not overwrite_existing and output_dir.exists() and any(output_dir.iterdir()):
                print("=" * 80)
                print(f"⚡ Skipping file {idx}/{len(pkl_files)}: {pkl_path.name} (already visualized)")
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            print("=" * 80)

            try:
                visualize_data(pkl_path, output_dir, fix_ee_rotation_wrap=fix_ee_rotation_wrap)
            except Exception as e:
                print(f"❌ Error processing {pkl_path.name}: {e}")
                continue
    else:
        # Process a selected episode or the newest file
        if selected_episode is not None:
            selected_path = base_data_dir / f"{selected_episode}.pkl"
            if not selected_path.exists():
                print(f"❌ Selected episode not found: {selected_path}")
                sys.exit(1)
            target_pkl = selected_path
        else:
            target_pkl = pkl_files[-1]

        episode_name = target_pkl.stem
        output_dir = base_output_dir / episode_name

        if not overwrite_existing and output_dir.exists() and any(output_dir.iterdir()):
            print("=" * 80)
            print(f"⚡ Skipping file: {target_pkl.name} (already visualized)")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            print("=" * 80)

            try:
                visualize_data(target_pkl, output_dir, fix_ee_rotation_wrap=fix_ee_rotation_wrap)
            except Exception as e:
                print(f"❌ Error processing {target_pkl.name}: {e}")

    print("=" * 80)
    print(f"✅ All visualizations completed! Check {base_output_dir}")
