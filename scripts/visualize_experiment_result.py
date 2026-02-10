#!/usr/bin/env python3

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import cv2
from PIL import Image
import os
import imageio.v2 as imageio
from matplotlib.ticker import MultipleLocator


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


def create_image_gif(pkl_data, output_path, topic_name="/realsense/front/im", fps=30, video_downsample_factor=1):
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

            # if i > 300 and i < 350 and topic_name == '/realsense/front/im':
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


def plot_joint_data(pkl_data, output_dir):
    """Plot Franka joint positions, external torques, and impedance commands"""
    print("Creating plots...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data for each topic
    topics = {
        "franka_state": "/franka/right/obs_franka_state",
        "external_torques": "/franka_robot_state_broadcaster/external_joint_torques",
        "impedance_cmd": "/joint_impedance_dynamic_gain_controller/joint_impedance_command",
        # 'impedance_cmd': '/joint_impedance_command_controller/joint_trajectory',
        "predictions": "/inference/ensembled_predictions",
        "raw_predictions": "/inference/raw_predictions",
        "attention": "/inference/attention",
        "measured_joints": "/franka_robot_state_broadcaster/measured_joint_states",
        "external_wrench": "/franka_robot_state_broadcaster/external_wrench_in_base_frame",
        "robot_state": "/franka_robot_state_broadcaster/robot_state",
    }

    data_dict = {}
    global_t0 = None
    for key, topic in topics.items():
        data, timestamps = extract_topic_data(pkl_data, topic)
        if data is not None and len(data) > 0:
            data_dict[key] = {
                "data": data,
                "timestamps": np.array(timestamps) / 1e9,  # Convert to seconds
            }
            if global_t0 is None:
                global_t0 = data_dict[key]["timestamps"][0]
            data_dict[key]["timestamps"] -= global_t0

    # # 1. Plot Franka Joint Positions
    # if 'franka_state' in data_dict:
    #     print("  Plotting Franka joint positions...")
    #     fig, axes = plt.subplots(7, 1, figsize=(12, 14))
    #     fig.suptitle('Franka Joint Positions', fontsize=16)

    #     positions = np.array([d['position'] for d in data_dict['franka_state']['data']])
    #     timestamps = data_dict['franka_state']['timestamps']

    #     for i in range(7):
    #         axes[i].plot(timestamps, positions[:, i], linewidth=2)
    #         axes[i].set_ylabel(f'Joint {i+1} [rad]', fontsize=10)
    #         axes[i].grid(True, alpha=0.3)
    #         if i == 6:
    #             axes[i].set_xlabel('Time [s]', fontsize=10)

    #     plt.tight_layout()
    #     output_path = output_dir / 'franka_joint_positions.png'
    #     plt.savefig(output_path, dpi=150, bbox_inches='tight')
    #     print(f"  ✅ Saved to {output_path}")
    #     plt.close()

    # 2. Plot External Joint Torques
    if "external_torques" in data_dict:
        fig, axes = plt.subplots(7, 1, figsize=(12, 14))
        fig.suptitle("External Joint Torques", fontsize=16)

        torques = np.array([d["effort"] for d in data_dict["external_torques"]["data"]])
        timestamps = data_dict["external_torques"]["timestamps"]

        steps = np.arange(len(torques))
        t_min = np.min(torques)
        t_max = np.max(torques)
        margin = (t_max - t_min) * 0.05  # 上下に5%のマージンを追加
        if margin == 0:
            margin = 1.0  # 値が全く変化しない場合の保護
        y_lims = (t_min - margin, t_max + margin)

        # for i in range(7):
        #     axes[i].plot(timestamps, torques[:, i], linewidth=1, color='blue')
        #     axes[i].set_ylabel(f'Joint {i+1} [Nm]', fontsize=10)
        #     axes[i].set_ylim(y_lims)
        #     axes[i].grid(True, alpha=0.3)
        #     if i == 6:
        #         axes[i].set_xlabel('Time [s]', fontsize=10)
        for i in range(7):
            # --- 変更点: 横軸を steps に変更 ---
            axes[i].plot(steps, torques[:, i], linewidth=1, color="blue")
            # --------------------------------

            axes[i].set_ylabel(f"Joint {i + 1} [Nm]", fontsize=10)
            axes[i].set_ylim(y_lims)
            axes[i].grid(True, alpha=0.3)
            if i == 6:
                # --- 変更点: ラベル変更 ---
                axes[i].set_xlabel("Timestep", fontsize=10)

        plt.tight_layout()
        output_path = output_dir / "external_joint_torques.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ Saved to {output_path}")
        plt.close()

    # 2b. Plot external wrench (force/torque in base frame)
    if "external_wrench" in data_dict:
        print("  Plotting external wrench in base frame...")
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle("External Wrench in Base Frame", fontsize=16)

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
        out = output_dir / "external_wrench.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  ✅ Saved to {out}")
        plt.close()

    # 4. Combined plot: Measured vs Commanded positions
    if "measured_joints" in data_dict and "impedance_cmd" in data_dict:
        fig, axes = plt.subplots(7, 1, figsize=(12, 14))
        fig.suptitle("Joint Positions: Measured vs Commanded", fontsize=16)

        measured_pos = np.array([d["position"] for d in data_dict["measured_joints"]["data"]])
        measured_ts = data_dict["measured_joints"]["timestamps"]

        commanded_pos = np.array([d["position"] for d in data_dict["impedance_cmd"]["data"]])
        commanded_ts = data_dict["impedance_cmd"]["timestamps"]

        for i in range(7):
            axes[i].plot(measured_ts, measured_pos[:, i], linewidth=2, label="Measured", alpha=0.7)
            axes[i].plot(commanded_ts, commanded_pos[:, i], linewidth=2, label="Commanded", alpha=0.7, linestyle="--")
            axes[i].set_ylabel(f"Joint {i + 1} [rad]", fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc="upper right", fontsize=8)
            if i == 6:
                axes[i].set_xlabel("Time [s]", fontsize=10)

        plt.tight_layout()
        output_path = output_dir / "measured_vs_commanded.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ Saved to {output_path}")
        plt.close()

    # 8. Plot Predictions (expanded to 25Hz, batches NOT connected)
    if "predictions" in data_dict:
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

        extra_rows = 0
        if entropy_available:
            extra_rows += 1
        if gains_available:
            extra_rows += 1

        total_rows = 7 + extra_rows
        fig, axes = plt.subplots(total_rows, 1, figsize=(16, 2.4 * total_rows), sharex=True)
        fig.suptitle("Measured vs Commanded vs Predictions", fontsize=16)

        # Ensure axes is indexable
        if total_rows == 1:
            axes = np.array([axes])

        for j in range(7):
            ax = axes[j]
            ax.plot(meas_ts, measured_pos[:, j], label="Measured", linewidth=1.5, alpha=0.8, color="black")

            # Only plot predictions for matching dimension
            if pred_pos.size > 0 and j < pred_pos.shape[1]:
                ax.plot(pred_ts, pred_pos[:, j], label="Predictions Ensembled", linewidth=1.4, alpha=0.7, color="blue")
                ax.plot(
                    raw_predictions_ts,
                    raw_predictions[:, j],
                    label="Predictions raw",
                    linewidth=1.4,
                    alpha=0.35,
                    color="grey",
                )

            # Add circle markers only for batches that were actually expanded
            for k, idx in enumerate(valid_pred_indices):
                batch = pred_batches[idx]
                ts = pred_ts_raw[idx]
                if "positions" in batch and len(batch["positions"]) > 0:
                    ax.plot(
                        ts,
                        batch["positions"][0][j],
                        "o",
                        color="blue",
                        markersize=2,
                        label="Batch Start (ensembled)" if j == 0 and k == 0 else "",
                    )

            for k, idx in enumerate(valid_raw_indices):
                batch = pred_batches_raw[idx]
                ts = pred_ts_raw_raw[idx]
                if "positions" in batch and len(batch["positions"]) > 0:
                    ax.plot(
                        ts,
                        batch["positions"][0][j],
                        "o",
                        color="grey",
                        markersize=2,
                        label="Batch Start (raw)" if j == 0 and k == 0 else "",
                    )

            ax.plot(cmd_ts, commanded_pos[:, j], label="Commanded", linewidth=1.3, alpha=1, color="red")

            ax.set_ylabel(f"Joint {j + 1}")
            ax.grid(True, alpha=0.3)
            if j == 6:
                ax.set_xlabel("Time [s]")
            ax.legend(fontsize=8, loc="upper right")

        # Plot entropy and gains in the extra rows, if available
        row_idx = 7
        if entropy_available:
            ax_ent = axes[row_idx]
            ent_ensembled = pred_pos[:, 7]
            ent_raw = raw_predictions[:, 7]
            ax_ent.plot(pred_ts, ent_ensembled, label="Variance (ensembled)", color="tab:blue", linewidth=1.2)
            ax_ent.plot(raw_predictions_ts, ent_raw, label="Variance (raw)", color="gray", linewidth=1.0, alpha=0.6)
            ax_ent.set_ylabel("Variance")  # 軸ラベルも変更
            ax_ent.grid(True, alpha=0.3)
            ax_ent.legend(fontsize=8)
            row_idx += 1

        if gains_available:
            ax_g = axes[row_idx]
            ax_g.plot(gains_ts, pct_mean, label="Mean Gains Percent (k+d)", color="tab:red", linewidth=1.4)
            ax_g.set_ylabel("Gains % (0=soft,100=stiff)")
            ax_g.set_xlabel("Time [s]")
            ax_g.set_ylim(-5, 105)
            ax_g.grid(True, alpha=0.3)
            ax_g.legend(fontsize=8)

        out = output_dir / "measured_vs_commanded_vs_predictions.png"
        plt.tight_layout()
        plt.savefig(out, dpi=300)
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
            else:
                k_vals = np.vstack(k_vals)
                d_vals = np.vstack(d_vals)

                # Reference soft/stiff gains (from inference_parameters.yaml)
                k_soft = np.array([69.89, 86.61, 232.11, 91.75, 32.38, 17.76, 10.49])
                d_soft = np.array([7.66, 7.36, 12.86, 6.61, 2.18, 1.11, 0.99])

                k_stiff = np.array([305.62, 303.85, 449.42, 309.65, 200.14, 110.16, 105.80])
                d_stiff = np.array([37.15, 36.63, 46.69, 33.14, 17.06, 9.16, 9.41])

                # Compute percent [0..100] per joint, clamp
                pct_k = (k_vals - k_soft) / (k_stiff - k_soft) * 100.0
                pct_d = (d_vals - d_soft) / (d_stiff - d_soft) * 100.0
                pct_k = np.clip(pct_k, 0.0, 100.0)
                pct_d = np.clip(pct_d, 0.0, 100.0)

                # Aggregate across joints (mean percent). You can change to plot per-joint if desired
                pct_k_mean = np.nanmean(pct_k, axis=1)
                pct_d_mean = np.nanmean(pct_d, axis=1)
                pct_mean = 0.5 * (pct_k_mean + pct_d_mean)

                # Build timestamps array for gains (imp_ts already in seconds and offset earlier)
                gains_ts = imp_ts

                # 3段構成 (Variance, Gains, Images)
                # ★ 8枚並べるために figsize の横幅を少し広げます
                fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
                fig.suptitle("Variance, Stiffness, and Key Frames (Fixed Timesteps)", fontsize=16)

                Hz = 30

                # --- 1段目: Variance plot ---
                axes[0].plot(
                    entropy_ts_ensembled * Hz,
                    entropy_ensembled,
                    label="Variance (ensembled)",
                    color="tab:blue",
                    linewidth=1.5,
                )
                axes[0].plot(
                    entropy_ts_raw * Hz, entropy_raw, label="Variance (raw)", color="gray", linewidth=1.0, alpha=0.6
                )
                axes[0].set_ylabel("Variance value")
                axes[0].set_ylim(-0.05, 1.05)
                axes[0].grid(True, alpha=0.3)
                axes[0].legend(fontsize=8, loc="upper right")

                # --- 2段目: Gains percent plot ---
                axes[1].plot(
                    gains_ts * Hz, pct_mean, label="Mean Stiffness Percent (k+d)", color="tab:red", linewidth=1.5
                )
                axes[1].set_ylabel("Stiffness [%]")
                axes[1].set_ylim(-5, 105)
                axes[1].grid(True, alpha=0.3)
                axes[1].legend(fontsize=8, loc="upper right")

                axes[1].tick_params(labelbottom=True)
                axes[1].set_xlabel("Timestep", fontsize=12)

                # --- 3段目: 固定ステップの画像表示 ---
                img_topic = "/realsense/front/im"
                image_msgs, img_timestamps = extract_topic_data(pkl_data, img_topic)
                axes[2].axis("off")

                if image_msgs:
                    # ★ 指定された固定ステップ (50から400まで50刻み)
                    # 8枚の画像を大きく並べるためのレイアウト設定
                    # 8枚の画像を大きく並べるためのレイアウト設定
                    num_imgs = 8
                    target_steps = [50, 100, 150, 200, 250, 300, 350, 400]
                    # target_steps = [50, 150, 250, 350, 450, 550, 650, 750]
                    # target_steps = [100, 200, 300, 400, 500, 600, 700, 750]

                    # Figureの作成 (横幅を広げ、画像が大きく見えるよう縦横比を調整)
                    fig = plt.figure(figsize=(22, 11))

                    # GridSpecでレイアウトを定義
                    # height_ratiosで3段目（画像）の比重を高くし、wspaceで画像同士の隙間を消す
                    gs = fig.add_gridspec(3, num_imgs, height_ratios=[1.2, 1.2, 1.8], hspace=0.45, wspace=0.02)

                    # --- 1段目: Variance (全列を結合) ---
                    ax_var = fig.add_subplot(gs[0, :])
                    ax_var.plot(
                        entropy_ts_ensembled * Hz,
                        entropy_ensembled,
                        label="Variance (ensembled)",
                        color="tab:blue",
                        linewidth=1.8,
                    )
                    ax_var.plot(
                        entropy_ts_raw * Hz, entropy_raw, label="Variance (raw)", color="gray", alpha=0.4, linewidth=1.0
                    )
                    ax_var.set_ylabel("Variance", fontsize=13)  # Varianceから変更
                    ax_var.set_ylim(-0.05, 1.05)
                    ax_var.set_xlim(0, 450)
                    ax_var.xaxis.set_major_locator(MultipleLocator(50))
                    ax_var.grid(True, which="major", alpha=0.3)
                    ax_var.legend(loc="upper right", fontsize=10)
                    ax_var.grid(True, alpha=0.3)

                    # --- 2段目: Gains (全列を結合) ---
                    ax_gain = fig.add_subplot(gs[1, :])
                    ax_gain.plot(gains_ts * Hz, pct_mean, label="Mean Stiffness", color="tab:red", linewidth=1.8)
                    ax_gain.set_ylabel("Stiffness [%]", fontsize=13)
                    ax_gain.set_ylim(-5, 105)
                    ax_gain.set_xlim(0, 450)
                    ax_gain.set_xlabel("Timestep", fontsize=13)
                    ax_gain.xaxis.set_major_locator(MultipleLocator(50))
                    ax_gain.grid(True, which="major", alpha=0.3)

                    # メモリの文字サイズを少し小さくしたい場合は以下を追加
                    ax_gain.tick_params(axis="x", labelsize=10)
                    ax_gain.grid(True, alpha=0.3)

                    # --- 3段目: 画像 (1枚ずつ配置) ---
                    img_topic = "/realsense/front/im"
                    image_msgs, img_timestamps = extract_topic_data(pkl_data, img_topic)

                    if image_msgs:
                        img_ts_seconds = (np.array(img_timestamps) / 1e9) - global_t0

                        for i, step in enumerate(target_steps):
                            t_target = step / Hz
                            img_ax = fig.add_subplot(gs[2, i])

                            if t_target <= img_ts_seconds.max():
                                idx = np.argmin(np.abs(img_ts_seconds - t_target))
                                data = image_msgs[idx]
                                img_array = np.frombuffer(data["data"], dtype=np.uint8).reshape(
                                    data["height"], data["width"], 3
                                )

                                img_ax.imshow(img_array)
                                # タイトル文字。weight='normal'で太字を避ける
                                img_ax.set_title(f"step={step}", fontsize=12, weight="normal", pad=8)

                            img_ax.axis("off")

                    # 保存設定
                    out = output_dir / "variance_and_gains_fixed_steps.png"
                    plt.savefig(out, dpi=250, bbox_inches="tight")
                    plt.close()
                    print(f"  ✅ Saved to {out}")

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

        ax.set_xlabel("Time [s]")
        ax.legend(fontsize=8)

        out = output_dir / "attention.png"
        plt.tight_layout()
        plt.savefig(out, dpi=250)
        plt.close()
        print(f"  ✅ Saved to {out}")

    # 10. Plot end-effector position and orientation from robot_state (ee_pose)
    if "robot_state" in data_dict:
        print("  Plotting end-effector pose from robot_state...")

        ee_pose = np.array([d["ee_pose"] for d in data_dict["robot_state"]["data"] if "ee_pose" in d])
        ee_pose_ts = data_dict["robot_state"]["timestamps"]

        # Extract quaternion (last 4 positions of ee_pose)
        quaternions = ee_pose[:, 3:]

        # Convert quaternion to Euler angles (roll, pitch, yaw)
        euler_angles = []
        for q in quaternions:
            x, y, z, w = q
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            # Pitch (y-axis rotation)
            sinp = 2 * (w * y - z * x)
            pitch = np.arcsin(sinp) if abs(sinp) <= 1 else np.sign(sinp) * np.pi / 2

            # Yaw (z-axis rotation)
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            euler_angles.append([roll, pitch, yaw])

        euler_angles = np.array(euler_angles)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle("End-Effector Pose from FrankaRobotState", fontsize=16)

        # Position (x, y, z) combined in one plot
        axes[0].plot(ee_pose_ts, ee_pose[:, 0], label="x", color="tab:blue")
        axes[0].plot(ee_pose_ts, ee_pose[:, 1], label="y", color="tab:orange")
        axes[0].plot(ee_pose_ts, ee_pose[:, 2], label="z", color="tab:green")
        axes[0].set_ylabel("Position [m]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8)

        # Orientation (roll, pitch, yaw) combined in one plot
        axes[1].plot(ee_pose_ts, euler_angles[:, 0], label="roll", color="tab:red")
        axes[1].plot(ee_pose_ts, euler_angles[:, 1], label="pitch", color="tab:purple")
        axes[1].plot(ee_pose_ts, euler_angles[:, 2], label="yaw", color="tab:brown")
        axes[1].set_ylabel("Orientation [rad]")
        axes[1].set_xlabel("Time [s]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right", fontsize=8)

        # fig, axes = plt.subplots(6, 1, figsize=(12, 16))
        # fig.suptitle('End-Effector Pose from FrankaRobotState', fontsize=16)

        # # Position
        # axes[0].plot(ee_pose_ts, ee_pose[:, 0], label='x', color='tab:blue')
        # axes[0].set_ylabel('x [m]')
        # axes[0].grid(True, alpha=0.3)

        # axes[1].plot(ee_pose_ts, ee_pose[:, 1], label='y', color='tab:orange')
        # axes[1].set_ylabel('y [m]')
        # axes[1].grid(True, alpha=0.3)

        # axes[2].plot(ee_pose_ts, ee_pose[:, 2], label='z', color='tab:green')
        # axes[2].set_ylabel('z [m]')
        # axes[2].grid(True, alpha=0.3)

        # axes[3].plot(ee_pose_ts, euler_angles[:, 0], label='roll', color='tab:red')
        # axes[3].set_ylabel('roll')
        # axes[3].grid(True, alpha=0.3)

        # axes[4].plot(ee_pose_ts, euler_angles[:, 1], label='pitch', color='tab:purple')
        # axes[4].set_ylabel('pitch')
        # axes[4].grid(True, alpha=0.3)

        # axes[5].plot(ee_pose_ts, euler_angles[:, 2], label='yaw', color='tab:brown')
        # axes[5].set_ylabel('yaw')
        # axes[5].set_xlabel('Time [s]')
        # axes[5].grid(True, alpha=0.3)

        for ax in axes:
            ax.legend(loc="upper right", fontsize=8)

        out = output_dir / "end_effector_pose.png"
        plt.tight_layout()
        plt.savefig(out, dpi=250, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Saved to {out}")

    print("All plots created!")


def visualize_data(data_path, output_dir=None):
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

    plot_joint_data(pkl_data, output_dir)
    create_image_gif(pkl_data, output_dir / "camera_rgb.mp4", "/realsense/front/im", fps=30, video_downsample_factor=2)


if __name__ == "__main__":
    import sys

    # 1. プロジェクトルートを取得 (他のコードと統一)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # 2. データの読み込み元を直接指定
    base_data_dir = Path("/data/otake/inference_test_up_side_final_finaltest_2/20260126/data")

    # 3. 保存先を Path オブジェクトで指定
    base_output_dir = Path("/data/otake/20260126/exp_result_output_output2_torque")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # 4. ディレクトリ内の .pkl ファイルを取得
    pkl_files = sorted(base_data_dir.glob("*.pkl"))

    if not pkl_files:
        print(f"❌ No .pkl files found in {base_data_dir}")
        sys.exit(1)

    print(f"Found {len(pkl_files)} files in {base_data_dir}")

    # --- 実行設定 ---
    process_all = True  # Falseなら最新の1件だけ、Trueなら全件
    overwrite_existing = True  # 上書きするかどうか

    if process_all:
        files_to_process = pkl_files
    else:
        # 最新の1件のみをリストにする
        files_to_process = [pkl_files[-1]]

    for idx, pkl_path in enumerate(files_to_process, start=1):
        episode_name = pkl_path.stem
        output_dir = base_output_dir / episode_name

        if not overwrite_existing and output_dir.exists() and any(output_dir.iterdir()):
            print("=" * 80)
            print(f"⚡ Skipping: {pkl_path.name} (already exists)")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 80)
        print(f"Processing ({idx}/{len(files_to_process)}): {pkl_path.name}")

        try:
            visualize_data(pkl_path, output_dir)
        except Exception as e:
            print(f"❌ Error processing {pkl_path.name}: {e}")
            continue

    print("=" * 80)
    print(f"✅ All visualizations completed! Check: {base_output_dir}")
