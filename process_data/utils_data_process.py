# ---------------------------------------------------------------------------
# FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning
# https://arxiv.org/abs/2502.17432
# Copyright (c) 2025 Jason Jingzhou Liu and Yulong Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------


import cv2
import numpy as np
from robobuf.buffers import ObsWrapper, Transition, ReplayBuffer
from scipy.signal import butter, filtfilt, medfilt

def gaussian_norm(list_of_array):
    data_array = np.concatenate(list_of_array, axis=0)
    
    print('Using in-place gaussian norm')
    mean = np.mean(data_array, axis=0)
    std = np.std(data_array, axis=0)
    if not std.all():  # handle situation with all 0 actions
        std[std == 0] = 1e-17

    for array in list_of_array:
        array -= mean
        array /= std
    normalization_stats = dict(
        mean=[float(x) for x in mean],
        std=[float(x) for x in std]
    )
    return normalization_stats

def generate_robobuf(trajectories):
    buffer = ReplayBuffer()
    for traj in trajectories:
        num_steps = traj['num_steps']
        actions = traj['actions']
        states = traj['states']
        goals = traj.get('goals')
        for i in range(num_steps):
            obs = {
                'state': states[i],
            }
            if goals is not None:
                obs['goals'] = goals[i]
            for k, v in traj.items():
                if k.startswith('enc_cam_'):
                    obs[k] = v[i]
            transition = Transition(
                obs = ObsWrapper(obs), 
                action = actions[i], 
                reward = (i==num_steps-1), 
            )
            buffer.add(transition, is_first = (i==0))

    return buffer

def get_diff_timestamps(timestamps):
    timestamps = np.array(timestamps)
    diff_timestamps = np.diff(timestamps)
    return diff_timestamps * 1e-9

def sync_data_slowest(traj_data, all_topics):
    timestamps = traj_data["timestamps"]
    for topic in all_topics:
        assert topic in timestamps, f"Topic {topic} not found in recorded data"
        timestamps[topic] = np.array(timestamps[topic])
    data = traj_data["data"]
    synced_data = {topic: [] for topic in all_topics}
    message_counts = [len(timestamps[topic]) for topic in all_topics]
    min_freq_topic = all_topics[int(np.argmin(message_counts))]
    
    timestamp_diffs = get_diff_timestamps(timestamps[min_freq_topic])
    avg_freq = 1/np.mean(timestamp_diffs)
    
    for i, target_ts in enumerate(timestamps[min_freq_topic]):
        for topic in all_topics:
            if topic == min_freq_topic:
                synced_data[topic].append(data[topic][i])
            else:
                closest_idx = np.argmin(np.abs(timestamps[topic] - target_ts))
                synced_data[topic].append(data[topic][closest_idx])
                
    return synced_data, avg_freq
    
def process_decoded_image(img):
    img = cv2.resize(img, (256, 256))
    return img
        
def process_image(img_enc):

    ###################
    """Handle both raw RGB list images and encoded JPEG images."""
    # Case 1 – Your raw RGB dictionary (list of ints)
    if isinstance(img_enc, dict) and "data" in img_enc:
        arr = np.array(img_enc["data"], dtype=np.uint8)
        h, w = img_enc.get("height", 480), img_enc.get("width", 640)
        if len(arr) != h * w * 3:
            raise ValueError(f"Unexpected image length {len(arr)} for {h}x{w}")
        decoded_image = arr.reshape((h, w, 3))
    else:
        # Case 2 – FACTR's normal encoded bytes path
        if not isinstance(img_enc, np.ndarray):
            img_enc = np.frombuffer(img_enc, np.uint8)
        decoded_image = cv2.imdecode(img_enc, cv2.IMREAD_COLOR)


    # decode, process, and encode
    # decoded_image = cv2.imdecode(img_enc, cv2.IMREAD_COLOR)    
    decoded_image = process_decoded_image(decoded_image)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    _, compressed_image = cv2.imencode('.jpg', decoded_image, encode_param)
    return compressed_image


def lowpass_filter(traj_data, cutoff_freq, fs, topic_name=None, key_options=("effort", "position", "data"), vector_size=7):
    """
    Apply a low-pass Butterworth filter to a numeric topic inside traj_data.

    Args:
        traj_data (dict): the trajectory dict with topic data
        topic_name (str): topic key, e.g. '/franka_robot_state_broadcaster/external_joint_torques'
        cutoff_freq (float): cutoff frequency (Hz)
        fs (float): sampling frequency (Hz)
        key_options (tuple): possible field names to extract data from (default: ('effort','position','data'))
        vector_size (int): fallback vector length if missing (default: 7)
    Returns:
        np.ndarray: filtered data (num_steps, vector_size)
    """
    if topic_name is not None:
        if topic_name not in traj_data:
            print(f"⚠️ Topic {topic_name} not found — skipping filter.")
            return None

        topic_data = traj_data[topic_name]

        # Collect numeric arrays
        rows = []
        for msg in topic_data:
            if isinstance(msg, dict):
                found = False
                for k in key_options:
                    if k in msg:
                        rows.append(np.array(msg[k], dtype=float))
                        found = True
                        break
                if not found:
                    rows.append(np.zeros(vector_size, dtype=float))
            else:
                rows.append(np.array(msg, dtype=float).flatten())

        # Stack and filter
        arr = np.stack(rows, axis=0)
    else:
        arr = np.array(traj_data, dtype=float)

    # Filter
    b, a = butter(N=2, Wn=cutoff_freq / (fs / 2), btype="low")
    filtered = filtfilt(b, a, arr, axis=0)

    traj_data[topic_name] = filtered.tolist()
    print(f"  🔉 Applied low-pass filter to {topic_name} with cutoff {cutoff_freq} Hz")
    return filtered



def medianfilter(data_array, kernel_size=3, key=None):
    """
    Apply a median filter to joint data.
    Handles arrays, lists of lists, or lists of dicts with numeric values.

    Args:
        data_array: list, np.ndarray, or list of dicts
        kernel_size (int): median filter window size (odd number)
        key (str or None): if elements are dicts, extract this key (e.g. 'effort', 'position')

    Returns:
        np.ndarray: filtered array of same shape
    """
    # Convert to list for inspection
    if len(data_array) == 0:
        return np.array(data_array)

    first_elem = data_array[0]

    # --- Case 1: dicts (e.g. [{'effort': [...]}, ...]) ---
    if isinstance(first_elem, dict):
        if key is None:
            # try to auto-detect the first key that holds numeric data
            key = next((k for k, v in first_elem.items() if isinstance(v, (list, np.ndarray)) and np.any(v)), None)
            if key is None:
                raise ValueError("medianfilter: cannot auto-detect numeric key in dicts or all values are 0")
        # extract list of numeric vectors
        data_array = np.array([np.array(d[key], dtype=float) for d in data_array])

    # --- Case 2: list of lists or ndarray ---
    else:
        data_array = np.asarray(data_array, dtype=float)

    # --- Apply median filter ---
    if data_array.ndim == 1:
        filtered = medfilt(data_array, kernel_size=kernel_size)
    elif data_array.ndim == 2:
        filtered = np.stack(
            [medfilt(data_array[:, j], kernel_size=kernel_size) for j in range(data_array.shape[1])],
            axis=1
        )
    else:
        raise ValueError(f"Unsupported shape {data_array.shape} for medianfilter")

    print(f"  🔉 Applied median filter to {key} with kernel size {kernel_size}")
    return filtered


def downsample_data(data, avg_freq, target_downsampling_freq):
    """
    Downsample data (dict, list, or numpy array) to target frequency.

    Args:
        data (dict | list | np.ndarray): input data to downsample
        avg_freq (float): original average frequency (Hz)
        target_downsampling_freq (float): desired target frequency (Hz)

    Returns:
        tuple: (downsampled_data, new_avg_freq)
    """
    # compute step
    step = max(1, round(avg_freq / target_downsampling_freq))
    new_freq = avg_freq / step

    # --- case 1: dict of lists or arrays ---
    if isinstance(data, dict):
        downsampled = {}
        for key, val in data.items():
            if isinstance(val, (list, np.ndarray)):
                downsampled[key] = val[::step]
            else:
                downsampled[key] = val
        print(f"  🔻 Downsampled dict from ~{avg_freq:.1f} Hz to ~{new_freq:.1f} Hz (step={step})")
        return downsampled, new_freq

    # --- case 2: numpy array or list ---
    elif isinstance(data, (list, np.ndarray)):
        data = np.asarray(data)
        downsampled = data[::step]
        print(f"🔻 Downsampled array from ~{avg_freq:.1f} Hz to ~{new_freq:.1f} Hz (step={step})")
        return downsampled, new_freq

    # --- unsupported type ---
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

def ema_filter(x, alpha=0.1):
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y