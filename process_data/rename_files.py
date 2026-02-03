import os
import re
from pathlib import Path

folder = Path("/home/ferdinand/factr_project/factr/process_data/data_to_process/20251107/data")
prefix = "data_log_"  # current prefix of the files
exts = [".pkl"]
pad = 1  # how many digits: ep_01, ep_02, ...

# --- 1️⃣ find all episode numbers ---
def extract_num(name):
    m = re.search(rf"{prefix}(\d+)\.pkl", name)
    return int(m.group(1)) if m else None

pkl_files = sorted([f for f in folder.glob(f"{prefix}*.pkl")], key=lambda f: extract_num(f.name))
nums = [extract_num(f.name) for f in pkl_files if extract_num(f.name) is not None]
print(f"Found {len(nums)} PKL files:", nums[:10], "...")

# --- 2️⃣ make a temporary rename to avoid collisions ---
for f in pkl_files:
    tmp = f.with_name(f"tmp_{f.name}")
    os.rename(f, tmp)

json_files = list(folder.glob(f"{prefix}*.json"))
for f in json_files:
    tmp = f.with_name(f"tmp_{f.name}")
    os.rename(f, tmp)

# --- 3️⃣ rename sequentially and match pairs ---
for i, old_num in enumerate(nums, start=1):
    new_name = f"ep_{i:0{pad}}"  # Change prefix to "ep"
    old_pkl = folder / f"tmp_{prefix}{old_num}.pkl"
    old_json = folder / f"tmp_{prefix}{old_num}.json"

    if old_pkl.exists():
        new_pkl = folder / f"{new_name}.pkl"
        os.rename(old_pkl, new_pkl)
        print(f"✅ {old_pkl.name} → {new_pkl.name}")
    if old_json.exists():
        new_json = folder / f"{new_name}.json"
        os.rename(old_json, new_json)
        print(f"✅ {old_json.name} → {new_json.name}")

# --- 4️⃣ cleanup stray tmp files ---
for f in folder.glob("tmp_*"):
    f.rename(f.name.replace("tmp_", ""))

print("\n✅ All files renumbered sequentially and safely.")

