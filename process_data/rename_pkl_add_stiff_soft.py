import os
import re

# Folder with your files
FOLDER = (
    "/home/ferdinand/activeinference/factr/process_data/data_to_process/fourgoals_3_stiff/data"  # change if needed
)

pattern = re.compile(r"ep_(\d+)\.pkl")

for filename in os.listdir(FOLDER):
    match = pattern.match(filename)
    if match:
        episode_num = match.group(1)
        new_name = f"ep_{episode_num}_stiff.pkl"
        old_path = os.path.join(FOLDER, filename)
        new_path = os.path.join(FOLDER, new_name)
        os.rename(old_path, new_path)
