import os
import re

# Folder with your files
FOLDER = "/home/ferdinand/factr_project/factr/process_data/raw_data_eval/bld_both"   # change if needed

pattern = re.compile(r"ep_(\d+)\.pkl")

for filename in os.listdir(FOLDER):
    match = pattern.match(filename)
    if match:
        new_name = f"ep_{match.group(1)}_soft.pkl"
        old_path = os.path.join(FOLDER, filename)
        new_path = os.path.join(FOLDER, new_name)
        os.rename(old_path, new_path)