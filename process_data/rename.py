import os
import re

# Folder with your files
FOLDER = (
    "/home/ferdinand/activeinference/factr/process_data/raw_data/fourgoals_1_stiff/20260203/data"  # change if needed
)

pattern = re.compile(r"ep_(\d+)\.pkl")

for filename in os.listdir(FOLDER):
    match = pattern.match(filename)
    if match:
        new_name = f"ep_{match.group(1)}_stiff.pkl"
        old_path = os.path.join(FOLDER, filename)
        new_path = os.path.join(FOLDER, new_name)
        os.rename(old_path, new_path)
