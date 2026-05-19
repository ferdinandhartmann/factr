## Copy folders from other computer to this computer

### Copy one folder to the currect folder

```bash
rsync -avzP --inplace --nocompression otake@192.168.1.172:/home/otake/factr_ws/raw_data/fourgoals_1_stiff .
```
### Copy multiple folders at once to the currect folder

```bash
rsync -avP otake@192.168.1.172:/home/otake/factr_ws/raw_data/{fourgoals_1_stiff,fourgoals_1_medium,fourgoals_1_soft} .
```

rsync -avzP ferdinand@192.168.1.44:/home/ferdinand/activeinference/franka-gpu-server/droid_100 .

### Check Stiffness Labels of train.buf (Available ones and count)

```bash
python - <<'PY'
import pickle
from collections import Counter

file = "/home/ferdinand/activeinference/factr/process_data/processed_data/fourgoals_12_allgauss_noclip_cmdinput/buf_train.pkl"

with open(file, "rb") as f:
    data = pickle.load(f)

labels = []

for ep in data:
    for step in ep:
        obs = step[0]
        labels.append(obs["stiffness_label"])

counter = Counter(labels)

print("Total labels:", len(labels))
print("Unique labels:", sorted(counter.keys()))
print()
print("Counts:")
for k, v in sorted(counter.items()):
    print(f"{k}: {v}")
PY
```

Per Edpisode:

```bash
python - <<'PY'
import pickle
from collections import Counter

file = "/home/ferdinand/activeinference/factr/process_data/processed_data/fourgoals_12_allgauss_noclip_cmdinput/buf_train.pkl"

with open(file, "rb") as f:
    data = pickle.load(f)

for i, ep in enumerate(data):
    labels = [step[0]["stiffness_label"] for step in ep]
    c = Counter(labels)
    print(f"Episode {i:03d}: len={len(ep)}, labels={dict(sorted(c.items()))}")
PY
```


## Check Folder File Sizes
To check the size of all files in the folder, sorted and displayed in a human-readable format, use the following command:

```bash
du -ah | sort -h
```

- `du -ah`: Displays the disk usage of all files and directories in human-readable format.
- `--max-depth=1`: Limits the depth to the current folder.
- `sort -h`: Sorts the output by size in human-readable format.

Check the space of the whole computer
```bash
df -h /
```

## Check Folder Disk Usage

To check the disk usage of files and directories in the current folder, sorted by size, use the following command:

```bash
du -sh * | sort -h
```


## Persistent Terminal Session with tmux

To ensure your terminal session continues running even after disconnecting from SSH, follow these steps:

1. Start a new tmux session:
    ```bash
    tmux new -s factr_train
    ```

2. Activate your conda environment and run the training script:
    ```bash
    conda activate factr
    ./train_bc.sh
    ```

3. Detach from the tmux session without stopping it:
    - Press `Ctrl + B`, then `D`.

4. To reattach to the tmux session later:
    ```bash
    tmux attach -t factr_train
    ```

5. To terminate the session:
    ```bash
    tmux kill-session -t factr_train
    ```

6. Additional tmux commands:
    - List all active sessions:
      ```bash
      tmux ls
      ```
    - Reattach to a specific session:
      ```bash
      tmux attach -t <name>
      ```


rsync -avz -e ssh otake@192.168.1.172:~/factr_ws/raw_data/box_lift_3 .


## Delete All JSON Files in the Folder

To delete all `.json` files in the current folder, use the following command:

```bash
rm *.json
```

- `rm`: Command to remove files.
- `*.json`: Matches all files with the `.json` extension in the current folder.

**Caution:** This command is irreversible. Double-check the folder contents before running it.

conda install -c conda-forge roboticstoolbox-python
