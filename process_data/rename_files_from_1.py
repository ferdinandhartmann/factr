import os

# Folder containing your .pkl files
FOLDER = "/home/ferdinand/activeinference/factr/process_data/data_to_process/bld_soft/data/"   # "." means current folder; change if needed

PREFIX = "ep_"
EXT = ".pkl"

def main():
    # Rename backwards from 60 to 0
    for i in range(60, -1, -1):
        old_name = f"{PREFIX}{i:02d}{EXT}"
        new_name = f"{PREFIX}{i+1:02d}{EXT}"

        old_path = os.path.join(FOLDER, old_name)
        new_path = os.path.join(FOLDER, new_name)

        if os.path.exists(old_path):
            print(f"Renaming {old_name} → {new_name}")
            os.rename(old_path, new_path)
        else:
            print(f"Skipping {old_name}: does not exist")

    print("Done.")

if __name__ == "__main__":
    main()
