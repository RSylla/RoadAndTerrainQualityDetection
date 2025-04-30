import os
import shutil
from pathlib import Path

def move_every_eighth_file(src_dir, dst_dir):
    # Create destination directory if it doesn't exist
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of files in source directory
    files = sorted(os.listdir(src_dir))
    
    # Move every 8th file
    for i, file in enumerate(files):
        if i % 8 == 0:  # Select every 8th file
            src_path = os.path.join(src_dir, file)
            dst_path = os.path.join(dst_dir, file)
            shutil.move(src_path, dst_path)
            print(f"Moved {file} to {dst_dir}")

# Define source and destination directories
image_src = "dataset/images/train"
image_dst = "dataset/images/test"
label_src = "dataset/labels/train"
label_dst = "dataset/labels/test"

# Move files from both directories
move_every_eighth_file(image_src, image_dst)
move_every_eighth_file(label_src, label_dst)

print("File moving completed.")