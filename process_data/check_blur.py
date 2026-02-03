import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from factr.utils import gaussian_2d_smoothing

H, W = 224, 224

img_path = "/home/ferdinand/factr_project/factr/process_data/data_to_process/20251107/visualizations/ep_4/curriculum_scale_0.png"

pil_img = Image.open(img_path).convert("RGB").resize((W, H))
img_np = np.array(pil_img).astype(np.float32) / 255.0
# Convert to torch tensor with shape (1, C, H, W)
img = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

FACTR_REPO = Path(__file__).resolve().parents[1]
save_dir = FACTR_REPO / "process_data" / "check_blur_outputs"
os.makedirs(save_dir, exist_ok=True)
print("")

# PREPOCESSING BLUR
size = H
kernel_size = int(0.05 * size)
kernel_size = kernel_size + (1 - kernel_size % 2)
print(f"Using kernel size: {kernel_size}")
# kernel_size = 11

img_for_cv = img_np.copy()
blurred_torch = transforms.GaussianBlur(kernel_size=kernel_size)(pil_img)
blurred_torch_np = np.array(blurred_torch).astype(np.float32) / 255.0

process_blurred_path = os.path.join(save_dir, "checkblur_original_processblur.png")
plt.imsave(process_blurred_path, blurred_torch_np)
print(f"Saved to {process_blurred_path}")


# SCALE BLUR
scale = 5.0

blurred = gaussian_2d_smoothing(img, scale=scale)

# Convert to numpy for plotting
img_np = img.squeeze(0).permute(1, 2, 0).numpy()
blurred_np = blurred.squeeze(0).permute(1, 2, 0).numpy()

# Ensure save directory is the process_data folder (same folder as this script)
orig_path = os.path.join(save_dir, "checkblur_original.png")
blur_path = os.path.join(save_dir, "checkblur_original_scale{scale}.png")
# Save images (expects HxWxC, floats in [0,1] are fine)
plt.imsave(orig_path, img_np)
plt.imsave(blur_path, blurred_np)
print(f"Saved original to {orig_path}")
print(f"Saved blurred to {blur_path}")
