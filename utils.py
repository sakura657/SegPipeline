# utils.py
import pandas as pd
import numpy as np
import os
import yaml
from box import Box # Makes accessing config easier (dict['key'] vs dict.key)
import torch
from monai.transforms import MapTransform
from skimage.measure import label as sk_label, regionprops
import re # For parsing filenames

# --- RLE Encoding/Decoding ---
def rle_decode(mask_rle: str, shape: tuple[int, int]) -> np.ndarray:
    """Decodes RLE string to binary mask."""
    if pd.isna(mask_rle) or not isinstance(mask_rle, str) or len(mask_rle) == 0:
        return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T # Transpose to match H, W format if RLE is column-major

def rle_encode(img: np.ndarray) -> str:
    """Encodes binary mask (H, W) into RLE string (column-major)."""
    pixels = img.T.flatten() # Transpose for column-major order
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# --- Configuration Loading ---
def load_config(config_path: str) -> Box:
    """Loads YAML config file into a Box object with variable substitution."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Convert to Box first to allow attribute access during substitution
    config_box = Box(config_dict, default_box=True, default_box_attr=None)

    def substitute_vars(node):
        if isinstance(node, Box):
            # Iterate through a copy of items for safe modification
            for key, value in list(node.items()):
                node[key] = substitute_vars(value)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                node[i] = substitute_vars(item)
        elif isinstance(node, str) and node.startswith("${") and node.endswith("}"):
            var_path = node[2:-1].split('.')
            value = config_box # Start substitution lookup from root
            try:
                for p in var_path:
                    value = getattr(value, p)
                    if value is None: # Check if substitution path leads to None
                         print(f"Warning: Variable substitution for {node} resulted in None at segment '{p}'. Using original string.")
                         return node
                # Check if the substituted value itself needs substitution
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                     # print(f"Recursive substitution for {node} -> {value}")
                     return substitute_vars(value) # Recursive substitution
                # print(f"Substituted {node} -> {value}")
                return value
            except (AttributeError, KeyError, TypeError) as e:
                print(f"Warning: Variable substitution failed for {node}. Error: {e}. Using original string.")
                return node # Return original string if substitution fails
        return node

    # Perform substitution directly on the Box object
    substitute_vars(config_box)
    # Ensure base_dir is absolute or resolved relative to config file location
    config_box.base_dir = os.path.abspath(os.path.join(os.path.dirname(config_path), config_box.base_dir))

    return config_box


# --- Kaggle Data Path/Info Parsing (for preprocess_data.py) ---
def parse_filepath_info(path: str, base_dir: str, raw_data_dir_relative: str) -> dict | None:
    """Extracts metadata (case, day, slice, dims, spacing) from image filepath."""
    try:
        # Make raw_data_dir absolute for reliable splitting
        raw_data_dir_abs = os.path.abspath(os.path.join(base_dir, raw_data_dir_relative))
        rel_path = os.path.relpath(path, raw_data_dir_abs)
        # Expected rel_path: train/caseXXX/caseXXX_dayYYY/scans/slice_ZZZZ_W_H_pxW_pxH.png
        parts = rel_path.split(os.sep)
        # print(parts) # Debugging

        if len(parts) != 4 or parts[0] != 'train' or parts[2] != 'scans':
            print(f"Warning: Unexpected path structure: {path}")
            return None

        case_day_str = parts[1] # caseXXX_dayYYY
        filename = parts[3] # slice_ZZZZ_W_H_pxW_pxH.png

        case_match = re.match(r"(case\d+)_day(\d+)", case_day_str)
        if not case_match:
            print(f"Warning: Could not parse case/day from folder: {case_day_str}")
            return None
        case_id_str = case_match.group(1)
        day_num_str = "day" + case_match.group(2) # Add 'day' prefix

        file_match = re.match(r"slice_(\d+)_(\d+)_(\d+)_([\d\.]+)_([\d\.]+)\.png", filename)
        if not file_match:
            print(f"Warning: Could not parse filename: {filename}")
            return None

        slice_id = int(file_match.group(1))
        slice_w = int(file_match.group(2)) # Width is first dimension in filename
        slice_h = int(file_match.group(3)) # Height is second dimension in filename
        px_spacing_w = float(file_match.group(4))
        px_spacing_h = float(file_match.group(5))

        return {
            "f_path": path,
            "case_id_str": case_id_str,
            "day_num_str": day_num_str,
            "slice_id": slice_id,
            "slice_h": slice_h,
            "slice_w": slice_w,
            "px_spacing_h": px_spacing_h,
            "px_spacing_w": px_spacing_w,
        }
    except Exception as e:
        print(f"Error parsing path {path}: {e}")
        return None

def construct_path_from_info(info: dict, config: Box) -> str:
     """Constructs the full image path from parsed info (inverse of parse)."""
     # This might be needed if generating paths dynamically later
     filename = f"slice_{info['slice_id']:04d}_{info['slice_w']}_{info['slice_h']}_{info['px_spacing_w']:.2f}_{info['px_spacing_h']:.2f}.png"
     # Ensure day format matches folder name (e.g., caseXXX_dayYY)
     day_folder_num = info['day_num_str'].replace('day','')
     path = os.path.join(
         config.base_dir,
         config.data.raw_data_dir,
         "train",
         f"{info['case_id_str']}",
         f"{info['case_id_str']}_day{day_folder_num}",
         "scans",
         filename
     )
     return path


# --- MONAI Utils ---
def create_checkpoint(model, optimizer, epoch, scheduler=None, scaler=None):
    """Creates a checkpoint dictionary for saving."""
    # Adapted from uploaded utils.py
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"): # Check if scaler has state_dict
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint

class DebugShapeD(MapTransform):
    """Print shapes for debugging MONAI transforms."""
    def __init__(self, keys, label="", enabled=True):
        super().__init__(keys, allow_missing_keys=True)
        self.label = label
        self.enabled = enabled

    def __call__(self, data):
        if not self.enabled:
            return data
        d = dict(data)
        print(f"--- Debug Transform: {self.label} ---")
        for key in self.key_iterator(d):
            item = d[key]
            shape = getattr(item, 'shape', 'N/A')
            dtype = getattr(item, 'dtype', 'N/A')
            device = getattr(item, 'device', 'N/A')
            meta_dict = d.get(f"{key}_meta_dict", None)
            affine = meta_dict.get('affine', 'N/A') if meta_dict else 'N/A'
            print(f"  Key='{key}': Shape={shape}, Dtype={dtype}, Device={device}, Affine=\n{affine}")
        print(f"--- End Debug: {self.label} ---")
        return d