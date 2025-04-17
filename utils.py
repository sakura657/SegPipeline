# utils.py
import pandas as pd
import numpy as np
import os
import yaml
from box import Box
import torch
from monai.transforms import MapTransform
from skimage.measure import label as sk_label, regionprops
import re
import logging
import copy # Import copy for deep copying config

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- RLE Encoding/Decoding (Keep as before) ---
def rle_decode(mask_rle: str, shape: tuple[int, int]) -> np.ndarray:
    # ... (implementation unchanged) ...
    if pd.isna(mask_rle) or not isinstance(mask_rle, str) or len(mask_rle) == 0:
        return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def rle_encode(img: np.ndarray) -> str:
    # ... (implementation unchanged) ...
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# --- Configuration Loading (REVISED AGAIN - Recursive Substitution) ---

def _recursive_substitute(node, root_config):
    """Recursively performs ${var} substitution within nodes (dicts, lists, strings)."""
    if isinstance(node, dict):
        # Process dictionary values recursively
        new_dict = {}
        for key, value in node.items():
            new_dict[key] = _recursive_substitute(value, root_config)
        return new_dict
    elif isinstance(node, list):
        # Process list items recursively
        new_list = []
        for item in node:
            new_list.append(_recursive_substitute(item, root_config))
        return new_list
    elif isinstance(node, str):
        # Perform substitution on strings
        original_string = node
        pattern = re.compile(r"\$\{(.+?)\}")
        matches = pattern.findall(original_string)
        temp_string = original_string

        if matches:
            for var_path_str in matches:
                var_path = var_path_str.split('.')
                value = root_config # Look up from the root dict/Box
                try:
                    for p in var_path:
                        if isinstance(value, dict):
                            value = value.get(p) # Use .get for safety
                        elif isinstance(value, Box):
                             value = getattr(value, p, None) # Use getattr for Box
                        else:
                             # Cannot traverse further
                             value = None
                             break
                        if value is None:
                             break # Stop if any part of the path is None

                    # Substitute only if value is found and is a primitive type
                    if value is not None and isinstance(value, (str, int, float, bool)):
                        placeholder = f"${{{var_path_str}}}"
                        # Use replace carefully, might replace unintended parts if placeholders overlap
                        # A more robust regex replace might be needed for complex cases
                        temp_string = temp_string.replace(placeholder, str(value))
                    # else: keep original placeholder if value not found or complex type
                except Exception as e:
                     # Keep original placeholder on error
                     # logging.warning(f"Substitution lookup failed for '${{{var_path_str}}}' in '{original_string}'. Error: {e}")
                     pass
        return temp_string # Return the (potentially) substituted string
    else:
        # Return non-dict/list/str nodes as is
        return node


def load_config(config_path: str) -> Box:
    """Loads YAML, performs simple substitutions recursively, resolves paths."""
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {config_path}: {e}")
        raise

    # Perform substitutions using the recursive helper
    # Multiple passes to handle nested dependencies
    max_passes = 5
    substituted_config = copy.deepcopy(config_dict) # Start with a copy
    previous_config_str = ""
    passes = 0

    while passes < max_passes:
        passes += 1
        current_config_str = str(substituted_config)
        if current_config_str == previous_config_str:
             #logging.debug(f"Substitution converged after {passes-1} passes.")
             break # Stop if no changes were made in the last pass
        previous_config_str = current_config_str
        substituted_config = _recursive_substitute(substituted_config, substituted_config) # Pass config itself for lookups

    if passes == max_passes and str(substituted_config) != previous_config_str:
         logging.warning("Max substitution passes reached, potential circular dependency or unresolved variables remain.")

    # Convert final dictionary to Box AFTER substitutions
    config_box = Box(substituted_config, default_box=True, default_box_attr=None)

    # --- Path Resolution (Same as previous version) ---
    # Resolve base_dir relative to config file location FIRST
    if config_box.get('base_dir') is None:
        config_box.base_dir = '.'
    config_file_dir = os.path.dirname(os.path.abspath(config_path))
    config_box.base_dir = os.path.abspath(os.path.join(config_file_dir, config_box.base_dir))
    logging.info(f"Resolved base_dir: {config_box.base_dir}")

    # Explicitly resolve known paths relative to base_dir
    try:
        # 1. Resolve raw_data_dir
        raw_data_dir = getattr(getattr(config_box, 'data', Box()), 'raw_data_dir', None)
        if raw_data_dir and isinstance(raw_data_dir, str) and not os.path.isabs(raw_data_dir):
            config_box.data.raw_data_dir = os.path.join(config_box.base_dir, raw_data_dir)
        if config_box.data.raw_data_dir: # Ensure absolute even if initially provided as such
             config_box.data.raw_data_dir = os.path.abspath(config_box.data.raw_data_dir)
        logging.info(f"Resolved data.raw_data_dir: {config_box.data.raw_data_dir}")

        # 2. Resolve train_csv (use resolved raw_data_dir)
        train_csv_path = getattr(getattr(config_box, 'data', Box()), 'train_csv', None)
        # Check if raw_data_dir was successfully resolved before proceeding
        if train_csv_path and isinstance(train_csv_path, str) and config_box.data.raw_data_dir and os.path.isdir(config_box.data.raw_data_dir):
            if not os.path.isabs(train_csv_path):
                 config_box.data.train_csv = os.path.join(config_box.data.raw_data_dir, train_csv_path)
            config_box.data.train_csv = os.path.abspath(config_box.data.train_csv)
            logging.info(f"Resolved data.train_csv: {config_box.data.train_csv}")
        elif not train_csv_path:
             logging.warning("data.train_csv not found in config.")
        elif not config_box.data.raw_data_dir or not os.path.isdir(config_box.data.raw_data_dir):
             logging.error(f"Cannot resolve data.train_csv because data.raw_data_dir ('{config_box.data.raw_data_dir}') is not set or not a valid directory.")
             # Optionally raise an error here if train_csv is essential
             # raise ValueError("data.raw_data_dir is required and must be valid to resolve train_csv")


        # 3. Resolve other paths relative to base_dir
        paths_to_resolve_relative_to_base = [
            ('output_dir',),
            ('data', 'preprocessed_dir'),
            ('training', 'checkpoint_dir'),
            ('training', 'log_dir'),
            ('training', 'load_checkpoint'),
            ('inference', 'checkpoint_path'),
            ('inference', 'output_dir'),
            ('inference', 'preprocessed_test_dir'),
            ('data', 'persistent_cache_dir')
        ]
        for key_tuple in paths_to_resolve_relative_to_base:
             # ... (path resolution logic using os.path.join(base_dir, ...) remains the same) ...
             current_level = config_box
             key_path_str = '.'.join(key_tuple)
             try:
                 for i, key in enumerate(key_tuple[:-1]):
                     current_level = getattr(current_level, key)
                     if current_level is None: raise AttributeError
                 final_key = key_tuple[-1]
                 original_path = getattr(current_level, final_key, None)
                 if original_path and isinstance(original_path, str):
                     resolved_path = os.path.join(config_box.base_dir, original_path) if not os.path.isabs(original_path) else original_path
                     setattr(current_level, final_key, resolved_path)
                     logging.debug(f"Resolved path {key_path_str}: {original_path} -> {resolved_path}")
             except (AttributeError, KeyError):
                  logging.debug(f"Path key not found or intermediate None, skipping resolution: {key_path_str}")
                  continue

    except AttributeError as e:
        logging.error(f"Error resolving paths: Missing key in config structure - {e}")
        raise
    except Exception as e:
         logging.error(f"Unexpected error during path resolution: {e}")
         raise

    logging.info("Configuration loading and path resolution complete.")
    return config_box

# --- Kaggle Data Path/Info Parsing (Keep as before) ---
def parse_filepath_info(path: str, base_dir: str, raw_data_dir_relative: str) -> dict | None:
    # ... (implementation unchanged) ...
    try:
        # Make raw_data_dir absolute for reliable splitting
        raw_data_dir_abs = os.path.abspath(os.path.join(base_dir, raw_data_dir_relative))
        # Check if raw_data_dir_abs exists before proceeding
        if not os.path.isdir(raw_data_dir_abs):
             logging.warning(f"Raw data directory not found during path parsing: {raw_data_dir_abs}")
             raw_data_dir_abs = base_dir # Fallback

        rel_path = os.path.relpath(os.path.abspath(path), raw_data_dir_abs)
        parts = rel_path.split(os.sep)

        # Adjust expected parts length based on the actual structure found
        # train / caseXXX / caseXXX_dayYY / scans / slice_ZZZ...png  -> 5 parts
        expected_parts_len = 5
        start_index = 0

        # Check if path starts relative to 'train' if relpath didn't work as expected
        if parts[0] != 'train':
             try:
                 start_index = parts.index('train')
                 parts = parts[start_index:]
             except ValueError:
                  logging.warning(f"Unexpected path structure (missing 'train'): {path} (Parts: {parts})")
                  return None

        # Check path structure and length using CORRECT indices
        if len(parts) != expected_parts_len or parts[0] != 'train' or parts[3] != 'scans':
            logging.warning(f"Unexpected path structure: {path} (Parts: {parts})")
            return None

        # Use CORRECT indices based on 5 parts
        # parts[0] = train
        # parts[1] = caseXXX
        # parts[2] = caseXXX_dayYY
        # parts[3] = scans
        # parts[4] = slice_ZZZ...png
        case_day_str = parts[2] # CORRECT index
        filename = parts[4] # CORRECT index


        case_match = re.match(r"(case\d+)_day(\d+)", case_day_str)
        if not case_match: return None
        case_id_str, day_num_str = case_match.group(1), "day" + case_match.group(2)
        file_match = re.match(r"slice_(\d+)_(\d+)_(\d+)_([\d\.]+)_([\d\.]+)\.png", filename)
        if not file_match: return None
        slice_id, slice_w, slice_h = int(file_match.group(1)), int(file_match.group(2)), int(file_match.group(3))
        px_spacing_w, px_spacing_h = float(file_match.group(4)), float(file_match.group(5))

        return {"f_path": os.path.abspath(path), "case_id_str": case_id_str, "day_num_str": day_num_str, "slice_id": slice_id, "slice_h": slice_h, "slice_w": slice_w, "px_spacing_h": px_spacing_h, "px_spacing_w": px_spacing_w}
    except Exception as e:
        logging.error(f"Error parsing path {path}: {e}"); return None


# --- MONAI Utils (Keep as before) ---
def create_checkpoint(model, optimizer, epoch, scheduler=None, scaler=None):
    # ... (implementation unchanged) ...
    checkpoint = {"model": model.state_dict(),"optimizer": optimizer.state_dict(),"epoch": epoch,}
    if scheduler is not None: checkpoint["scheduler"] = scheduler.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"): checkpoint["scaler"] = scaler.state_dict()
    return checkpoint

class DebugShapeD(MapTransform):
    # ... (implementation unchanged) ...
    def __init__(self, keys, label="", enabled=True): super().__init__(keys, allow_missing_keys=True); self.label, self.enabled = label, enabled
    def __call__(self, data):
        if not self.enabled: return data
        d = dict(data); print(f"--- Debug Transform: {self.label} ---")
        for key in self.key_iterator(d):
            item=d[key]; shape=getattr(item, 'shape', 'N/A'); dtype=getattr(item, 'dtype', 'N/A'); device=getattr(item, 'device', 'N/A'); meta_dict=d.get(f"{key}_meta_dict"); affine=meta_dict.get('affine','N/A') if meta_dict else 'N/A'
            print(f"  Key='{key}': Shape={shape}, Dtype={dtype}, Device={device}, Affine=\n{affine}")
        print(f"--- End Debug: {self.label} ---"); return d