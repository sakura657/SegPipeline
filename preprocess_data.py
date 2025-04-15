# preprocess_data.py
import pandas as pd
import numpy as np
import os
from glob import glob
import SimpleITK as sitk
from skimage.transform import resize
from tqdm import tqdm
import yaml
from box import Box
import argparse
import random
import json

# Import RLE and path functions from utils.py
from utils import rle_decode, load_config, parse_filepath_info, construct_path_from_info

def create_nifti_volume(group_df, config):
    """
    Loads 2D slices, decodes RLE masks, stacks them into a 3D volume,
    and saves as NIfTI files.
    """
    group_df = group_df.sort_values("slice_id").reset_index()
    case_id = group_df.loc[0, "case_id_str"]
    day_num = group_df.loc[0, "day_num_str"]
    group_id = f"{case_id}_{day_num}"
    print(f"Processing: {group_id}")

    img_slices = []
    mask_slices_lb = []
    mask_slices_sb = []
    mask_slices_st = []
    spacings = []

    target_h = config.data.get("target_height", None) # Optional target size before stacking
    target_w = config.data.get("target_width", None)

    for i, row in group_df.iterrows():
        img_path = row["f_path"]
        height, width, px_h, px_w = row["slice_h"], row["slice_w"], row["px_spacing_h"], row["px_spacing_w"]
        shape = (height, width)
        spacings.append((px_h, px_w))

        # Load Image
        img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img)[0] # Assuming single channel grayscale

        # Decode Masks
        lb_mask = rle_decode(row["lb_seg_rle"], shape) if pd.notna(row["lb_seg_rle"]) else np.zeros(shape, dtype=np.uint8)
        sb_mask = rle_decode(row["sb_seg_rle"], shape) if pd.notna(row["sb_seg_rle"]) else np.zeros(shape, dtype=np.uint8)
        st_mask = rle_decode(row["st_seg_rle"], shape) if pd.notna(row["st_seg_rle"]) else np.zeros(shape, dtype=np.uint8)

        # Optional Resize before stacking
        if target_h and target_w and (height != target_h or width != target_w):
             img_array = resize(img_array, (target_h, target_w), order=1, preserve_range=True, anti_aliasing=True).astype(img_array.dtype)
             lb_mask = resize(lb_mask, (target_h, target_w), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
             sb_mask = resize(sb_mask, (target_h, target_w), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
             st_mask = resize(st_mask, (target_h, target_w), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
             height, width = target_h, target_w # Update shape info
             # Assume spacing changes proportionally, though this might not be accurate
             # px_h = px_h * (row["slice_h"] / target_h)
             # px_w = px_w * (row["slice_w"] / target_w)
             # spacings[-1] = (px_h, px_w) # Update spacing (use average later?)

        img_slices.append(img_array)
        mask_slices_lb.append(lb_mask)
        mask_slices_sb.append(sb_mask)
        mask_slices_st.append(st_mask)

    # Stack slices - Resulting shape (D, H, W)
    img_vol = np.stack(img_slices, axis=0).astype(np.int16) # Use int16 based on PNG bit depth
    mask_vol_lb = np.stack(mask_slices_lb, axis=0).astype(np.uint8)
    mask_vol_sb = np.stack(mask_slices_sb, axis=0).astype(np.uint8)
    mask_vol_st = np.stack(mask_slices_st, axis=0).astype(np.uint8)

    # Combine masks into one multi-channel mask (C, D, H, W) -> (D, H, W, C) for SITK -> (C, D, H, W) after saving/loading
    mask_vol = np.stack([mask_vol_lb, mask_vol_sb, mask_vol_st], axis=0).astype(np.uint8) # Shape (C, D, H, W)

    # Create SimpleITK Images
    img_sitk = sitk.GetImageFromArray(img_vol, isVector=False) # D, H, W
    mask_sitk = sitk.GetImageFromArray(np.moveaxis(mask_vol, 0, -1), isVector=True) # D, H, W, C

    # Set Spacing and Origin
    # Use average spacing or spacing from middle slice? Using middle slice here.
    avg_px_h = np.mean([s[0] for s in spacings])
    avg_px_w = np.mean([s[1] for s in spacings])
    z_spacing = 3.0 # Given in competition description
    img_sitk.SetSpacing([avg_px_w, avg_px_h, z_spacing])
    mask_sitk.SetSpacing([avg_px_w, avg_px_h, z_spacing])
    img_sitk.SetOrigin([0,0,0])
    mask_sitk.SetOrigin([0,0,0])

    # Define output paths
    fold = config.fold # Assuming preprocessing per fold if desired
    # Ensure fold directory exists before accessing config.data.preprocessed_dir
    fold_dir = os.path.join(config.base_dir, "preprocessed_data", f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    # Update preprocessed_dir in config if it wasn't absolute
    # config.data.preprocessed_dir = fold_dir

    # Use absolute path based on base_dir for saving
    img_out_path = os.path.join(fold_dir, f"{group_id}_image.nii.gz")
    mask_out_path = os.path.join(fold_dir, f"{group_id}_mask.nii.gz")

    # Save NIfTI files
    sitk.WriteImage(img_sitk, img_out_path)
    sitk.WriteImage(mask_sitk, mask_out_path)

    return {
        "id": group_id,
        "image": img_out_path,
        "mask": mask_out_path,
        "fold": fold # Store fold info if splitting later
    }

def main(config_path, target_fold=None):
    config = load_config(config_path)
    # Ensure output and preprocessed directories exist relative to base_dir
    os.makedirs(os.path.join(config.base_dir, os.path.dirname(config.output_dir)), exist_ok=True)


    print("Loading train.csv...")
    df = pd.read_csv(os.path.join(config.base_dir, config.data.train_csv))

    print("Preprocessing DataFrame...")
    # 1. Add case, day, slice_id from 'id'
    df[["case_id_str", "day_num_str", "slice_id"]] = df["id"].str.split("_", expand=True)
    df["slice_id"] = df["slice_id"].astype(int)
    df["case_id_str"] = df["case_id_str"].str.replace("case","case") # Keep 'case' prefix for consistency
    df["day_num_str"] = df["day_num_str"].str.replace("day","day") # Keep 'day' prefix

    # 2. Find all image paths
    print("Finding image files...")
    all_train_images = glob(os.path.join(config.base_dir, config.data.raw_data_dir, "train", "**", "*.png"), recursive=True)
    path_df_info = [parse_filepath_info(p, config.base_dir, config.data.raw_data_dir) for p in tqdm(all_train_images, desc="Parsing paths")]
    path_df = pd.DataFrame(filter(None, path_df_info)) # Filter out potential parsing errors

    # 3. Merge df with paths and metadata
    print("Merging DataFrame with file paths...")
    # Ensure types match for merging
    df["slice_id"] = df["slice_id"].astype(int)
    path_df["slice_id"] = path_df["slice_id"].astype(int)
    # case_id_str and day_num_str should already be strings like 'caseXXX', 'dayYYY'
    df = pd.merge(df, path_df, on=["case_id_str", "day_num_str", "slice_id"], how="left")

    # Handle missing files
    missing_files = df[df['f_path'].isnull()]
    if not missing_files.empty:
        print(f"Warning: Could not find image paths for {len(missing_files)} rows. IDs: {missing_files['id'].tolist()}")
        df = df.dropna(subset=['f_path'])

    # 4. Pivot RLE masks to columns
    print("Pivoting RLE masks...")
    df_pivot = df.pivot(index="id", columns="class", values="segmentation").reset_index()
    # Keep only one row per image slice, merge metadata back
    df_unique = df.drop(columns=["class", "segmentation"]).drop_duplicates(subset="id").reset_index(drop=True)
    df_processed = pd.merge(df_unique, df_pivot, on="id", how="left")
    # Rename pivoted columns for clarity
    df_processed = df_processed.rename(columns={
        "large_bowel": "lb_seg_rle",
        "small_bowel": "sb_seg_rle",
        "stomach": "st_seg_rle"
    })

    # --- Create 3D NIfTI volumes ---
    print("Grouping data by case and day...")
    grouped = df_processed.groupby(["case_id_str", "day_num_str"])
    nifti_file_list = [] # To store info for dataset JSON

    num_groups = len(grouped)
    print(f"Found {num_groups} unique case-day groups.")

    for i, (group_name, group_df) in enumerate(tqdm(grouped, desc="Creating NIfTI volumes")):
        config.fold = i % 5 # Example 5-fold split based on case-day group index
        if target_fold is not None and config.fold != target_fold:
            continue

        # Create NIfTI and get info dict
        file_info = create_nifti_volume(group_df, config)
        nifti_file_list.append(file_info)


    # --- Save dataset JSON list(s) ---
    print("Saving dataset JSON file list(s)...")
    all_folds_data = {}
    for item in nifti_file_list:
        fold = item['fold']
        if fold not in all_folds_data:
            all_folds_data[fold] = {'train': [], 'val': []} # Assuming split within fold later

    # Example: Split each fold's data (can be done here or in training script)
    # For simplicity, let's just save all data per fold for now.
    # The training script can handle the train/val split from this list.
    for fold, data_list in all_folds_data.items():
        fold_dir = os.path.join(config.base_dir, "preprocessed_data", f"fold_{fold}")
        output_json_path = os.path.join(fold_dir, f"dataset_fold_{fold}.json")
        fold_items = [item for item in nifti_file_list if item['fold'] == fold]

        # Simple random split for demonstration (adjust as needed)
        random.shuffle(fold_items)
        split_idx = int(len(fold_items) * 0.8) # 80/20 split
        train_files = fold_items[:split_idx]
        val_files = fold_items[split_idx:]

        dataset_split = {'train': train_files, 'val': val_files}

        with open(output_json_path, 'w') as f:
            json.dump(dataset_split, f, indent=4)
        print(f"Saved dataset JSON for fold {fold} to {output_json_path}")
        # Update config path (optional, can also be constructed in train script)
        # config.data.data_list_json = output_json_path.replace(config.base_dir+'/', '')


    print("Preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Kaggle GI Tract data into 3D NIfTI volumes.")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    parser.add_argument("-f", "--fold", type=int, default=None, help="Specific fold to process (optional). Processes all folds if None.")
    args = parser.parse_args()
    main(args.config, args.fold)