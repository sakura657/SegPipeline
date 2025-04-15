# inference.py
import argparse
import os
import gc
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import SimpleITK as sitk
from glob import glob
import json

from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

# Import local modules
from utils import load_config, rle_encode, parse_filepath_info
from model_factory import get_model
from transforms import get_inference_transforms, get_postprocessing_transforms


def get_test_data_list(config):
    """Creates the list of data dictionaries for inference."""
    # This needs to adapt based on whether it's the hidden test set or predicting specific files
    mode = config.inference.get('mode', 'test')
    data_list = []

    if mode == 'test':
        # Logic for Kaggle hidden test set
        # Assumes test data has same structure as train: <base>/test/caseX/caseX_dayY/scans/...
        test_img_dir = os.path.join(config.base_dir, config.data.raw_data_dir, "test")
        if not os.path.exists(test_img_dir):
             print(f"Warning: Test directory not found at {test_img_dir}. Using placeholder sample_submission logic.")
             # Fallback: use sample_submission to get IDs, assume paths need construction or are dummy
             sub_df = pd.read_csv(os.path.join(config.base_dir, config.data.raw_data_dir, "sample_submission.csv"))
             if 'predicted' in sub_df.columns: # Check if it's the real submission file or sample
                 sub_df = sub_df.drop(columns=['predicted']).drop_duplicates(subset=['id', 'class'])
             else: # Sample submission only has id, class
                  sub_df = sub_df.drop_duplicates(subset=['id', 'class'])

             # Attempt to parse info from ID like in preprocessing
             sub_df[["case_id_str", "day_num_str", "slice_id"]] = sub_df["id"].str.split("_", expand=True)
             # Need a way to link ID to a file path - THIS IS THE HARD PART FOR HIDDEN TEST
             # For now, let's assume we need to process slice-by-slice based on the ID
             # We might need a dummy path or handle loading differently
             print("Warning: Using sample_submission. Actual file paths for hidden test set are unknown.")
             for idx, row in sub_df.iterrows():
                 # Cannot easily form 3D NIfTI path. Inference might need to be 2.5D
                 # Or adapt the pipeline to take 2D slices if 3D model was trained
                 # THIS REQUIRES SIGNIFICANT CHANGES if pipeline is 3D NIfTI based.
                 # --- Placeholder for 2D approach ---
                 # 1. Construct expected 2D path based on ID (best guess)
                 # slice_info = parse_filepath_info(???) # Need a function based on ID only? Difficult.
                 # dummy_path = f"placeholder_{row['id']}.png"
                 # data_list.append({"image": dummy_path, "id": row['id']}) # Need transform to handle this
                 pass # Skip for now as 3D pipeline is default
             if not data_list:
                  print("Error: Cannot proceed with inference using sample_submission for a 3D NIfTI pipeline without known test file paths.")
                  return []

        else:
             # Option: Preprocess test set like train set into NIfTI first
             # Or: Find all test NIfTI files if preprocess_data was run on test set
             nifti_files = glob(os.path.join(config.inference.get("preprocessed_test_dir", os.path.join(config.base_dir, "preprocessed_data", "test")), "*.nii.gz"))
             image_files = sorted([f for f in nifti_files if "_image.nii.gz" in f])
             if not image_files:
                 print(f"Error: No preprocessed NIfTI image files found for inference in {config.inference.get('preprocessed_test_dir')}")
                 return []
             for img_path in image_files:
                 # Extract ID from filename (e.g., caseX_dayY)
                 img_id = os.path.basename(img_path).replace("_image.nii.gz", "")
                 data_list.append({"image": img_path, "id": img_id}) # ID here is case_day ID

    elif mode == 'predict':
        # Load specific files listed in config
        predict_files_config = config.inference.get('predict_files', [])
        for item in predict_files_config:
            if isinstance(item, str): # Just a path
                img_path = os.path.join(config.base_dir, item)
                img_id = os.path.basename(img_path).split('.')[0] # Use filename as ID
                data_list.append({"image": img_path, "id": img_id})
            elif isinstance(item, dict) and 'image' in item: # Dict with path and maybe ID
                img_path = os.path.join(config.base_dir, item['image'])
                img_id = item.get('id', os.path.basename(img_path).split('.')[0])
                data_list.append({"image": img_path, "id": img_id})
            else:
                print(f"Warning: Skipping invalid item in predict_files: {item}")
    else:
        raise ValueError(f"Unknown inference mode: {mode}")

    print(f"Found {len(data_list)} items for inference.")
    return data_list


def main(config_path):
    # Load Configuration
    config = load_config(config_path)
    print("Inference Configuration loaded.")

    # Set Seed (optional for inference, but good practice)
    if config.get("seed", -1) >= 0:
        set_determinism(config.seed)

    # Directories
    checkpoint_path = os.path.join(config.base_dir, config.inference.checkpoint_path)
    output_dir = os.path.join(config.base_dir, config.inference.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create Test Dataset and DataLoader
    print("Creating inference dataset...")
    inference_data_list = get_test_data_list(config)
    if not inference_data_list:
        print("No data found for inference. Exiting.")
        return

    infer_transforms = get_inference_transforms(config)
    # Use standard Dataset for inference, no caching needed usually
    infer_ds = Dataset(data=inference_data_list, transform=infer_transforms)
    infer_loader = DataLoader(
        infer_ds,
        batch_size=config.inference.batch_size,
        shuffle=False, # No shuffling for inference
        num_workers=config.training.get('num_workers', 4) # Reuse training workers config
    )

    # Create Model and Load Checkpoint
    print("Creating model...")
    model = get_model(config).to(config.device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    # Handle different checkpoint formats (e.g., containing 'model' key)
    if 'model' in checkpoint:
         model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
         model.load_state_dict(checkpoint['state_dict'])
    else:
         model.load_state_dict(checkpoint)
    model.eval()

    # Post-processing Transforms
    post_transforms = get_postprocessing_transforms(config)

    # Inference Loop Variables
    results_list = [] # To store RLE encoded results
    roi_size = config.inference.roi_size
    sw_batch_size = config.inference.sw_batch_size
    overlap = config.inference.overlap
    mode = config.inference.mode
    run_tta = config.inference.get('run_tta', False)

    progress_bar = tqdm(infer_loader, desc="Running Inference")
    with torch.no_grad():
        for batch_data in progress_bar:
            infer_inputs = batch_data["image"].to(config.device)
            # Store metadata needed for saving/submission
            input_ids = batch_data["id"] # This is case_day ID for NIfTI input
            input_meta = batch_data.get("image_meta_dict", None)

            # Sliding window inference
            outputs = sliding_window_inference(infer_inputs, roi_size, sw_batch_size, model, overlap=overlap, mode=mode)

            # Test Time Augmentation (TTA) - Example flips
            if run_tta:
                tta_outputs = outputs.clone()
                tta_count = 1
                # Flip Dim 2 (H)
                flip_dims = [2]
                flip_inputs = torch.flip(infer_inputs, dims=flip_dims)
                flip_outputs = sliding_window_inference(flip_inputs, roi_size, sw_batch_size, model, overlap=overlap, mode=mode)
                tta_outputs += torch.flip(flip_outputs, dims=flip_dims)
                tta_count += 1
                 # Flip Dim 3 (W)
                flip_dims = [3]
                flip_inputs = torch.flip(infer_inputs, dims=flip_dims)
                flip_outputs = sliding_window_inference(flip_inputs, roi_size, sw_batch_size, model, overlap=overlap, mode=mode)
                tta_outputs += torch.flip(flip_outputs, dims=flip_dims)
                tta_count += 1
                 # Flip Dim 4 (D)
                flip_dims = [4]
                flip_inputs = torch.flip(infer_inputs, dims=flip_dims)
                flip_outputs = sliding_window_inference(flip_inputs, roi_size, sw_batch_size, model, overlap=overlap, mode=mode)
                tta_outputs += torch.flip(flip_outputs, dims=flip_dims)
                tta_count += 1
                # Average TTA results
                outputs = tta_outputs / tta_count
                del flip_inputs, flip_outputs, tta_outputs

            # Apply post-processing (sigmoid/softmax, threshold, maybe connected components)
            # Input to post_transforms needs to be a dictionary
            post_input_list = [{"pred": item} for item in outputs]
            processed_batch = [post_transforms(item) for item in post_input_list]
            final_preds = torch.stack([item['pred'] for item in processed_batch]).cpu().numpy().astype(np.uint8) # (B, C, H, W, D)

            # --- RLE Encoding and Submission Formatting ---
            # Need to iterate through batch, classes, and *slices*
            for batch_idx in range(final_preds.shape[0]):
                case_day_id = input_ids[batch_idx] # e.g., case123_day20
                pred_volume = final_preds[batch_idx] # (C, H, W, D)

                # Get original 2D slice IDs and dimensions if possible (needed for RLE)
                # This is tricky if only NIfTI is loaded. We might need to:
                # 1. Load original train.csv mapping during inference.
                # 2. Assume slice index corresponds to Z dim (if no resizing occurred).
                # 3. Save slice metadata alongside NIfTI during preprocessing.

                # Approach 2: Assume Z dim corresponds to slices (simplest if valid)
                num_slices = pred_volume.shape[3]
                original_h, original_w = pred_volume.shape[1], pred_volume.shape[2] # Assumes no resizing or need to invert resize

                # Try to get original case/day/slice info (Requires loading train.csv or similar mapping)
                # This part needs refinement based on how slice IDs are tracked.
                # For now, generating dummy slice IDs based on Z index.
                case_str, day_str = case_day_id.split('_') # caseXXX, dayYY

                for slice_idx in range(num_slices):
                    slice_id_num = slice_idx # Placeholder - NEEDS CORRECTION TO MATCH ORIGINAL SLICE ID
                    base_submission_id = f"{case_str}_{day_str}_{slice_id_num}" # Reconstruct the ID required by submission

                    # Extract prediction for this slice
                    pred_slice_lb = pred_volume[0, :, :, slice_idx] # (H, W)
                    pred_slice_sb = pred_volume[1, :, :, slice_idx]
                    pred_slice_st = pred_volume[2, :, :, slice_idx]

                    # Encode each class mask
                    rle_lb = rle_encode(pred_slice_lb)
                    rle_sb = rle_encode(pred_slice_sb)
                    rle_st = rle_encode(pred_slice_st)

                    # Append results for submission CSV
                    if len(rle_lb) > 0: # Only add row if mask is not empty
                        results_list.append({"id": base_submission_id, "class": "large_bowel", "predicted": rle_lb})
                    if len(rle_sb) > 0:
                         results_list.append({"id": base_submission_id, "class": "small_bowel", "predicted": rle_sb})
                    if len(rle_st) > 0:
                         results_list.append({"id": base_submission_id, "class": "stomach", "predicted": rle_st})

                # Optional: Save NIfTI prediction
                if config.inference.get('save_nifti', False):
                    pred_sitk = sitk.GetImageFromArray(np.moveaxis(pred_volume, 0, -1), isVector=True) # (H, W, D, C)
                    # Try to copy spacing/origin from input metadata if available
                    if input_meta and batch_idx < len(input_meta):
                         try:
                             affine = input_meta[batch_idx].get('original_affine', np.eye(4))
                             # SITK uses LPS, MONAI uses RAS - conversion might be needed if affine is used directly
                             # For spacing/origin, direct copy is often ok if orientation wasn't changed drastically
                             pred_sitk.SetSpacing(affine[:3, :3].diagonal())
                             pred_sitk.SetOrigin(affine[:3, 3])
                         except Exception as e:
                              print(f"Warning: Could not set spacing/origin from metadata. {e}")

                    nifti_filename = os.path.join(output_dir, f"{case_day_id}_pred.nii.gz")
                    sitk.WriteImage(pred_sitk, nifti_filename)

            del infer_inputs, outputs, final_preds, batch_data
            torch.cuda.empty_cache()
            gc.collect()

    # Create Submission DataFrame
    print("Creating submission file...")
    submission_df = pd.DataFrame(results_list)

    # Ensure required IDs from sample_submission are present, even if empty prediction
    sample_sub_df = pd.read_csv(os.path.join(config.base_dir, config.data.raw_data_dir, "sample_submission.csv"))
    # Keep only id and class from sample
    sample_sub_df = sample_sub_df[['id', 'class']].drop_duplicates()

    # Merge predictions, filling missing with empty string
    final_submission = pd.merge(sample_sub_df, submission_df, on=['id', 'class'], how='left')
    final_submission['predicted'] = final_submission['predicted'].fillna('') # Fill NaN with empty string

    # Save Submission CSV
    sub_filename = config.inference.get('submission_filename', 'submission.csv')
    sub_path = os.path.join(config.base_dir, output_dir, sub_filename) # Save inside output_dir
    final_submission.to_csv(sub_path, index=False)
    print(f"Submission file saved to: {sub_path}")

    print("Inference finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MONAI model inference for segmentation.")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    # Add command-line arguments to override config, e.g., checkpoint path
    parser.add_argument("--ckpt", type=str, help="Override checkpoint path in config.")

    args = parser.parse_args()

    # Override config if needed
    config = load_config(args.config)
    if args.ckpt:
        config.inference.checkpoint_path = args.ckpt
        # Save temp config or pass object? For simplicity, just use loaded+modified config
        temp_config_path = "temp_inference_config.yaml"
        with open(temp_config_path, 'w') as f:
             yaml.dump(config.to_dict(), f)
        main(temp_config_path)
        os.remove(temp_config_path)
    else:
        main(args.config)