# train.py
import argparse
import os
import shutil
import time
import gc
import yaml # Import yaml for saving config

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

# Import local modules
from utils import load_config, create_checkpoint # Removed DebugShapeD import, assuming not needed here
from dataset import get_datasets, get_dataloaders
from model_factory import get_model
from losses import get_loss_function
from metrics import get_metric_functions
from transforms import get_postprocessing_transforms


def run_train_epoch(model, loader, optimizer, scheduler, loss_fn, scaler, config, epoch, writer, global_step):
    # --- Start of run_train_epoch ---
    model.train()
    epoch_loss = 0.0
    step_loss = 0.0
    progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{config.training.epochs} Training")

    for step, batch_data in progress_bar:
        try:
            inputs = batch_data["image"].to(config.device, non_blocking=True) # Use non_blocking for potential speedup
            labels = batch_data["mask"].to(config.device, non_blocking=True)
        except Exception as e:
             print(f"\nError loading batch data to device at step {step}: {e}")
             print(f"Batch keys: {batch_data.keys()}")
             # Optionally check shapes/types here if error persists
             continue # Skip this batch

        optimizer.zero_grad() # Use set_to_none=True for potential slight speedup if optimizer supports it

        use_amp = config.training.get('mixed_precision', False)
        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss # Accumulate loss directly
        step_loss += batch_loss

        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({"loss": f"{batch_loss:.4f}", "lr": f"{current_lr:.6f}"})
        if (step + 1) % config.training.get('log_interval', 50) == 0:
            avg_step_loss = step_loss / config.training.get('log_interval', 50)
            writer.add_scalar("Loss/train_step", avg_step_loss, global_step)
            writer.add_scalar("LR/train", current_lr, global_step)
            step_loss = 0.0 # Reset step loss accumulator

        global_step += 1
        # No need to explicitly delete batch variables if loop continues, Python GC handles it.
        # del inputs, labels, outputs, loss, batch_data
        # torch.cuda.empty_cache() # Avoid calling this every step, can slow things down

    # Calculate average loss using total accumulated loss and dataset length
    avg_epoch_loss = epoch_loss / len(loader) # Loss per batch average
    writer.add_scalar("Loss/train_epoch", avg_epoch_loss, epoch)
    print(f"Epoch {epoch+1} Train Avg Loss: {avg_epoch_loss:.4f}")

    # Step epoch-based schedulers
    if scheduler and hasattr(scheduler, 'step'): # Check if scheduler exists and has step method
        # Some schedulers step based on epoch, some based on validation metric
        # Check scheduler type if needed
        if isinstance(scheduler, (lr_scheduler.CosineAnnealingLR, lr_scheduler.CosineAnnealingWarmRestarts)):
             scheduler.step()
        elif isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            pass # This scheduler steps in the validation phase based on metric
        else:
             # Assume other schedulers might be epoch based? Or handle specific cases
             scheduler.step()

    torch.cuda.empty_cache() # Clear cache at end of epoch
    gc.collect()
    # --- End of run_train_epoch ---
    return avg_epoch_loss, global_step


def run_validation_epoch(model, loader, metrics_dict, post_transforms, config, epoch, writer):
    # --- Start of run_validation_epoch ---
    model.eval()
    progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{config.training.epochs} Validation")
    # Reset metrics
    for metric in metrics_dict.values():
        metric.reset()

    use_amp = config.training.get('val_amp', False)
    roi_size = config.training.get('val_roi_size', (160, 160, 80))
    sw_batch_size = config.training.get('val_sw_batch_size', 4)
    overlap = config.training.get('val_overlap', 0.5)
    mode = config.training.get('val_mode', 'gaussian')
    run_tta = config.training.get('run_tta_val', False)

    with torch.no_grad():
        for step, batch_data in progress_bar:
            try:
                val_inputs = batch_data["image"].to(config.device, non_blocking=True)
                val_labels = batch_data["mask"].to(config.device, non_blocking=True)
            except Exception as e:
                print(f"\nError loading batch data to device during validation step {step}: {e}")
                continue # Skip batch

            # Sliding window inference
            with autocast(enabled=use_amp):
                 val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model, overlap=overlap, mode=mode)

                 # Optional TTA
                 if run_tta:
                     # --- TTA Logic (simplified, add more flips if needed) ---
                     tta_outputs = val_outputs.clone()
                     tta_count = 1
                     flip_dims_list = [[2], [3], [4], [2,3], [2,4], [3,4], [2,3,4]] # Example 3D flips
                     for flip_dims in flip_dims_list:
                         try:
                             flip_inputs = torch.flip(val_inputs, dims=flip_dims)
                             flip_outputs = sliding_window_inference(flip_inputs, roi_size, sw_batch_size, model, overlap=overlap, mode=mode)
                             tta_outputs += torch.flip(flip_outputs, dims=flip_dims)
                             tta_count += 1
                         except Exception as e:
                             print(f"TTA failed for flip {flip_dims}: {e}")
                     val_outputs = tta_outputs / tta_count
                     # --- End TTA Logic ---

            # Apply post-processing
            # Detach outputs before sending to post_transforms if they require CPU/numpy
            post_input_list = [{"pred": v_o.detach(), "mask": v_l.detach()} for v_o, v_l in zip(val_outputs, val_labels)]
            processed_outputs = [post_transforms(item) for item in post_input_list]

            # Extract processed predictions and labels for metric calculation
            # Ensure they are tensors on the correct device for MONAI metrics if needed
            try:
                final_preds = torch.stack([item['pred'] for item in processed_outputs]).to(config.device)
                final_labels = torch.stack([item['mask'] for item in processed_outputs]).to(config.device)
            except Exception as e:
                 print(f"\nError stacking/moving processed outputs at validation step {step}: {e}")
                 continue # Skip batch

            # Update metrics
            for metric in metrics_dict.values():
                metric(y_pred=final_preds, y=final_labels)

            # No explicit deletes needed here either

    # Aggregate metrics
    aggregated_metrics = {}
    print(f"--- Validation Results Epoch {epoch+1} ---")
    for name, metric in metrics_dict.items():
        try:
            score = metric.aggregate().item()
        except Exception as e:
             print(f"Could not aggregate metric {name}: {e}")
             score = 0.0 # Assign default value on error
        aggregated_metrics[name] = score
        writer.add_scalar(f"ValMetric/{name}", score, epoch)
        print(f"  {name}: {score:.4f}")
        metric.reset()

    # Calculate combined score
    score_weights = config.evaluation.get('score_weights', {'dice': 0.4, 'hausdorff': 0.6})
    dice_score = aggregated_metrics.get('DiceMetric', 0.0)
    hd_score = aggregated_metrics.get('HausdorffScore', 0.0)
    # Check if both scores were calculated
    if 'DiceMetric' in aggregated_metrics and 'HausdorffScore' in aggregated_metrics:
        combined_score = dice_score * score_weights.get('dice', 0.4) + \
                         hd_score * score_weights.get('hausdorff', 0.6)
    else:
         combined_score = 0.0 # Or handle based on available metrics
         print("Warning: Could not calculate combined score due to missing metrics.")

    aggregated_metrics['combined_score'] = combined_score
    writer.add_scalar("ValMetric/combined_score", combined_score, epoch)
    print(f"  Combined Score: {combined_score:.4f} (Dice={dice_score:.4f}, HD_Score={hd_score:.4f})")
    print("------------------------------------")

    torch.cuda.empty_cache()
    gc.collect()
    # --- End of run_validation_epoch ---
    return aggregated_metrics


def main(config_path):
    # Load Configuration
    config = load_config(config_path)
    print("Configuration loaded.")
    # print(config) # Optionally print full config for verification

    # Set Seed
    if config.get("seed", -1) >= 0:
        print(f"Setting deterministic seed: {config.seed}")
        set_determinism(config.seed)
    else:
        print("No seed set, results may vary.")


    # --- Directory Setup (Revised) ---
    # Format paths that require the fold number
    # Assume config paths from load_config are absolute, potentially with placeholders
    try:
        # Format checkpoint directory path
        checkpoint_dir_template = config.training.checkpoint_dir
        checkpoint_dir = checkpoint_dir_template.format(fold=config.fold)

        # Format log directory path
        log_dir_template = config.training.log_dir
        log_dir = log_dir_template.format(fold=config.fold)

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        print(f"Using checkpoint directory: {checkpoint_dir}")
        print(f"Using log directory: {log_dir}")

        # Save config to the specific fold's log directory (or checkpoint dir)
        config_save_path = os.path.join(log_dir, 'used_config.yaml')
        with open(config_save_path, 'w') as f:
            # Use to_dict() to save the Box content cleanly
            yaml.dump(config.to_dict(), f, default_flow_style=False)
        print(f"Saved used configuration to: {config_save_path}")

    except Exception as e:
         print(f"Error setting up directories or saving config: {e}")
         # Decide if execution should stop
         return


    # Create Datasets and DataLoaders
    try:
        print("Creating datasets...")
        train_ds, val_ds = get_datasets(config) # get_datasets should handle fold internally
        print("Creating dataloaders...")
        train_loader, val_loader = get_dataloaders(train_ds, val_ds, config)
    except Exception as e:
        print(f"Error creating datasets/dataloaders: {e}")
        return

    # Create Model
    print("Creating model...")
    model = get_model(config).to(config.device)

    # Create Loss Function
    loss_fn = get_loss_function(config)

    # Create Optimizer and Scheduler
    optimizer_params = config.training.optimizer.get('params', {'lr': 1e-4})
    # Ensure optimizer name is valid before getattr
    optimizer_name = config.training.optimizer.get('name', 'Adam')
    if not hasattr(torch.optim, optimizer_name):
         print(f"Error: Optimizer '{optimizer_name}' not found in torch.optim")
         return
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **optimizer_params)

    scheduler = None # Initialize scheduler to None
    if config.training.get('lr_scheduler'): # Check if scheduler config exists
        scheduler_name = config.training.lr_scheduler.get('name')
        scheduler_params = config.training.lr_scheduler.get('params', {})
        if scheduler_name and hasattr(lr_scheduler, scheduler_name):
            # Adjust T_max/T_0 for epoch-based schedulers based on num steps per epoch
            if scheduler_name in ["CosineAnnealingLR", "CosineAnnealingWarmRestarts"]:
                try:
                    steps_per_epoch = len(train_loader)
                    if 'T_max' in scheduler_params: # For CosineAnnealingLR
                        # Calculate total steps if T_max is defined in epochs
                        if isinstance(scheduler_params['T_max'], (int, float)) and scheduler_params['T_max'] <= config.training.epochs:
                             scheduler_params['T_max'] = scheduler_params['T_max'] * steps_per_epoch
                        # Otherwise assume T_max is already in steps or handled by MONAI scheduler
                    if 'T_0' in scheduler_params: # For CosineAnnealingWarmRestarts
                        if isinstance(scheduler_params['T_0'], (int, float)):
                             scheduler_params['T_0'] = scheduler_params['T_0'] * steps_per_epoch
                    print(f"Adjusted Scheduler Params (if applicable): {scheduler_params}")
                except Exception as e:
                     print(f"Warning: Could not adjust scheduler params automatically. Using raw params. Error: {e}")

            scheduler = getattr(lr_scheduler, scheduler_name)(optimizer, **scheduler_params)
            print(f"Using LR Scheduler: {scheduler_name}")
        elif scheduler_name:
             print(f"Warning: LR Scheduler '{scheduler_name}' not found in torch.optim.lr_scheduler. No scheduler used.")
    else:
        print("No LR Scheduler configured.")


    # AMP Scaler
    use_amp = config.training.get('mixed_precision', False)
    scaler = GradScaler(enabled=use_amp)
    print(f"Using Mixed Precision (AMP): {use_amp}")

    # Metrics and Post-processing
    metrics_dict = get_metric_functions(config)
    post_transforms = get_postprocessing_transforms(config) # For validation output processing

    # Tensorboard Writer
    writer = SummaryWriter(log_dir=log_dir)

    # Training Loop Variables
    start_epoch = 0
    best_metric_config = config.training.get('save_best_metric', 'combined_score')
    greater_is_better = config.training.get('greater_is_better', True)
    best_metric_score = -float('inf') if greater_is_better else float('inf') # Initialize correctly
    best_metric_epoch = -1
    global_step = 0

    # Load Checkpoint if specified
    load_ckpt_path = config.training.get('load_checkpoint', None)
    # Ensure load_ckpt_path is resolved correctly (should be absolute from load_config)
    if load_ckpt_path and isinstance(load_ckpt_path, str): # Check if path is valid string
        if os.path.exists(load_ckpt_path):
            print(f"Loading checkpoint: {load_ckpt_path}")
            try:
                checkpoint = torch.load(load_ckpt_path, map_location=config.device)
                model.load_state_dict(checkpoint['model'])
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                else:
                    print("Warning: Optimizer state not found in checkpoint.")

                if 'scheduler' in checkpoint and scheduler:
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler'])
                    except Exception as e:
                         print(f"Warning: Could not load scheduler state: {e}")

                if 'scaler' in checkpoint and scaler and hasattr(scaler, 'load_state_dict'):
                    try:
                        scaler.load_state_dict(checkpoint['scaler'])
                    except Exception as e:
                         print(f"Warning: Could not load AMP scaler state: {e}")

                start_epoch = checkpoint.get('epoch', -1) + 1
                # Optional: Load best metric score from checkpoint if saved
                if 'best_metric_score' in checkpoint:
                    # Make sure loaded score is comparable (float)
                    try:
                        best_metric_score = float(checkpoint['best_metric_score'])
                        print(f"Loaded best metric score from checkpoint: {best_metric_score:.4f}")
                    except:
                         print("Warning: Could not parse best_metric_score from checkpoint.")
                else: # If not saved, try to re-validate to get starting score? Or assume initial value.
                     print("Warning: best_metric_score not found in checkpoint. Initializing based on 'greater_is_better'.")
                     best_metric_score = -float('inf') if greater_is_better else float('inf')

                print(f"Resuming training from epoch {start_epoch}")
            except Exception as e:
                 print(f"Error loading checkpoint: {e}. Starting training from scratch.")
                 start_epoch = 0
        else:
            print(f"Warning: Checkpoint path specified but not found: {load_ckpt_path}. Starting training from scratch.")
            start_epoch = 0


    # --- Training & Validation Loop ---
    start_time = time.time()
    for epoch in range(start_epoch, config.training.epochs):

        # Train one epoch
        _, global_step = run_train_epoch(model, train_loader, optimizer, scheduler, loss_fn, scaler, config, epoch, writer, global_step)

        # Run validation periodically
        run_val_now = False
        if config.training.get('run_validation', True):
             if (epoch + 1) % config.training.validation_interval == 0:
                 run_val_now = True
             # Optionally run validation on first/last epoch regardless of interval
             # if epoch == start_epoch or epoch == config.training.epochs - 1:
             #     run_val_now = True

        if run_val_now:
            val_metrics = run_validation_epoch(model, val_loader, metrics_dict, post_transforms, config, epoch, writer)
            current_metric_score = val_metrics.get(best_metric_config, None)

            # Step scheduler if it depends on validation metric (e.g., ReduceLROnPlateau)
            if scheduler and isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                 if current_metric_score is not None:
                      scheduler.step(current_metric_score)
                 else:
                      print(f"Warning: Cannot step ReduceLROnPlateau scheduler, metric '{best_metric_config}' not found.")

            if current_metric_score is not None:
                # Save best model checkpoint
                is_better = (current_metric_score > best_metric_score) if greater_is_better else (current_metric_score < best_metric_score)
                if is_better:
                    print(f"Validation metric '{best_metric_config}' improved ({best_metric_score:.4f} -> {current_metric_score:.4f}).")
                    best_metric_score = current_metric_score
                    best_metric_epoch = epoch + 1
                    checkpoint = create_checkpoint(model, optimizer, epoch, scheduler, scaler)
                    checkpoint['best_metric_score'] = best_metric_score # Save score in checkpoint
                    best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
                    torch.save(checkpoint, best_ckpt_path)
                    print(f"Saved new best model to {best_ckpt_path} at epoch {best_metric_epoch}")
                else:
                     print(f"Validation metric '{best_metric_config}' did not improve ({current_metric_score:.4f} vs best {best_metric_score:.4f}).")

        # Save latest checkpoint periodically
        if (epoch + 1) % config.training.get('save_interval', 50) == 0:
             checkpoint = create_checkpoint(model, optimizer, epoch, scheduler, scaler)
             # Include best score found so far in latest checkpoint for easier resuming
             checkpoint['best_metric_score'] = best_metric_score
             latest_ckpt_path = os.path.join(checkpoint_dir, "latest_model.pth")
             torch.save(checkpoint, latest_ckpt_path)
             print(f"Saved latest checkpoint to {latest_ckpt_path} at epoch {epoch+1}")

    # --- End of Training ---
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/3600:.2f} hours.")
    print(f"Best validation metric ({best_metric_config}): {best_metric_score:.4f} achieved at epoch {best_metric_epoch}")

    # Optionally copy best model weights with score in filename
    if config.training.get('save_best_with_score_filename', True) and best_metric_epoch != -1:
         best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
         if os.path.exists(best_ckpt_path):
             final_filename = os.path.join(checkpoint_dir, f"best_model_epoch{best_metric_epoch}_{best_metric_score:.4f}.pth")
             shutil.copyfile(best_ckpt_path, final_filename)
             print(f"Copied best model weights to: {final_filename}")
         else:
              print("Warning: Best checkpoint file not found at end of training to copy with score.")


    # Optionally load best model weights into model object at the very end
    if config.training.get('load_best_at_end', True) and best_metric_epoch != -1:
         best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
         if os.path.exists(best_ckpt_path):
             print(f"Loading best model weights from epoch {best_metric_epoch} into final model state.")
             checkpoint = torch.load(best_ckpt_path, map_location=config.device)
             model.load_state_dict(checkpoint['model'])
         else:
              print("Warning: Best checkpoint file not found at end of training to load.")

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a MONAI model for segmentation.")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    parser.add_argument("-f", "--fold", type=int, help="Override fold number in config (optional).")

    args = parser.parse_args()

    # Load config initially to potentially override fold
    temp_config = load_config(args.config)
    if args.fold is not None:
         print(f"Overriding fold in config from {temp_config.fold} to {args.fold}")
         temp_config.fold = args.fold
         # Save the modified config to a temporary file to pass its path
         # Alternatively, pass the modified config object directly if main supports it
         temp_config_path = "temp_run_config.yaml"
         with open(temp_config_path, 'w') as f:
              yaml.dump(temp_config.to_dict(), f)
         main(temp_config_path)
         try:
             os.remove(temp_config_path) # Clean up temp file
         except OSError as e:
              print(f"Error removing temporary config file: {e}")
    else:
        # Run with the original config path if fold is not overridden
        main(args.config)