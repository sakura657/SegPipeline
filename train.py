# train.py
import argparse
import os
import shutil
import time
import gc

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
from utils import load_config, create_checkpoint, DebugShapeD
from dataset import get_datasets, get_dataloaders
from model_factory import get_model
from losses import get_loss_function
from metrics import get_metric_functions
from transforms import get_postprocessing_transforms


def run_train_epoch(model, loader, optimizer, scheduler, loss_fn, scaler, config, epoch, writer, global_step):
    model.train()
    epoch_loss = 0.0
    step_loss = 0.0
    progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{config.training.epochs} Training")

    for step, batch_data in progress_bar:
        inputs = batch_data["image"].to(config.device)
        labels = batch_data["mask"].to(config.device)

        optimizer.zero_grad()

        use_amp = config.training.get('mixed_precision', False)
        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Step the scheduler (check if it's step-based or epoch-based)
        # Assuming CosineAnnealingLR or CosineAnnealingWarmRestarts (epoch based, step in main loop)
        # If using a step-based scheduler, call scheduler.step() here

        batch_loss = loss.item()
        epoch_loss += batch_loss * inputs.size(0) # Accumulate loss weighted by batch size
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
        del inputs, labels, outputs, loss, batch_data
        torch.cuda.empty_cache()
        gc.collect()


    avg_epoch_loss = epoch_loss / len(loader.dataset)
    writer.add_scalar("Loss/train_epoch", avg_epoch_loss, epoch)
    print(f"Epoch {epoch+1} Train Avg Loss: {avg_epoch_loss:.4f}")

    # Step epoch-based schedulers
    if isinstance(scheduler, (lr_scheduler.CosineAnnealingLR, lr_scheduler.CosineAnnealingWarmRestarts)):
        scheduler.step()

    return avg_epoch_loss, global_step


def run_validation_epoch(model, loader, metrics_dict, post_transforms, config, epoch, writer):
    model.eval()
    progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{config.training.epochs} Validation")
    # Reset metrics
    for metric in metrics_dict.values():
        metric.reset()

    use_amp = config.training.get('val_amp', False) # Use separate flag for validation AMP
    roi_size = config.training.get('val_roi_size', (160, 160, 80))
    sw_batch_size = config.training.get('val_sw_batch_size', 4)
    overlap = config.training.get('val_overlap', 0.5)
    mode = config.training.get('val_mode', 'gaussian')
    run_tta = config.training.get('run_tta_val', False)

    with torch.no_grad():
        for step, batch_data in progress_bar:
            val_inputs = batch_data["image"].to(config.device)
            val_labels = batch_data["mask"].to(config.device) # Shape (B, C, H, W, D)

            # Sliding window inference
            with autocast(enabled=use_amp):
                 val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model, overlap=overlap, mode=mode)

                 # Optional TTA
                 if run_tta:
                     tta_outputs = val_outputs.clone()
                     tta_count = 1
                     # Example: Flip Dim 2 (Height)
                     flip_dims = [2]
                     flip_inputs = torch.flip(val_inputs, dims=flip_dims)
                     flip_outputs = sliding_window_inference(flip_inputs, roi_size, sw_batch_size, model, overlap=overlap, mode=mode)
                     tta_outputs += torch.flip(flip_outputs, dims=flip_dims)
                     tta_count +=1
                     # Example: Flip Dim 3 (Width)
                     flip_dims = [3]
                     flip_inputs = torch.flip(val_inputs, dims=flip_dims)
                     flip_outputs = sliding_window_inference(flip_inputs, roi_size, sw_batch_size, model, overlap=overlap, mode=mode)
                     tta_outputs += torch.flip(flip_outputs, dims=flip_dims)
                     tta_count +=1
                     # Average TTA results
                     val_outputs = tta_outputs / tta_count
                     del flip_inputs, flip_outputs, tta_outputs


            # Apply post-processing (activation, thresholding, etc.)
            # Assume val_outputs shape (B, C, H, W, D)
            # Post transforms expect dictionary input
            post_input_dict = [{"pred": v_o, "mask": v_l} for v_o, v_l in zip(val_outputs, val_labels)] # Create list of dicts for batch
            processed_outputs = [post_transforms(item) for item in post_input_dict]

            # Extract processed predictions and labels for metric calculation
            # Output of post_transforms should still have 'pred' and 'mask' keys
            final_preds = torch.stack([item['pred'] for item in processed_outputs]) # Shape (B, C, H, W, D), binary {0,1}
            final_labels = torch.stack([item['mask'] for item in processed_outputs])# Shape (B, C, H, W, D)

            # Update metrics
            for metric in metrics_dict.values():
                metric(y_pred=final_preds, y=final_labels)

            del val_inputs, val_labels, val_outputs, post_input_dict, processed_outputs, final_preds, final_labels, batch_data
            torch.cuda.empty_cache()


    # Aggregate metrics
    aggregated_metrics = {}
    print(f"--- Validation Results Epoch {epoch+1} ---")
    for name, metric in metrics_dict.items():
        score = metric.aggregate().item()
        aggregated_metrics[name] = score
        writer.add_scalar(f"ValMetric/{name}", score, epoch)
        print(f"  {name}: {score:.4f}")
        metric.reset() # Reset again for safety

    # Calculate combined score based on config weights
    score_weights = config.evaluation.get('score_weights', {'dice': 0.4, 'hausdorff': 0.6})
    dice_score = aggregated_metrics.get('DiceMetric', 0.0)
    hd_score = aggregated_metrics.get('HausdorffScore', 0.0) # Use custom score
    combined_score = dice_score * score_weights.get('dice', 0.4) + \
                     hd_score * score_weights.get('hausdorff', 0.6)

    aggregated_metrics['combined_score'] = combined_score
    writer.add_scalar("ValMetric/combined_score", combined_score, epoch)
    print(f"  Combined Score: {combined_score:.4f} (Dice={dice_score:.4f}, HD_Score={hd_score:.4f})")
    print("------------------------------------")

    return aggregated_metrics


def main(config_path):
    # Load Configuration
    config = load_config(config_path)
    print("Configuration loaded:")
    print(config)

    # Set Seed
    if config.get("seed", -1) >= 0:
        print(f"Setting deterministic seed: {config.seed}")
        set_determinism(config.seed)

    # Directories
    fold_output_dir = os.path.join(config.base_dir, config.output_dir, f"fold_{config.fold}")
    checkpoint_dir = os.path.join(config.base_dir, config.training.checkpoint_dir)
    log_dir = os.path.join(config.base_dir, config.training.log_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Save config to output directory
    with open(os.path.join(fold_output_dir, 'used_config.yaml'), 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    # Create Datasets and DataLoaders
    print("Creating datasets...")
    train_ds, val_ds = get_datasets(config)
    print("Creating dataloaders...")
    train_loader, val_loader = get_dataloaders(train_ds, val_ds, config)

    # Create Model
    print("Creating model...")
    model = get_model(config).to(config.device)

    # Create Loss Function
    loss_fn = get_loss_function(config)

    # Create Optimizer and Scheduler
    optimizer_params = config.training.optimizer.get('params', {'lr': 1e-4})
    optimizer = getattr(torch.optim, config.training.optimizer.get('name', 'Adam'))(model.parameters(), **optimizer_params)

    scheduler_name = config.training.lr_scheduler.get('name', 'CosineAnnealingLR')
    scheduler_params = config.training.lr_scheduler.get('params', {})
    # Adjust T_max/T_0 if they are epoch-based
    if scheduler_name in ["CosineAnnealingLR", "CosineAnnealingWarmRestarts"]:
         # Calculate steps per epoch if needed, or use epochs directly
         steps_per_epoch = len(train_loader) # Or len(train_ds) // config.training.batch_size
         if 'T_max' in scheduler_params:
              scheduler_params['T_max'] = config.training.epochs * steps_per_epoch
         if 'T_0' in scheduler_params: # For WarmRestarts
              scheduler_params['T_0'] = scheduler_params['T_0'] * steps_per_epoch
         print(f"Adjusted Scheduler Params: {scheduler_params}")

    scheduler = getattr(lr_scheduler, scheduler_name)(optimizer, **scheduler_params)


    # AMP Scaler
    use_amp = config.training.get('mixed_precision', False)
    scaler = GradScaler(enabled=use_amp)

    # Metrics and Post-processing
    metrics_dict = get_metric_functions(config)
    post_transforms = get_postprocessing_transforms(config) # For validation output processing

    # Tensorboard Writer
    writer = SummaryWriter(log_dir=log_dir)

    # Training Loop Variables
    start_epoch = 0
    best_metric_score = -1.0 if config.training.get('greater_is_better', True) else float('inf')
    best_metric_epoch = -1
    global_step = 0

    # Load Checkpoint if specified
    load_ckpt_path = config.training.get('load_checkpoint', None)
    if load_ckpt_path:
        load_ckpt_path = os.path.join(config.base_dir, load_ckpt_path) # Make path absolute if relative
        if os.path.exists(load_ckpt_path):
            print(f"Loading checkpoint: {load_ckpt_path}")
            checkpoint = torch.load(load_ckpt_path, map_location=config.device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint and scheduler:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if 'scaler' in checkpoint and scaler and 'state_dict' in scaler.__dir__():
                 scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint['epoch'] + 1
            # Optional: Load best metric score from checkpoint if saved
            if 'best_metric_score' in checkpoint:
                best_metric_score = checkpoint['best_metric_score']
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print(f"Warning: Checkpoint path specified but not found: {load_ckpt_path}")


    # --- Training & Validation Loop ---
    start_time = time.time()
    for epoch in range(start_epoch, config.training.epochs):

        # Train one epoch
        _, global_step = run_train_epoch(model, train_loader, optimizer, scheduler, loss_fn, scaler, config, epoch, writer, global_step)

        # Run validation periodically
        if config.training.get('run_validation', True) and (epoch + 1) % config.training.validation_interval == 0:
            val_metrics = run_validation_epoch(model, val_loader, metrics_dict, post_transforms, config, epoch, writer)
            current_metric_score = val_metrics.get(config.training.get('save_best_metric', 'combined_score'), None)

            if current_metric_score is not None:
                # Save best model checkpoint
                is_better = (current_metric_score > best_metric_score) if config.training.get('greater_is_better', True) else (current_metric_score < best_metric_score)
                if is_better:
                    best_metric_score = current_metric_score
                    best_metric_epoch = epoch + 1
                    checkpoint = create_checkpoint(model, optimizer, epoch, scheduler, scaler)
                    checkpoint['best_metric_score'] = best_metric_score # Save score in checkpoint
                    best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
                    torch.save(checkpoint, best_ckpt_path)
                    print(f"Saved new best model based on '{config.training.save_best_metric}' score: {best_metric_score:.4f} at epoch {best_metric_epoch}")

        # Save latest checkpoint periodically
        if (epoch + 1) % config.training.get('save_interval', 50) == 0:
             checkpoint = create_checkpoint(model, optimizer, epoch, scheduler, scaler)
             latest_ckpt_path = os.path.join(checkpoint_dir, "latest_model.pth")
             torch.save(checkpoint, latest_ckpt_path)
             print(f"Saved latest checkpoint at epoch {epoch+1}")

    # --- End of Training ---
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/3600:.2f} hours.")
    print(f"Best validation metric ({config.training.save_best_metric}): {best_metric_score:.4f} at epoch {best_metric_epoch}")

    # Optionally load best model weights at the end
    if config.training.get('load_best_at_end', True) and best_metric_epoch != -1:
         best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
         if os.path.exists(best_ckpt_path):
             print(f"Loading best model weights from epoch {best_metric_epoch}")
             checkpoint = torch.load(best_ckpt_path, map_location=config.device)
             model.load_state_dict(checkpoint['model'])
         else:
              print("Warning: Best checkpoint file not found at end of training.")

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a MONAI model for segmentation.")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    # Add other command-line arguments if needed (e.g., overriding fold)
    # parser.add_argument("-f", "--fold", type=int, help="Override fold number in config.")

    args = parser.parse_args()

    # # Override config fold if provided via command line
    # if args.fold is not None:
    #     config = load_config(args.config)
    #     config.fold = args.fold
    #     # Need to save the modified config or pass it directly? Pass path for now.
    #     temp_config_path = "temp_run_config.yaml"
    #     with open(temp_config_path, 'w') as f:
    #          yaml.dump(config.to_dict(), f)
    #     main(temp_config_path)
    #     os.remove(temp_config_path) # Clean up temp file
    # else:
    main(args.config)