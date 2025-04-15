# losses.py
import torch
import torch.nn as nn
import monai.losses as monai_losses
from monai.utils import LossReduction

class DiceBceMultilabelLoss(nn.Module):
    """Combines Dice and BCE for multi-label tasks."""
    # Adapted from uploaded utils.py
    def __init__(self, w_dice=0.5, w_bce=0.5, sigmoid=True, reduction='mean', **dice_kwargs):
        super().__init__()
        self.w_dice = w_dice
        self.w_bce = w_bce
        # Pass sigmoid=True to DiceLoss if model output is logits
        self.dice_loss = monai_losses.DiceLoss(sigmoid=sigmoid, **dice_kwargs)
        # BCEWithLogitsLoss expects logits as input
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, pred, label):
        # Ensure label is float for BCE
        label = label.float()
        # pred is assumed to be logits here
        dice_loss_val = self.dice_loss(pred, label)
        bce_loss_val = self.bce_loss(pred, label)
        # print(f"Dice: {dice_loss_val.item():.4f}, BCE: {bce_loss_val.item():.4f}") # Debug
        loss = dice_loss_val * self.w_dice + bce_loss_val * self.w_bce
        return loss


def get_loss_function(config):
    """Creates the loss function based on the configuration."""
    loss_name = config.training.loss_function.get('name', 'DiceCELoss')
    loss_params = config.training.loss_function.get('params', {})

    # Handle custom losses
    if loss_name == "DiceBceMultilabelLoss":
        return DiceBceMultilabelLoss(**loss_params)
    # Add other custom losses here...

    try:
        # Get loss from MONAI
        loss_class = getattr(monai_losses, loss_name)
        print(f"Using MONAI loss: {loss_name} with params: {loss_params}")
        return loss_class(**loss_params)
    except AttributeError:
        raise ValueError(f"Loss function '{loss_name}' not found in monai.losses or custom losses.")
    except Exception as e:
         raise ValueError(f"Error initializing loss '{loss_name}' with params {loss_params}: {e}")