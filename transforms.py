# transforms.py
import monai.transforms as mt
from monai.transforms import Compose, OneOf
import torch
import importlib
import random
import numpy as np
from utils import DebugShapeD # Import custom transforms

# Dynamically get MONAI transforms, handling OneOf separately
def _get_transform(transform_config: dict, is_oneof_member=False):
    """Dynamically creates a MONAI transform instance from config."""
    name = transform_config.get('name')
    params = transform_config.get('params', {})
    prob = transform_config.get('prob', 1.0) # Get probability if specified

    if not name:
        raise ValueError(f"Transform config missing 'name': {transform_config}")

    # Handle special cases like OneOf or custom transforms
    if name == "OneOf":
        if 'transforms' not in transform_config or not isinstance(transform_config['transforms'], list):
             raise ValueError("OneOf transform requires a 'transforms' list in config.")
        if 'weights' in transform_config and len(transform_config['weights']) != len(transform_config['transforms']):
            raise ValueError("Length of 'weights' must match length of 'transforms' for OneOf.")

        sub_transforms = [_get_transform(t_conf, is_oneof_member=True) for t_conf in transform_config['transforms']]
        weights = transform_config.get('weights', None)
        # MONAI OneOf doesn't directly take a global prob, apply probability outside if needed
        return OneOf(transforms=sub_transforms, weights=weights)

    elif name == "DebugShapeD":
        return DebugShapeD(**params)
    # Add other custom transforms here...

    try:
        transform_class = getattr(mt, name)
        instance = transform_class(**params)

        # Apply probability wrapper if needed and not already handled (like inside OneOf)
        # Rand* transforms handle prob internally, others might need RandApplied
        if not name.startswith("Rand") and prob < 1.0 and not is_oneof_member:
             print(f"Applying RandApplied wrapper for {name} with prob={prob}")
             return mt.RandApplied(transform=instance, prob=prob, keys=params.get('keys'))

        return instance

    except AttributeError:
        raise ValueError(f"Transform '{name}' not found in monai.transforms.")
    except Exception as e:
        raise ValueError(f"Error initializing transform '{name}' with params {params}: {e}")

def create_transform_pipeline(config_section: list | dict | None, default_keys: list = ["image", "mask"]) -> list:
    """Creates a list of MONAI transforms from a config section."""
    transform_list = []
    if not config_section:
        return transform_list

    # Handle different structures (list of transforms, or dict with sub-sections)
    if isinstance(config_section, list):
        items_to_process = config_section
    elif isinstance(config_section, dict):
        # Process items in the order they appear in the YAML (requires Python 3.7+)
        items_to_process = [{"name": k, **v} if isinstance(v, dict) and 'name' not in v else v for k, v in config_section.items()]
         # Fallback if names aren't keys: iterate through values if they are lists/dicts
        if not all(isinstance(item, dict) and 'name' in item for item in items_to_process):
            items_to_process = []
            for key, value in config_section.items():
                 if isinstance(value, list): # If a key contains a list of transforms
                     items_to_process.extend(value)
                 elif isinstance(value, dict) and 'name' in value: # If a key contains a single transform dict
                     items_to_process.append(value)
    else:
        return transform_list

    for t_config in items_to_process:
        if not isinstance(t_config, dict) or 'name' not in t_config:
             print(f"Skipping invalid transform config item: {t_config}")
             continue

        # Inject default keys if not specified
        if 'keys' not in t_config.get('params', {}):
            t_config.setdefault('params', {})['keys'] = default_keys
        transform_list.append(_get_transform(t_config))

    return transform_list

def get_train_transforms(config) -> Compose:
    """Creates the MONAI Compose object for training transforms."""
    all_transforms = []
    keys = ["image", "mask"]

    # 1. Load/Base Preprocessing (if not done in preprocess_data.py)
    if not config.data.use_preprocessed_3d:
         all_transforms.extend(create_transform_pipeline(config.get_safe('preprocessing_2d.load', []), default_keys=keys))
         all_transforms.extend(create_transform_pipeline(config.get_safe('preprocessing_2d.channel', []), default_keys=keys))
         all_transforms.extend(create_transform_pipeline(config.get_safe('preprocessing_2d.intensity', []), default_keys=keys))
         all_transforms.append(DebugShapeD(keys=keys, label="After Load/Intensity (2D)"))
    else:
        # Load 3D NIfTI
         all_transforms.extend(create_transform_pipeline(config.get_safe('preprocessing_3d.load', []), default_keys=keys))
         all_transforms.extend(create_transform_pipeline(config.get_safe('preprocessing_3d.channel', []), default_keys=keys))
         all_transforms.extend(create_transform_pipeline(config.get_safe('preprocessing_3d.intensity', []), default_keys=keys))
         all_transforms.append(DebugShapeD(keys=keys, label="After Load/Intensity (3D)"))


    # 2. Augmentations
    all_transforms.extend(create_transform_pipeline(config.get_safe('augmentation.spatial', []), default_keys=keys))
    all_transforms.extend(create_transform_pipeline(config.get_safe('augmentation.intensity', []), default_keys=keys))
    all_transforms.extend(create_transform_pipeline(config.get_safe('augmentation.other', []), default_keys=keys))
    all_transforms.append(DebugShapeD(keys=keys, label="After Augmentation"))

    # 3. Ensure Type and Device (should be last before DataLoader)
    all_transforms.append(mt.EnsureTyped(keys=keys, dtype=torch.float32)) # Ensure float32
    # all_transforms.append(mt.ToDeviced(keys=keys, device=config.device)) # Move to device in DataLoader/train loop instead

    return Compose(all_transforms)

def get_val_transforms(config) -> Compose:
    """Creates the MONAI Compose object for validation transforms."""
    all_transforms = []
    keys = ["image", "mask"]

    # 1. Load/Base Preprocessing (match training setup)
    if not config.data.use_preprocessed_3d:
        # Add 2D loading/preprocessing here if needed
        pass
    else:
        # Load 3D NIfTI
         all_transforms.extend(create_transform_pipeline(config.get_safe('preprocessing_3d.load', []), default_keys=keys))
         all_transforms.extend(create_transform_pipeline(config.get_safe('preprocessing_3d.channel', []), default_keys=keys))
         # Apply *validation* intensity transforms (often same as training base)
         all_transforms.extend(create_transform_pipeline(config.get_safe('validation_transforms.intensity', []), default_keys=keys))

    # 2. Ensure Type
    all_transforms.append(mt.EnsureTyped(keys=keys, dtype=torch.float32))
    # all_transforms.append(mt.ToDeviced(keys=keys, device=config.device))

    # Add debug transform if needed
    all_transforms.append(DebugShapeD(keys=keys, label="Validation Transforms Output", enabled=False))

    return Compose(all_transforms)

def get_inference_transforms(config) -> Compose:
    """Creates the MONAI Compose object for inference (minimal transforms)."""
    # Similar to validation, but potentially only loads image
    all_transforms = []
    keys = ["image"]

    # Load 3D NIfTI Image
    all_transforms.extend(create_transform_pipeline(config.get_safe('preprocessing_3d.load', []), default_keys=keys))
    all_transforms.extend(create_transform_pipeline(config.get_safe('preprocessing_3d.channel', []), default_keys=keys))
    # Apply validation/inference intensity transforms
    all_transforms.extend(create_transform_pipeline(config.get_safe('validation_transforms.intensity', []), default_keys=keys))
    all_transforms.append(mt.EnsureTyped(keys=keys, dtype=torch.float32))
    # all_transforms.append(mt.ToDeviced(keys=keys, device=config.device))
    return Compose(all_transforms)


def get_postprocessing_transforms(config) -> Compose:
    """Creates MONAI Compose object for post-processing predictions."""
    all_transforms = []
    pred_key = "pred" # Standard key for predictions

    # 1. Activation (applied first)
    activation_type = config.postprocessing.get('activation', 'sigmoid')
    if activation_type == 'sigmoid':
        all_transforms.append(mt.Activationsd(keys=pred_key, sigmoid=True))
    elif activation_type == 'softmax':
         all_transforms.append(mt.Activationsd(keys=pred_key, softmax=True))
    else:
        print(f"Warning: Unknown activation type '{activation_type}' in postprocessing. Skipping activation.")

    # 2. Discretization/Thresholding
    threshold = config.postprocessing.get('threshold', 0.5)
    # AsDiscrete needs input C=num_classes, H, W, D
    all_transforms.append(mt.AsDiscreted(keys=pred_key, threshold=threshold))

    # 3. Further mask transforms (e.g., connected components)
    mask_transforms_config = config.postprocessing.get('mask_transforms', [])
    all_transforms.extend(create_transform_pipeline(mask_transforms_config, default_keys=[pred_key]))

    return Compose(all_transforms)