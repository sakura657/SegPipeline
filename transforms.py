# transforms.py
import monai.transforms as mt
from monai.transforms import Compose, OneOf
import torch
import importlib
import random
import numpy as np
from utils import DebugShapeD # Import custom transforms
import logging # Use logging

# Dynamically get MONAI transforms, handling OneOf separately
def _get_transform(transform_config: dict, is_oneof_member=False):
    """Dynamically creates a MONAI transform instance from config."""
    # --- Start _get_transform ---
    if not isinstance(transform_config, dict):
         logging.error(f"_get_transform expected dict, got {type(transform_config)}: {transform_config}")
         raise TypeError(f"_get_transform expected dict, got {type(transform_config)}")

    name = transform_config.get('name')
    params = transform_config.get('params', {})
    prob = transform_config.get('prob', 1.0) # Get probability if specified
    keys = params.get('keys', None) # Get keys for RandApplied if needed

    if not name:
        logging.error(f"Transform config missing 'name': {transform_config}")
        raise ValueError(f"Transform config missing 'name': {transform_config}")

    logging.debug(f"Attempting to create transform: {name} with params: {params}")

    # Handle special cases like OneOf or custom transforms
    if name == "OneOf":
        if 'transforms' not in transform_config or not isinstance(transform_config['transforms'], list):
             logging.error("OneOf transform requires a 'transforms' list in config.")
             raise ValueError("OneOf transform requires a 'transforms' list in config.")
        if 'weights' in transform_config and len(transform_config['weights']) != len(transform_config['transforms']):
            logging.error("Length of 'weights' must match length of 'transforms' for OneOf.")
            raise ValueError("Length of 'weights' must match length of 'transforms' for OneOf.")

        sub_transforms = [_get_transform(t_conf, is_oneof_member=True) for t_conf in transform_config['transforms']]
        weights = transform_config.get('weights', None)
        logging.debug(f"Creating OneOf with transforms: {sub_transforms} and weights: {weights}")
        return OneOf(transforms=sub_transforms, weights=weights)

    elif name == "DebugShapeD":
         logging.debug("Creating DebugShapeD")
         return DebugShapeD(**params)
    # Add other custom transforms here...

    try:
        transform_class = getattr(mt, name)
        logging.debug(f"Found transform class: {transform_class}")
        instance = transform_class(**params)
        logging.debug(f"Instantiated transform: {name}")

        # Apply probability wrapper if needed and not already handled (like inside OneOf)
        if not name.startswith("Rand") and prob < 1.0 and not is_oneof_member:
             logging.debug(f"Applying RandApplied wrapper for {name} with prob={prob}")
             if keys is None:
                  logging.warning(f"Applying RandApplied to {name} without explicit 'keys' might be ambiguous.")
             return mt.RandApplied(transform=instance, prob=prob, keys=keys)

        return instance

    except AttributeError:
        logging.error(f"Transform '{name}' not found in monai.transforms.")
        raise ValueError(f"Transform '{name}' not found in monai.transforms.")
    except Exception as e:
        logging.error(f"Error initializing transform '{name}' with params {params}: {e}")
        raise ValueError(f"Error initializing transform '{name}' with params {params}: {e}")
    # --- End _get_transform ---


def create_transform_pipeline(config_section_name: str, config_section: list | dict | None, default_keys: list = ["image", "mask"]) -> list:
    """Creates a list of MONAI transforms from a config section."""
    # --- Start create_transform_pipeline ---
    transform_list = []
    logging.debug(f"Creating transform pipeline for section: '{config_section_name}'")
    if not config_section:
        logging.warning(f"Config section '{config_section_name}' is empty or None.")
        return transform_list

    items_to_process = []
    if isinstance(config_section, list):
        items_to_process = config_section
    elif isinstance(config_section, dict):
        items_to_process = list(config_section.values()) # Process based on values if it's a dict of transforms
    else:
         logging.warning(f"Config section '{config_section_name}' has unexpected type: {type(config_section)}. Returning empty list.")
         return transform_list

    for i, t_config in enumerate(items_to_process):
        logging.debug(f"Processing transform {i+1}/{len(items_to_process)} in '{config_section_name}'...")
        if not isinstance(t_config, dict) or 'name' not in t_config:
             logging.warning(f"Skipping invalid transform config item in '{config_section_name}': {t_config}")
             continue
        try:
            # Inject default keys if not specified
            current_params = t_config.get('params', {})
            if 'keys' not in current_params:
                current_params['keys'] = default_keys
                t_config['params'] = current_params # Update params in config dict

            transform_instance = _get_transform(t_config)
            transform_list.append(transform_instance)
            logging.debug(f"Successfully added transform: {t_config.get('name')}")
        except Exception as e:
             logging.error(f"Failed to create transform item in '{config_section_name}': {t_config}. Error: {e}", exc_info=True) # Log traceback
             # Decide whether to raise the error or just skip the problematic transform
             raise # Re-raise the exception to stop execution and see the error
             # print(f"Skipping problematic transform: {t_config.get('name')}")
             # continue

    logging.debug(f"Finished creating pipeline for section: '{config_section_name}' with {len(transform_list)} transforms.")
    # --- End create_transform_pipeline ---
    return transform_list

def get_train_transforms(config) -> Compose | None: # Added | None to annotation for clarity
    """Creates the MONAI Compose object for training transforms."""
    # --- Start get_train_transforms ---
    logging.info("Creating training transforms...")
    all_transforms = []
    keys = ["image", "mask"]

    try:
        # 1. Load/Base Preprocessing
        if not config.data.get('use_preprocessed_3d', False): # Use .get for safety
             # Add 2D preprocessing transforms here if implemented
             logging.warning("Direct 2D PNG processing not fully implemented in transforms. Expecting use_preprocessed_3d=True.")
             # Example placeholder:
             # all_transforms.extend(create_transform_pipeline('preprocessing_2d.load', config.get('preprocessing_2d', {}).get('load', []), default_keys=keys))
             # all_transforms.extend(create_transform_pipeline('preprocessing_2d.channel', config.get('preprocessing_2d', {}).get('channel', []), default_keys=keys))
             # all_transforms.extend(create_transform_pipeline('preprocessing_2d.intensity', config.get('preprocessing_2d', {}).get('intensity', []), default_keys=keys))
             pass
        else:
            # Load 3D NIfTI - use .get for robustness against missing keys
            logging.debug("Adding 3D preprocessing transforms...")
            all_transforms.extend(create_transform_pipeline('preprocessing_3d.load', config.get('preprocessing_3d', {}).get('load', []), default_keys=keys))
            all_transforms.extend(create_transform_pipeline('preprocessing_3d.channel', config.get('preprocessing_3d', {}).get('channel', []), default_keys=keys))
            all_transforms.extend(create_transform_pipeline('preprocessing_3d.intensity', config.get('preprocessing_3d', {}).get('intensity', []), default_keys=keys))
            all_transforms.append(DebugShapeD(keys=keys, label="After Load/Intensity (3D)", enabled=False)) # Keep disabled by default


        # 2. Augmentations
        logging.debug("Adding augmentation transforms...")
        all_transforms.extend(create_transform_pipeline('augmentation.spatial', config.get('augmentation', {}).get('spatial', []), default_keys=keys))
        all_transforms.extend(create_transform_pipeline('augmentation.intensity', config.get('augmentation', {}).get('intensity', []), default_keys=keys))
        all_transforms.extend(create_transform_pipeline('augmentation.other', config.get('augmentation', {}).get('other', []), default_keys=keys))
        all_transforms.append(DebugShapeD(keys=keys, label="After Augmentation", enabled=False)) # Keep disabled by default

        # 3. Ensure Type and Device (should be last before DataLoader)
        logging.debug("Adding EnsureTyped transform...")
        all_transforms.append(mt.EnsureTyped(keys=keys, dtype=torch.float32, track_meta=False)) # Ensure float32, track_meta=False can save memory

        logging.info(f"Successfully created training transform sequence with {len(all_transforms)} steps.")
        # print(f"DEBUG: Final train transforms list: {all_transforms}") # Optional: Print the final list
        return Compose(all_transforms)

    except Exception as e:
         logging.error(f"Error occurred during get_train_transforms: {e}", exc_info=True) # Log full traceback
         return None # Explicitly return None on error
    # --- End get_train_transforms ---


# --- get_val_transforms --- (Apply similar changes: .get, logging, try/except)
def get_val_transforms(config) -> Compose | None:
    """Creates the MONAI Compose object for validation transforms."""
    # --- Start get_val_transforms ---
    logging.info("Creating validation transforms...")
    all_transforms = []
    keys = ["image", "mask"]

    try:
        # 1. Load/Base Preprocessing (match training setup)
        if not config.data.get('use_preprocessed_3d', False):
            # Add 2D loading/preprocessing here if needed
            logging.warning("Direct 2D PNG processing not fully implemented in transforms. Expecting use_preprocessed_3d=True.")
            pass
        else:
            # Load 3D NIfTI
            logging.debug("Adding 3D preprocessing transforms for validation...")
            all_transforms.extend(create_transform_pipeline('preprocessing_3d.load (val)', config.get('preprocessing_3d', {}).get('load', []), default_keys=keys))
            all_transforms.extend(create_transform_pipeline('preprocessing_3d.channel (val)', config.get('preprocessing_3d', {}).get('channel', []), default_keys=keys))
            # Apply *validation* intensity transforms
            all_transforms.extend(create_transform_pipeline('validation_transforms.intensity', config.get('validation_transforms', {}).get('intensity', []), default_keys=keys))

        # 2. Ensure Type
        logging.debug("Adding EnsureTyped transform for validation...")
        all_transforms.append(mt.EnsureTyped(keys=keys, dtype=torch.float32, track_meta=False))

        all_transforms.append(DebugShapeD(keys=keys, label="Validation Transforms Output", enabled=False))

        logging.info(f"Successfully created validation transform sequence with {len(all_transforms)} steps.")
        # print(f"DEBUG: Final val transforms list: {all_transforms}")
        return Compose(all_transforms)

    except Exception as e:
         logging.error(f"Error occurred during get_val_transforms: {e}", exc_info=True)
         return None
    # --- End get_val_transforms ---


# --- get_inference_transforms --- (Apply similar changes)
def get_inference_transforms(config) -> Compose | None:
    """Creates the MONAI Compose object for inference (minimal transforms)."""
    logging.info("Creating inference transforms...")
    all_transforms = []
    keys = ["image"] # Usually only process image for inference
    try:
        # Load 3D NIfTI Image
        all_transforms.extend(create_transform_pipeline('preprocessing_3d.load (infer)', config.get('preprocessing_3d', {}).get('load', []), default_keys=keys))
        all_transforms.extend(create_transform_pipeline('preprocessing_3d.channel (infer)', config.get('preprocessing_3d', {}).get('channel', []), default_keys=keys))
        # Apply validation/inference intensity transforms
        all_transforms.extend(create_transform_pipeline('validation_transforms.intensity (infer)', config.get('validation_transforms', {}).get('intensity', []), default_keys=keys))
        all_transforms.append(mt.EnsureTyped(keys=keys, dtype=torch.float32, track_meta=False)) # track_meta often needed for Invertd if used later

        logging.info(f"Successfully created inference transform sequence with {len(all_transforms)} steps.")
        return Compose(all_transforms)
    except Exception as e:
         logging.error(f"Error occurred during get_inference_transforms: {e}", exc_info=True)
         return None


# --- get_postprocessing_transforms --- (Apply similar changes)
def get_postprocessing_transforms(config) -> Compose | None:
    """Creates MONAI Compose object for post-processing predictions."""
    logging.info("Creating postprocessing transforms...")
    all_transforms = []
    pred_key = "pred" # Standard key for predictions
    try:
        # 1. Activation (applied first)
        activation_type = config.postprocessing.get('activation', 'sigmoid')
        if activation_type == 'sigmoid':
            all_transforms.append(mt.Activationsd(keys=pred_key, sigmoid=True))
        elif activation_type == 'softmax':
             all_transforms.append(mt.Activationsd(keys=pred_key, softmax=True))
        else:
            logging.warning(f"Unknown activation type '{activation_type}' in postprocessing. Skipping activation.")

        # 2. Discretization/Thresholding
        threshold = config.postprocessing.get('threshold', 0.5)
        all_transforms.append(mt.AsDiscreted(keys=pred_key, threshold=threshold))

        # 3. Further mask transforms
        mask_transforms_config = config.postprocessing.get('mask_transforms', [])
        # Ensure default key is only 'pred' for these transforms
        all_transforms.extend(create_transform_pipeline('postprocessing.mask_transforms', mask_transforms_config, default_keys=[pred_key]))

        logging.info(f"Successfully created postprocessing transform sequence with {len(all_transforms)} steps.")
        return Compose(all_transforms)
    except Exception as e:
         logging.error(f"Error occurred during get_postprocessing_transforms: {e}", exc_info=True)
         return None