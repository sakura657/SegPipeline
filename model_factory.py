# model_factory.py
import monai.networks.nets as nets
import torch

def get_model(config):
    """Creates the MONAI model based on the configuration."""
    model_name = config.model.get('name', 'UNet') # Default to UNet
    model_params = config.model.get('params', {})

    try:
        model_class = getattr(nets, model_name)
        model = model_class(**model_params)
        print(f"Created model: {model_name} with params: {model_params}")

        # Optional: Load pretrained weights if specified
        pretrained_path = config.model.get('pretrained_weights', None)
        if pretrained_path:
            try:
                # Load weights, potentially needing strict=False if architecture differs slightly
                state_dict = torch.load(pretrained_path, map_location='cpu')
                # Adjust key names if needed (e.g., remove 'module.' prefix)
                if isinstance(state_dict, dict) and 'model' in state_dict:
                     state_dict = state_dict['model'] # Common pattern
                # Handle potential DataParallel prefix
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"Warning: Missing keys when loading pretrained weights: {missing_keys}")
                if unexpected_keys:
                     print(f"Warning: Unexpected keys when loading pretrained weights: {unexpected_keys}")
                print(f"Loaded pretrained weights from: {pretrained_path}")
            except Exception as e:
                print(f"Error loading pretrained weights from {pretrained_path}: {e}")

        return model

    except AttributeError:
        raise ValueError(f"Model '{model_name}' not found in monai.networks.nets.")
    except Exception as e:
        raise ValueError(f"Error initializing model '{model_name}' with params {model_params}: {e}")