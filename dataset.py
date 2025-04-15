# dataset.py
import os
import json
from monai.data import Dataset, DataLoader, CacheDataset, PersistentDataset, ThreadDataLoader
from monai.utils import first
import torch

# Import transforms functions
from transforms import get_train_transforms, get_val_transforms

def get_datasets(config):
    """Loads data lists and creates MONAI Datasets."""

    if not config.data.use_preprocessed_3d:
        raise NotImplementedError("Direct 2D PNG processing not fully implemented yet. Run preprocess_data.py first.")

    # Load the JSON file created by preprocess_data.py
    json_path_template = os.path.join(config.base_dir, config.data.preprocessed_dir, f"dataset_fold_{config.fold}.json")
    json_path = json_path_template.replace(f"fold_{config.fold}", f"fold_{config.fold}") # Ensure correct fold formatting
    print(f"Loading data list from: {json_path}")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset JSON file not found: {json_path}. Run preprocess_data.py first.")

    with open(json_path, 'r') as f:
        data_list = json.load(f)

    train_files = data_list['train']
    val_files = data_list['val']

    # Create transform objects
    train_ts = get_train_transforms(config)
    val_ts = get_val_transforms(config)

    # Choose Dataset type based on config
    dataset_type = config.data.get('dataset_type', 'CacheDataset').lower()
    train_cache_rate = config.data.get('train_cache_rate', 1.0)
    val_cache_rate = config.data.get('val_cache_rate', 1.0)
    num_workers = config.training.get('num_workers', 4)
    persistent_cache_dir = os.path.join(config.base_dir, "data_cache", f"fold_{config.fold}")

    print(f"Using {dataset_type} for training (cache={train_cache_rate}) and validation (cache={val_cache_rate})")

    if dataset_type == 'persistentdataset':
        os.makedirs(persistent_cache_dir, exist_ok=True)
        train_ds = PersistentDataset(data=train_files, transform=train_ts, cache_dir=os.path.join(persistent_cache_dir, "train"))
        val_ds = PersistentDataset(data=val_files, transform=val_ts, cache_dir=os.path.join(persistent_cache_dir, "val"))
    elif dataset_type == 'cachedataset':
         # copy_cache=False to avoid excessive disk usage if cache_rate=1.0
        train_ds = CacheDataset(data=train_files, transform=train_ts, cache_rate=train_cache_rate, num_workers=num_workers, copy_cache=False)
        val_ds = CacheDataset(data=val_files, transform=val_ts, cache_rate=val_cache_rate, num_workers=num_workers, copy_cache=False)
    else: # Default to standard Dataset
        train_ds = Dataset(data=train_files, transform=train_ts)
        val_ds = Dataset(data=val_files, transform=val_ts)


    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")

    # Check dataset output sample
    check_loader = DataLoader(train_ds, batch_size=1, num_workers=0) # Use 0 workers for check
    try:
         check_data = first(check_loader)
         image, label = check_data["image"], check_data["mask"]
         print(f"Sample - Image shape: {image.shape}, Label shape: {label.shape}")
         print(f"Sample - Image dtype: {image.dtype}, Label dtype: {label.dtype}")
         print(f"Sample - Image range: [{image.min()}, {image.max()}]")
         print(f"Sample - Label values: {torch.unique(label)}")
    except Exception as e:
         print(f"Warning: Could not check first item of DataLoader. Error: {e}")
    del check_loader, check_data

    return train_ds, val_ds


def get_dataloaders(train_ds, val_ds, config):
    """Creates MONAI DataLoaders."""

    num_workers = config.training.get('num_workers', 4)
    # MONAI's ThreadDataLoader can sometimes be faster for cached datasets by overlapping CPU/GPU work
    loader_type = config.data.get('loader_type', 'DataLoader').lower()

    print(f"Using {loader_type} for DataLoaders.")

    if loader_type == 'threaddataloader':
        # Often used with CacheDataset where transforms are pre-computed
        # num_workers for ThreadDataLoader is often set to 0 as caching handles parallel loading
        train_loader = ThreadDataLoader(
            train_ds,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0, # Set to 0 for ThreadDataLoader usually
            pin_memory=torch.cuda.is_available()
        )
        val_loader = ThreadDataLoader(
            val_ds,
            batch_size=config.training.val_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
    else: # Default to PyTorch DataLoader
        train_loader = DataLoader(
            train_ds,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True # Drop last batch if incomplete
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.training.val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    return train_loader, val_loader