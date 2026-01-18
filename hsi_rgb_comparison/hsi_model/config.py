
def get_default_config():
    """
    Returns the default configuration dictionary for the HSI Model.
    """
    config = {
        "workers": 0,
        "devices": "auto",
        "accelerator": "auto",
        "batch_size": 1,
        
        # Model Architecture
        "architecture": 'retinanet',
        "num_classes": 1,
        "in_channels": 3,
        "selected_bands": None, # List of indices
        "nms_thresh": 0.05,
        "score_thresh": 0.1,
        "backbone_weights": "DEFAULT", # Use default imagenet weights
        
        "label_dict": {"Tree": 0},
        
        # Training
        "train": {
            "csv_file": None,
            "root_dir": None,
            "lr": 0.001,
            "epochs": 10,
            "fast_dev_run": False,
            "augmentations": [
                {"HorizontalFlip": {"p": 0.5}}
            ]
        },
        
        # Validation
        "validation": {
            "csv_file": None,
            "root_dir": None,
            "iou_threshold": 0.5
        }
    }
    return config
