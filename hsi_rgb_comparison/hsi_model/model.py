import torch
import torchvision
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetRegressionHead

class HSIModel(torch.nn.Module):
    """
    A DeepForest-style RetinaNet model rewritten for flexibility.
    """
    def __init__(self, num_classes=1, backbone_weights=None, nms_thresh=0.05, score_thresh=0.5, in_channels=3):
        super(HSIModel, self).__init__()
        
        # Load backbone (ResNet50 FPN)
        # We start with a pre-trained backbone if specified
        weights = None
        if backbone_weights == "DEFAULT":
            weights = torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
        
        self.model = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=weights,
            weights_backbone=None, # handled by weights normally, or can be specific
            num_classes=91 # Initialize with COCO classes then replace head
        )

        # Update first layer if in_channels is not 3
        if in_channels != 3:
            old_conv = self.model.backbone.body.conv1
            new_conv = torch.nn.Conv2d(
                in_channels, 
                old_conv.out_channels, 
                kernel_size=old_conv.kernel_size, 
                stride=old_conv.stride, 
                padding=old_conv.padding, 
                bias=old_conv.bias
            )
            
            # Initialize new conv weights
            # For the first 3 channels, we can copy weights if we want, or simple init
            # Here we use Xavier init
            torch.nn.init.xavier_uniform_(new_conv.weight)
            
            self.model.backbone.body.conv1 = new_conv

            # CRITICAL FIX: Update normalization parameters for non-RGB input
            # RetinaNet expects 3-channel mean/std by default. We must resize them.
            # Using mean=0, std=1 is safer for unknown domains unless statistics are known.
            self.model.transform.image_mean = [0.0] * in_channels
            self.model.transform.image_std = [1.0] * in_channels
        
        # Replace the head for our specific number of classes
        # This is standard procedure for finetuning
        # Note: If we changed the backbone, the out_channels might be same but we should check
        backbone_out_channels = self.model.backbone.out_channels
        num_anchors = self.model.head.classification_head.num_anchors
        
        self.model.head.classification_head = RetinaNetClassificationHead(
            backbone_out_channels,
            num_anchors,
            num_classes
        )
        
        # Adjust Inference Parameters
        self.model.nms_thresh = nms_thresh
        self.model.score_thresh = score_thresh
        
    def forward(self, images, targets=None):
        """
        Forward pass.
        Args:
            images: list of tensors
            targets: list of dicts (optional, for training)
        """
        return self.model(images, targets)
    
    def load_backbone_weights(self, path):
        """
        Load custom backbone weights similar to the original concept.
        """
        state_dict = torch.load(path)
        self.model.backbone.load_state_dict(state_dict, strict=False)

    def predict(self, images):
        """
        Prediction helper
        """
        self.eval()
        with torch.no_grad():
            return self.model(images)
