"""
Depth Estimation Model using ViT Encoder + Decoder.

Architecture:
- ViT encoder extracts patch features (14x14 grid for 224px input with patch16)
- ResNet encoder extracts features (stride 32)
- Simple decoder upsamples back to input resolution
- Outputs single-channel depth map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DepthDecoder(nn.Module):
    """
    Simple convolutional decoder to upsample ViT features to full resolution.
    
    Takes (B, D, H_patch, W_patch) and outputs (B, 1, H_img, W_img)
    For ViT-small with patch16 on 224px: (B, 384, 14, 14) -> (B, 1, 224, 224)
    """
    
    def __init__(self, in_channels=384, hidden_channels=256, out_size=224, num_upsample=4):
        super().__init__()
        self.out_size = out_size
        # Progressive upsampling: 14 -> 28 -> 56 -> 112 -> 224

        hc_sizes = [
            in_channels,
            hidden_channels,
            hidden_channels // 2,
            hidden_channels // 4,
            hidden_channels // 8,
            32,
        ]
        
        # Adjust for extra upsampling if needed (e.g. for ResNet stride 32)
        if num_upsample == 5:
            hc_sizes = [
                in_channels,      # e.g. 2048
                hidden_channels, # 512
                hidden_channels//2,     # 256
                hidden_channels // 4, # 128
                hidden_channels // 8, # 64
                32,
            ]

        self.net = nn.Sequential(*[ 
            nn.Sequential(
                nn.ConvTranspose2d(hc_sizes[i], hc_sizes[i+1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hc_sizes[i+1]),
                nn.GELU(),
            )
            for i in range(num_upsample)
        ])
        
        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),  # Output in [0, 1] for normalized depth
        )
        
    def forward(self, x):
        """
        x: (B, D, H_patch, W_patch) - spatial features from ViT
        Returns: (B, 1, H_img, W_img) - depth map
        """
        x = self.net(x)
        x = self.head(x)
        
        # Ensure exact output size (in case input wasn't perfect multiple)
        if x.shape[-1] != self.out_size:
            x = F.interpolate(x, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)
        
        return x


class DepthViT(nn.Module):
    """
    ViT-based depth estimation model.
    
    Uses pretrained ViT encoder, extracts patch tokens,
    reshapes to spatial grid, and decodes to depth map.
    
    Supports dynamic image sizes for multi-view JEPA training.
    """
    
    def __init__(
        self, 
        model_name="vit_small_patch16_224.augreg_in21k",
        img_size=224,
        pretrained=True,
        freeze_encoder=False,
        num_upsample=4,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = 16  # Standard for ViT
        self.grid_size = img_size // self.patch_size  # 224/16 = 14
        
        # Create ViT backbone with dynamic image size support
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            drop_path_rate=0.1,
            dynamic_img_size=True,  # Enable dynamic sizes for multi-view
        )
        
        self.feat_dim = self.backbone.num_features  # 384 for vit_small, 768 for vit_base
        
        # Optionally freeze encoder for faster training
        if freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Gradient checkpointing for memory efficiency
        self.backbone.set_grad_checkpointing(True)
        
        # Decoder for global view size
        self.decoder = DepthDecoder(
            in_channels=self.feat_dim,
            hidden_channels=256,
            out_size=img_size,
            num_upsample=num_upsample
        )
        
        # For JEPA: projection head to get embedding for SIGReg
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512, affine=False),
        )
        
    def forward_features(self, x):
        """
        Extract patch features from ViT.
        
        Handles dynamic image sizes for multi-view JEPA training.
        
        Returns:
            patch_features: (B, D, H_grid, W_grid) - spatial feature map
            cls_token: (B, D) - global embedding
        """
        B, C, H, W = x.shape
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        
        # Use timm's forward_features which handles dynamic sizes
        # Returns (B, N+1, D) where N = grid_h * grid_w
        features = self.backbone.forward_features(x)
        
        # Separate CLS and patch tokens
        cls_token = features[:, 0]  # (B, D)
        patch_tokens = features[:, 1:]  # (B, N, D)
        
        # Reshape patches to spatial grid: (B, N, D) -> (B, D, H, W)
        patch_features = patch_tokens.transpose(1, 2).reshape(
            B, self.feat_dim, grid_h, grid_w
        )
        
        return patch_features, cls_token
    
    def forward(self, x, return_embedding=False):
        """
        Forward pass.
        
        Args:
            x: (B, 3, H, W) RGB input
            return_embedding: If True, also return global embedding for SIGReg
            
        Returns:
            depth: (B, 1, H, W) predicted depth map
            embedding: (B, D) global embedding (if return_embedding=True)
        """
        patch_features, cls_token = self.forward_features(x)
        
        # Decode to depth map
        depth = self.decoder(patch_features)
        
        if return_embedding:
            embedding = self.proj(cls_token)
            return depth, embedding
        
        return depth


class DepthResNet(nn.Module):
    """
    ResNet-based depth estimation model.
    Supports varying input sizes.
    """
    
    def __init__(
        self,
        model_name="resnet50.a1_in1k",
        img_size=224, # Used for decoder target size
        pretrained=True,
        freeze_encoder=False,
    ):
        super().__init__()
        
        self.img_size = img_size
        
        # Create ResNet backbone
        # features_only=True returns a list of features from different stages
        # out_indices=(4,) gives us the final feature map (stride 32)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),
        )
        
        self.feat_dim = self.backbone.feature_info[0]['num_chs'] # Should be 2048 for ResNet50
        
        if freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Decoder for ResNet (stride 32 -> 5 upsamples)
        self.decoder = DepthDecoder(
            in_channels=self.feat_dim,
            hidden_channels=256, 
            out_size=img_size,
            num_upsample=5
        )
        
        # Projection head for JEPA/SIGReg
        # Global pooling for embedding
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512, affine=False),
        )

    def forward(self, x, return_embedding=False):
        # x: (B, 3, H, W)
        features = self.backbone(x) # List of one tensor
        x_feat = features[0] # (B, 2048, H/32, W/32)
        
        # Map to depth
        depth = self.decoder(x_feat)
        
        if return_embedding:
            # Global Average Pooling for embedding
            pooled = self.pool(x_feat).flatten(1)
            embedding = self.proj(pooled)
            return depth, embedding
            
        return depth

def get_depth_model(model_name, **kwargs):
    if "resnet" in model_name:
        return DepthResNet(model_name=model_name, **kwargs)
    else:
        return DepthViT(model_name=model_name, **kwargs)


class ScaleInvariantLoss(nn.Module):
    """
    Scale-Invariant Loss for depth estimation.
    
    From "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"
    (Eigen et al., NIPS 2014)
    
    L = (1/n) * sum(d_i^2) - (lambda/n^2) * (sum(d_i))^2
    where d_i = log(pred_i) - log(gt_i)
    """
    
    def __init__(self, lambd=0.5, eps=1e-6):
        super().__init__()
        self.lambd = lambd
        self.eps = eps
    
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: (B, 1, H, W) predicted depth [0, 1]
            target: (B, 1, H, W) ground truth depth [0, 1]
            mask: Optional (B, 1, H, W) valid pixel mask
        """
        # Add epsilon to avoid log(0)
        pred = pred.clamp(min=self.eps)
        target = target.clamp(min=self.eps)
        
        # Log difference
        d = torch.log(pred) - torch.log(target)
        
        if mask is not None:
            d = d * mask
            n = mask.sum() + self.eps
        else:
            n = d.numel()
        
        # Scale-invariant loss
        loss = (d ** 2).sum() / n - self.lambd * (d.sum() ** 2) / (n ** 2)
        
        return loss


class DepthSmoothL1Loss(nn.Module):
    """
    Smooth L1 loss with gradient penalty for depth estimation.
    Encourages smoothness while preserving edges.
    """
    
    def __init__(self, edge_weight=0.5):
        super().__init__()
        self.edge_weight = edge_weight
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) predicted depth
            target: (B, 1, H, W) ground truth depth
        """
        # Main reconstruction loss
        recon_loss = F.smooth_l1_loss(pred, target)
        
        # Gradient loss (edge-aware)
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        grad_loss = F.smooth_l1_loss(pred_dx, target_dx) + F.smooth_l1_loss(pred_dy, target_dy)
        
        return recon_loss + self.edge_weight * grad_loss


# Test
if __name__ == "__main__":
    # Test ViT
    print("Testing ViT...")
    model = DepthViT(model_name="vit_small_patch16_224.augreg_in21k", img_size=224)
    x = torch.randn(2, 3, 224, 224)
    depth, emb = model(x, return_embedding=True)
    print(f"ViT Output: {depth.shape}, Emb: {emb.shape}")
    
    # Test ResNet
    print("\nTesting ResNet...")
    model_res = DepthResNet(model_name="resnet50.a1_in1k", img_size=224)
    x = torch.randn(2, 3, 224, 224)
    depth, emb = model_res(x, return_embedding=True)
    print(f"ResNet Output: {depth.shape}, Emb: {emb.shape}")
