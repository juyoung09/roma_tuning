import torch
import torch.nn as nn
import torchvision.models as tvm
from romatch.utils.utils import get_autocast_params

class VGG19(nn.Module):
    def __init__(self, pretrained=True, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        if pretrained:
            weights = tvm.vgg.VGG19_BN_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.layers = nn.ModuleList(tvm.vgg19_bn(weights=weights).features[:40])
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x
                    scale = scale*2
                x = layer(x)
            return feats

class TimmWrapper(nn.Module):
    def __init__(self, model_name, **kwargs):
        super().__init__()
        import timm
        self.model = timm.create_model(model_name, pretrained=True, dynamic_img_size=True, **kwargs)
        self.model.eval()

    def forward_features(self, x):
        out = self.model.forward_features(x)
        # Check if output is (B, N, C) or (B, C, H, W)
        if out.ndim == 3:
             # Assume (B, N, C)
             # Use num_prefix_tokens to handle CLS and Registers
             n_prefix = getattr(self.model, 'num_prefix_tokens', 1)
             return {'x_norm_patchtokens': out[:, n_prefix:]}
        elif out.ndim == 4:
            # (B, C, H, W) -> Flatten to (B, N, C)
            B, C, H, W = out.shape
            patches = out.permute(0, 2, 3, 1).reshape(B, -1, C)
            return {'x_norm_patchtokens': patches}
        else:
            raise ValueError(f"Unexpected output shape from timm model: {out.shape}")


class DINOv3Wrapper(nn.Module):
    """
    DINOv3 wrapper for Hugging Face transformers models.
    
    Supports models like:
    - facebook/dinov3-vit7b16-pretrain-sat493m (satellite, 7B params, patch 16)
    - facebook/dinov3-vitl16-pretrain-sat493m (satellite, Large)
    - facebook/dinov3-vitl16-pretrain (web data, Large)
    
    DINOv3 특성:
    - patch_size: 16 (DINOv2는 14)
    - ViT-L: embed_dim=1024, ViT-7B: embed_dim=4096
    - 4 register tokens + 1 CLS token
    """
    
    # 모델별 설정
    MODEL_CONFIGS = {
        'facebook/dinov3-vit7b16-pretrain-sat493m': {'embed_dim': 4096, 'patch_size': 16},
        'facebook/dinov3-vitl16-pretrain-sat493m': {'embed_dim': 1024, 'patch_size': 16},
        'facebook/dinov3-vitl16-pretrain': {'embed_dim': 1024, 'patch_size': 16},
        'facebook/dinov3-vitb16-pretrain': {'embed_dim': 768, 'patch_size': 16},
        'facebook/dinov3-vits16-pretrain': {'embed_dim': 384, 'patch_size': 16},
    }
    
    def __init__(self, model_name="facebook/dinov3-vit7b16-pretrain-sat493m", device_map="auto"):
        super().__init__()
        from transformers import AutoModel
        
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=torch.float16,  # torch_dtype is deprecated
            trust_remote_code=True,
        )
        self.model.eval()
        
        # 모델 설정 가져오기
        if model_name in self.MODEL_CONFIGS:
            config = self.MODEL_CONFIGS[model_name]
            self.embed_dim = config['embed_dim']
            self.patch_size = config['patch_size']
        else:
            # 기본값 (Large 모델 기준)
            self.embed_dim = getattr(self.model.config, 'hidden_size', 1024)
            self.patch_size = getattr(self.model.config, 'patch_size', 16)
        
        # DINOv3는 CLS + 4 registers = 5 prefix tokens
        self.num_prefix_tokens = 5
        
        print(f"[DINOv3] Loaded {model_name}")
        print(f"[DINOv3] embed_dim={self.embed_dim}, patch_size={self.patch_size}")
    
    @property
    def num_features(self):
        return self.embed_dim
    
    def forward_features(self, x):
        """
        Forward pass to get patch features.
        
        Args:
            x: Input tensor (B, 3, H, W), normalized to [0, 1] or ImageNet norm
        
        Returns:
            dict with 'x_norm_patchtokens': (B, N, C) patch tokens
        """
        # DINOv3는 이미지 정규화가 내부에서 처리됨
        # 하지만 RoMa는 이미 정규화된 이미지를 전달하므로 그대로 사용
        
        with torch.no_grad():
            outputs = self.model(pixel_values=x, output_hidden_states=True)
        
        # last_hidden_state: (B, N, C) where N = 1 (CLS) + 4 (registers) + num_patches
        hidden_states = outputs.last_hidden_state
        
        # CLS + register tokens 제외하고 patch tokens만 추출
        patch_tokens = hidden_states[:, self.num_prefix_tokens:, :]
        
        return {'x_norm_patchtokens': patch_tokens}

class CNNandDinov2(nn.Module):
    """
    CNN + DINOv2/v3 hybrid encoder for RoMa.
    
    Supports:
    - DINOv2 (default): patch_size=14, embed_dim=1024
    - DINOv3: patch_size=16, embed_dim varies (1024 for L, 4096 for 7B)
    - Timm backbones
    """
    
    def __init__(self, cnn_kwargs=None, amp=False, dinov2_weights=None, amp_dtype=torch.float16, 
                 backbone=None, dinov3_model=None):
        super().__init__()
        
        self.patch_size = 14  # default for DINOv2
        self.use_dinov3 = False
        
        if dinov3_model is not None:
            # DINOv3 모델 사용
            backbone_model = DINOv3Wrapper(dinov3_model)
            self.patch_size = backbone_model.patch_size
            self.use_dinov3 = True
            print(f"[Encoder] Using DINOv3: {dinov3_model}, patch_size={self.patch_size}")
        elif backbone is not None:
            backbone_model = TimmWrapper(backbone)
        else:
            # 기본 DINOv2
            if dinov2_weights is None:
                dinov2_weights = torch.hub.load_state_dict_from_url(
                    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", 
                    map_location="cpu"
                )
            from .transformer import vit_large
            vit_kwargs = dict(
                img_size=518,
                patch_size=14,
                init_values=1.0,
                ffn_layer="mlp",
                block_chunks=0,
            )
            dinov2_vitl14 = vit_large(**vit_kwargs).eval()
            dinov2_vitl14.load_state_dict(dinov2_weights)
            backbone_model = dinov2_vitl14

        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        self.cnn = VGG19(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        
        if self.amp and not self.use_dinov3:
            # DINOv3는 이미 float16으로 로드됨
            if isinstance(backbone_model, nn.Module):
                backbone_model = backbone_model.to(self.amp_dtype)
        
        self.dinov2_vitl14 = [backbone_model]  # ugly hack to not show parameters to DDP
        
        # Check backbone dimension and create projection if needed
        self.proj = None
        target_dim = 1024  # RoMa decoder expects 1024
        
        # Heuristic to find output dim
        embed_dim = 1024  # default for vit_large
        if self.use_dinov3:
            embed_dim = backbone_model.embed_dim
        elif isinstance(backbone_model, TimmWrapper):
            embed_dim = backbone_model.model.num_features
        elif hasattr(backbone_model, 'embed_dim'):
            embed_dim = backbone_model.embed_dim
        
        self.embed_dim = embed_dim
        
        if embed_dim != target_dim:
            self.proj = nn.Conv2d(embed_dim, target_dim, 1, 1)
            print(f"[Encoder] Added projection: {embed_dim} -> {target_dim}")
    
    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def forward(self, x, upsample=False):
        B, C, H, W = x.shape
        feature_pyramid = self.cnn(x)
        
        if not upsample:
            with torch.no_grad():
                backbone = self.dinov2_vitl14[0]
                
                if backbone is not None:
                    # Device handling
                    if self.use_dinov3:
                        # DINOv3는 device_map="auto"로 이미 배치됨
                        input_tensor = x.to(torch.float16)
                    else:
                        if hasattr(backbone, 'device') and backbone.device != x.device:
                            backbone = backbone.to(x.device).to(self.amp_dtype)
                            self.dinov2_vitl14[0] = backbone
                        if isinstance(backbone, TimmWrapper):
                            if next(backbone.parameters()).device != x.device:
                                backbone.to(x.device)
                        input_tensor = x.to(self.amp_dtype)
                    
                    # Forward pass
                    features_out = backbone.forward_features(input_tensor)
                    patch_tokens = features_out['x_norm_patchtokens']
                    
                    # Reshape: (B, N, C) -> (B, C, H', W')
                    # H' = H // patch_size, W' = W // patch_size
                    H_feat = H // self.patch_size
                    W_feat = W // self.patch_size
                    
                    features_16 = patch_tokens.permute(0, 2, 1).reshape(B, -1, H_feat, W_feat)
                
                # Project if needed (e.g., DINOv3 7B: 4096 -> 1024)
                if self.proj is not None:
                    # Ensure proj is on correct device and dtype
                    if self.proj.weight.device != features_16.device or self.proj.weight.dtype != features_16.dtype:
                        self.proj = self.proj.to(device=features_16.device, dtype=features_16.dtype)
                    features_16 = self.proj(features_16)
                
                feature_pyramid[16] = features_16
        
        return feature_pyramid