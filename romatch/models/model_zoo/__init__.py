from typing import Union
import torch
from .roma_models import roma_model, tiny_roma_v1_model


weight_urls = {
    "romatch": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
        "indoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
    },
    "tiny_roma_v1": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/tiny_roma_v1_outdoor.pth",
    },
    "dinov2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
}

# DINOv3 모델 목록 (위성 영상용)
DINOV3_MODELS = {
    "sat-7b": "facebook/dinov3-vit7b16-pretrain-sat493m",  # 위성 전용 7B
    "sat-large": "facebook/dinov3-vitl16-pretrain-sat493m",  # 위성 전용 Large
    "web-large": "facebook/dinov3-vitl16-pretrain",  # 웹 데이터 Large
    "web-base": "facebook/dinov3-vitb16-pretrain",  # 웹 데이터 Base
}


def tiny_roma_v1_outdoor(device, weights=None, xfeat=None):
    if weights is None:
        weights = torch.hub.load_state_dict_from_url(
            weight_urls["tiny_roma_v1"]["outdoor"], map_location=device
        )
    if xfeat is None:
        xfeat = torch.hub.load(
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=4096
        ).net

    return tiny_roma_v1_model(weights=weights, xfeat=xfeat).to(device)


def roma_outdoor(
    device,
    weights=None,
    dinov2_weights=None,
    coarse_res: Union[int, tuple[int, int]] = 560,
    upsample_res: Union[int, tuple[int, int]] = 864,
    amp_dtype: torch.dtype = torch.float16,
    symmetric=True,
    use_custom_corr=True,
    upsample_preds=True,
    **kwargs,
):
    if weights is None:
        weights = torch.hub.load_state_dict_from_url(
            weight_urls["romatch"]["outdoor"], map_location=device
        )
    if dinov2_weights is None:
        dinov2_weights = torch.hub.load_state_dict_from_url(
            weight_urls["dinov2"], map_location=device
        )
    model = roma_model(
        resolution=coarse_res,
        upsample_preds=upsample_preds,
        weights=weights,
        dinov2_weights=dinov2_weights,
        device=device,
        amp_dtype=amp_dtype,
        symmetric=symmetric,
        use_custom_corr=use_custom_corr,
        upsample_res=upsample_res,
        **kwargs,
    )
    return model


def roma_indoor(
    device,
    weights=None,
    dinov2_weights=None,
    coarse_res: Union[int, tuple[int, int]] = 560,
    upsample_res: Union[int, tuple[int, int]] = 864,
    amp_dtype: torch.dtype = torch.float16,
    symmetric=True,
    use_custom_corr=True,
    upsample_preds=True,
):
    if weights is None:
        weights = torch.hub.load_state_dict_from_url(
            weight_urls["romatch"]["indoor"], map_location=device
        )
    if dinov2_weights is None:
        dinov2_weights = torch.hub.load_state_dict_from_url(
            weight_urls["dinov2"], map_location=device
        )
    model = roma_model(
        resolution=coarse_res,
        upsample_preds=upsample_preds,
        weights=weights,
        dinov2_weights=dinov2_weights,
        device=device,
        amp_dtype=amp_dtype,
        symmetric=symmetric,
        use_custom_corr=use_custom_corr,
        upsample_res=upsample_res,
    )
    return model


def roma_outdoor_dinov3(
    device,
    weights=None,
    dinov3_model: str = "sat-7b",
    coarse_res: Union[int, tuple[int, int]] = 560,
    upsample_res: Union[int, tuple[int, int]] = 864,
    amp_dtype: torch.dtype = torch.float16,
    symmetric=True,
    use_custom_corr=True,
    upsample_preds=True,
    **kwargs,
):
    """
    RoMa with DINOv3 backbone for satellite imagery.
    
    Args:
        device: torch device
        weights: RoMa decoder weights (optional, uses pretrained if None)
        dinov3_model: DINOv3 model name. Options:
            - "sat-7b": facebook/dinov3-vit7b16-pretrain-sat493m (recommended for satellite)
            - "sat-large": facebook/dinov3-vitl16-pretrain-sat493m
            - "web-large": facebook/dinov3-vitl16-pretrain
            - "web-base": facebook/dinov3-vitb16-pretrain
            - Or full HuggingFace model path
        coarse_res: Resolution for coarse matching (must be multiple of 16 for DINOv3)
        upsample_res: Resolution for upsampled predictions
        amp_dtype: Mixed precision dtype
        symmetric: Use symmetric matching
        use_custom_corr: Use custom correlation (Linux only)
        upsample_preds: Upsample predictions
    
    Returns:
        RoMa matcher model
    """
    # DINOv3 모델 이름 해석
    if dinov3_model in DINOV3_MODELS:
        dinov3_model_name = DINOV3_MODELS[dinov3_model]
    else:
        dinov3_model_name = dinov3_model
    
    print(f"[roma_outdoor_dinov3] Using DINOv3: {dinov3_model_name}")
    
    # RoMa decoder weights
    if weights is None:
        weights = torch.hub.load_state_dict_from_url(
            weight_urls["romatch"]["outdoor"], map_location=device
        )
    
    # DINOv3는 patch_size=16이므로 resolution이 16의 배수여야 함
    if isinstance(coarse_res, int):
        coarse_res = (coarse_res, coarse_res)
    
    # 16의 배수로 조정
    coarse_res = (
        (coarse_res[0] // 16) * 16,
        (coarse_res[1] // 16) * 16
    )
    print(f"[roma_outdoor_dinov3] Coarse resolution: {coarse_res}")
    
    model = roma_model(
        resolution=coarse_res,
        upsample_preds=upsample_preds,
        weights=weights,
        dinov2_weights=None,  # DINOv3 사용 시 불필요
        device=device,
        amp_dtype=amp_dtype,
        symmetric=symmetric,
        use_custom_corr=use_custom_corr,
        upsample_res=upsample_res,
        dinov3_model=dinov3_model_name,  # DINOv3 모델 전달
        **kwargs,
    )
    return model