import torch
import os

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

try:
    print("Loading DINOv2 from torch.hub...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    print("DINOv2 loaded successfully!")
except Exception as e:
    print(f"DINOv2 failed: {e}")

try:
    print("\nLoading DINOv3 from torch.hub...")
    # RoMaV2에서 사용하는 정확한 인자
    model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov3:adc254450203739c8149213a7a69d8d905b4fcfa",
        model="dinov3_vitl16",
        pretrained=False, # 가중치 없이 구조만 로드 시도
        weights=None,
        skip_validation=True,
    )
    print("DINOv3 loaded successfully!")
except Exception as e:
    print(f"DINOv3 failed: {e}")



