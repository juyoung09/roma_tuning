#!/usr/bin/env python3
"""RoMaV2 단계별 디버그 스크립트"""

import sys
import os

print("=" * 60)
print("Step 1: Basic imports")
print("=" * 60)
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n" + "=" * 60)
print("Step 2: Check xformers")
print("=" * 60)
try:
    import xformers
    print(f"xformers version: {xformers.__version__}")
    import xformers.ops
    print("xformers.ops imported successfully")
except ImportError as e:
    print(f"xformers not available: {e}")
except Exception as e:
    print(f"xformers error: {e}")

print("\n" + "=" * 60)
print("Step 3: Add RoMaV2 to path")
print("=" * 60)
romav2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RoMaV2', 'src')
if not os.path.exists(romav2_path):
    # 상위 디렉토리에서 찾기
    romav2_path = '/gpfs/home/amkjy2545/ai/PyProjects/RoMa/RoMaV2/src'
print(f"RoMaV2 path: {romav2_path}")
print(f"Path exists: {os.path.exists(romav2_path)}")
if os.path.exists(romav2_path):
    sys.path.insert(0, romav2_path)
    print("Added to sys.path")

print("\n" + "=" * 60)
print("Step 4: List RoMaV2 contents")
print("=" * 60)
if os.path.exists(romav2_path):
    for item in os.listdir(romav2_path):
        print(f"  {item}")

print("\n" + "=" * 60)
print("Step 5: Import romav2 module")
print("=" * 60)
try:
    print("Attempting: from romav2 import RoMaV2")
    sys.stdout.flush()
    from romav2 import RoMaV2
    print("Import successful!")
except Exception as e:
    print(f"Import failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Step 6: Create RoMaV2 instance (CPU first)")
print("=" * 60)
try:
    print("Creating RoMaV2() on CPU...")
    sys.stdout.flush()
    model = RoMaV2()
    print("RoMaV2 instance created!")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"Failed to create RoMaV2: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Step 7: Apply settings")
print("=" * 60)
try:
    print("Applying 'fast' setting...")
    sys.stdout.flush()
    model.apply_setting("fast")
    print("Settings applied!")
except Exception as e:
    print(f"Failed to apply settings: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Step 8: Move to CUDA")
print("=" * 60)
try:
    device = torch.device('cuda')
    print(f"Moving model to {device}...")
    sys.stdout.flush()
    model.to(device)
    print("Model moved to CUDA!")
except Exception as e:
    print(f"Failed to move to CUDA: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Step 9: Set eval mode")
print("=" * 60)
try:
    model.eval()
    print("Eval mode set!")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("All steps completed successfully!")
print("=" * 60)