"""
NeonSAT 벤치마크 - 위성 영상 정합 평가

평가 방식:
1. 호모그래피 기반 평가 (RANSAC으로 아웃라이어 제거)
2. 다양한 PCK 임계값 (1, 3, 5, 10, 20픽셀)
3. 위성 영상 특성을 고려한 평가 기준
"""

import os
import glob
import argparse
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

def parse_labels(label_path):
    """Parses the label file to extract keypoints.
    Returns a dictionary {id: (x, y)}
    """
    kpts = {}
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                idx = int(parts[0])
                x = float(parts[1].replace(',', ''))
                y = float(parts[2].replace(',', ''))
                kpts[idx] = np.array([x, y])
    return kpts

def get_matches(kpts_A, kpts_B):
    """Finds common keypoints by ID.
    Returns Nx2 arrays for A and B.
    """
    common_ids = sorted(list(set(kpts_A.keys()) & set(kpts_B.keys())))
    pts_A = []
    pts_B = []
    for idx in common_ids:
        pts_A.append(kpts_A[idx])
        pts_B.append(kpts_B[idx])
    return np.array(pts_A), np.array(pts_B)


def compute_homography_accuracy(pred_kps, gt_kps, thresholds=[1, 3, 5, 10, 20]):
    """
    호모그래피 기반 정확도 계산
    
    Args:
        pred_kps: 예측된 키포인트 (N, 2)
        gt_kps: GT 키포인트 (N, 2)
        thresholds: PCK 임계값 리스트 (픽셀)
    
    Returns:
        dict: 각 임계값에 대한 정확도
    """
    if len(pred_kps) < 4 or len(gt_kps) < 4:
        return {t: 0.0 for t in thresholds}, np.inf, None
    
    # RANSAC으로 호모그래피 추정 및 인라이어 식별
    H, inlier_mask = cv2.findHomography(
        pred_kps.reshape(-1, 1, 2).astype(np.float32),
        gt_kps.reshape(-1, 1, 2).astype(np.float32),
        cv2.RANSAC,
        ransacReprojThreshold=5.0,
        confidence=0.999,
        maxIters=5000
    )
    
    if H is None:
        return {t: 0.0 for t in thresholds}, np.inf, None
    
    # 호모그래피로 변환된 좌표와 GT 비교
    pred_kps_h = np.hstack([pred_kps, np.ones((len(pred_kps), 1))])
    transformed = (H @ pred_kps_h.T).T
    transformed = transformed[:, :2] / (transformed[:, 2:3] + 1e-10)
    
    # 거리 계산
    distances = np.linalg.norm(transformed - gt_kps, axis=1)
    
    # 각 임계값에 대한 PCK 계산
    pck_results = {}
    for t in thresholds:
        pck_results[t] = np.mean(distances < t) * 100.0
    
    mean_error = np.mean(distances)
    
    return pck_results, mean_error, H


def run_benchmark(data_dir, model, device='cuda', limit=None, visualize=True, save_results=True):
    """
    NeonSAT 벤치마크 실행
    
    Args:
        data_dir: 데이터 디렉토리
        model: RoMa 모델
        device: 디바이스
        limit: 처리할 페어 수 제한
        visualize: 시각화 여부
        save_results: 결과 저장 여부
    
    Returns:
        dict: 벤치마크 결과
    """
    import torch
    import torch.nn.functional as F
    
    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')
    
    # K3A, K3, K3 이미지 모두 찾기
    k3_images = []
    for pattern in ['*K3A*.png', '*K3_*.png']:
        k3_images.extend(glob.glob(os.path.join(image_dir, pattern)))
    k3_images = sorted(list(set(k3_images)))
    
    print(f"Found {len(k3_images)} image pairs.")
    if limit is not None:
        print(f"Limiting to random {limit} pairs.")
        # 공정한 비교를 위해 시드 고정
        random.seed(42)
        random.shuffle(k3_images)
        k3_images = k3_images[:limit]
    
    # 결과 저장용
    all_errors = []
    all_pck = {t: [] for t in [1, 3, 5, 10, 20]}
    pair_results = []
    
    for im_A_path in tqdm(k3_images, desc="Evaluating"):
        basename = os.path.basename(im_A_path)
        coreg_id = basename.split('_')[1]
        
        # N01 이미지 찾기
        n01_pattern = os.path.join(image_dir, f"coreg_{coreg_id}_*N01*.png")
        n01_candidates = glob.glob(n01_pattern)
        if not n01_candidates:
            continue
        im_B_path = n01_candidates[0]
        
        # 라벨 파일 찾기
        label_A_pattern = os.path.join(label_dir, f"coreg_{coreg_id}_*K3*.txt")
        label_B_pattern = os.path.join(label_dir, f"coreg_{coreg_id}_*N01*.txt")
        
        label_A_files = glob.glob(label_A_pattern)
        label_B_files = glob.glob(label_B_pattern)
        
        if not label_A_files or not label_B_files:
            continue
        
        kpts_A_dict = parse_labels(label_A_files[0])
        kpts_B_dict = parse_labels(label_B_files[0])
        pts_A, pts_B = get_matches(kpts_A_dict, kpts_B_dict)
        
        if len(pts_A) < 4:
            continue
        
        # 이미지 로드
        im_A = Image.open(im_A_path).convert("RGB")
        im_B = Image.open(im_B_path).convert("RGB")
        W_A, H_A = im_A.size
        W_B, H_B = im_B.size
        
        # RoMa 매칭
        with torch.no_grad():
            warp, certainty = model.match(im_A, im_B, device=device)
        
        if warp.dim() == 4:
            warp = warp[0]
        
        # symmetric=True일 경우 warp shape: (H, W*2, 4)
        # warp[:, :W, :] = A->B, warp[:, W:, :] = B->A
        H_warp, W_warp, _ = warp.shape
        if W_warp > H_warp:  # symmetric으로 width가 2배
            warp = warp[:, :W_warp//2, :]  # A->B 부분만 사용
        
        # GT 키포인트 위치에서 예측된 B 좌표 샘플링
        pts_A_norm = torch.tensor(pts_A, device=device, dtype=torch.float32)
        pts_A_norm[:, 0] = 2 * pts_A_norm[:, 0] / W_A - 1
        pts_A_norm[:, 1] = 2 * pts_A_norm[:, 1] / H_A - 1
        pts_A_norm = pts_A_norm.unsqueeze(0).unsqueeze(0)
        
        pred_B_map = warp[..., 2:].permute(2, 0, 1).unsqueeze(0)
        pred_B_norm = F.grid_sample(pred_B_map, pts_A_norm, align_corners=False, mode='bilinear')
        pred_B_norm = pred_B_norm.squeeze(0).squeeze(1).permute(1, 0)
        
        # 픽셀 좌표로 변환
        pred_B_pix = pred_B_norm.clone()
        pred_B_pix[:, 0] = (pred_B_pix[:, 0] + 1) * W_B / 2
        pred_B_pix[:, 1] = (pred_B_pix[:, 1] + 1) * H_B / 2
        pred_B_pix = pred_B_pix.cpu().numpy()
        
        # 직접적인 오차 계산 (호모그래피 없이)
        direct_errors = np.linalg.norm(pred_B_pix - pts_B, axis=1)
        
        # 호모그래피 기반 정확도 계산
        pck_results, mean_error, H = compute_homography_accuracy(pred_B_pix, pts_B)
        
        # 결과 저장
        pair_result = {
            'id': coreg_id,
            'num_keypoints': len(pts_A),
            'direct_mean_error': np.mean(direct_errors),
            'direct_median_error': np.median(direct_errors),
            'homo_mean_error': mean_error,
        }
        
        for t in [1, 3, 5, 10, 20]:
            # 직접 오차 기반 PCK
            direct_pck = np.mean(direct_errors < t) * 100.0
            pair_result[f'direct_pck@{t}'] = direct_pck
            all_pck[t].append(direct_pck)
        
        all_errors.extend(direct_errors)
        pair_results.append(pair_result)
        
        # 첫 번째 페어 디버그 출력
        if len(pair_results) == 1 and visualize:
            print(f"\nDebug: Image {basename}")
            print(f"  Image Size A: {W_A}x{H_A}, B: {W_B}x{H_B}")
            print(f"  Number of keypoints: {len(pts_A)}")
            print(f"  Direct Mean Error: {np.mean(direct_errors):.2f}px")
            print(f"  Direct Median Error: {np.median(direct_errors):.2f}px")
            print(f"\n  First 5 GT matches (A -> B):")
            for i in range(min(5, len(pts_A))):
                print(f"    A({pts_A[i][0]:.1f}, {pts_A[i][1]:.1f}) -> GT B({pts_B[i][0]:.1f}, {pts_B[i][1]:.1f})")
                print(f"                                -> Pred B({pred_B_pix[i, 0]:.1f}, {pred_B_pix[i, 1]:.1f})")
                print(f"                                   Error: {direct_errors[i]:.2f}px")
            
            # 시각화
            try:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # 이미지 A
                axes[0].imshow(im_A)
                axes[0].scatter(pts_A[:, 0], pts_A[:, 1], c='red', s=20, marker='x', label='GT Keypoints')
                axes[0].set_title(f"Image A (K3A)\n{len(pts_A)} keypoints")
                axes[0].legend()
                
                # 이미지 B with GT and Pred
                axes[1].imshow(im_B)
                axes[1].scatter(pts_B[:, 0], pts_B[:, 1], c='green', s=30, marker='o', label='GT B', alpha=0.7)
                axes[1].scatter(pred_B_pix[:, 0], pred_B_pix[:, 1], c='blue', s=20, marker='x', label='Pred B')
                axes[1].set_title(f"Image B (N01)\nGreen=GT, Blue=Pred")
                axes[1].legend()
                
                # 오차 분포
                axes[2].hist(direct_errors, bins=50, edgecolor='black', alpha=0.7)
                axes[2].axvline(x=np.mean(direct_errors), color='r', linestyle='--', label=f'Mean: {np.mean(direct_errors):.2f}px')
                axes[2].axvline(x=np.median(direct_errors), color='g', linestyle='--', label=f'Median: {np.median(direct_errors):.2f}px')
                axes[2].set_xlabel('Error (pixels)')
                axes[2].set_ylabel('Count')
                axes[2].set_title('Error Distribution')
                axes[2].legend()
                
                plt.tight_layout()
                plt.savefig("debug_match_vis.png", dpi=150)
                print("Saved debug visualization to debug_match_vis.png")
                plt.close()
            except Exception as e:
                print(f"Visualization error: {e}")
    
    # 최종 결과 계산
    all_errors = np.array(all_errors)
    
    results = {
        'num_pairs': len(pair_results),
        'num_keypoints_total': len(all_errors),
        'mean_error': np.mean(all_errors),
        'median_error': np.median(all_errors),
        'std_error': np.std(all_errors),
    }
    
    print("\n" + "="*60)
    print("NeonSAT Benchmark Results")
    print("="*60)
    print(f"Total pairs evaluated: {results['num_pairs']}")
    print(f"Total keypoints: {results['num_keypoints_total']}")
    print(f"\nError Statistics:")
    print(f"  Mean Error: {results['mean_error']:.2f}px")
    print(f"  Median Error: {results['median_error']:.2f}px")
    print(f"  Std Error: {results['std_error']:.2f}px")
    
    print(f"\nPCK (Percentage of Correct Keypoints):")
    for t in [1, 3, 5, 10, 20]:
        pck_value = np.mean(all_pck[t])
        results[f'pck@{t}'] = pck_value
        indicator = "✓" if pck_value >= 90.0 else " "
        print(f"  [{indicator}] PCK@{t}px: {pck_value:.2f}%")
    
    # 위성 영상 특화 지표 (10px, 20px 기준)
    print(f"\nSatellite Image Metrics (relaxed thresholds):")
    for t in [10, 20]:
        print(f"  PCK@{t}px: {results[f'pck@{t}']:.2f}%")
    
    # 90% 정확도 달성 여부
    print("\n" + "="*60)
    print("Accuracy Target Check (>= 90%)")
    print("="*60)
    target_achieved = False
    for t in [1, 3, 5, 10, 20]:
        if results[f'pck@{t}'] >= 90.0:
            print(f"  ✓ PCK@{t}px >= 90%: {results[f'pck@{t}']:.2f}%")
            target_achieved = True
    
    if not target_achieved:
        print("  ✗ 90% 정확도 미달성")
        # 몇 픽셀에서 90% 달성하는지 계산
        for threshold in range(1, 101):
            pck = np.mean(all_errors < threshold) * 100
            if pck >= 90.0:
                print(f"  → PCK@{threshold}px에서 90% 달성: {pck:.2f}%")
                results['threshold_for_90'] = threshold
                break
    
    print("="*60)
    
    # 결과 저장
    if save_results:
        import json
        output_file = os.path.join(data_dir, 'benchmark_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    print("Starting benchmark script...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data_neonsat")
    parser.add_argument("--dinov2_weights", type=str, default=None, help="Path to dinov2 weights or HF model ID")
    parser.add_argument("--backbone", type=str, default=None, help="Backbone type (e.g., dinov2, dinov3/vit)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pairs to process")
    args = parser.parse_args()
    
    print(f"Data directory: {args.data_dir}")
    if not os.path.exists(args.data_dir):
        print(f"Error: {args.data_dir} does not exist.")
        exit(1)

    print("Importing torch...")
    try:
        import torch
        print(f"Torch version: {torch.__version__}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
    except Exception as e:
        print(f"Error importing torch: {e}")
        exit(1)

    print("Importing roma_model...")
    try:
        from romatch.models.model_zoo import roma_outdoor
    except Exception as e:
        print(f"Error importing roma_outdoor: {e}")
        exit(1)

    # Initialize model
    # If backbone is provided, pass it. Otherwise default behavior.
    kwargs = {}
    if args.dinov2_weights:
        kwargs['dinov2_weights'] = args.dinov2_weights
    if args.backbone:
         kwargs['backbone'] = args.backbone
         
    print("Loading model...")
    # roma_outdoor loads weights automatically
    # use_custom_corr=False is forced because GPU/CUDA kernel is likely unavailable or failing
    model = roma_outdoor(device=device, upsample_preds=True, use_custom_corr=False, **kwargs)
    print("Model loaded.")
    
    run_benchmark(args.data_dir, model, device, limit=args.limit)