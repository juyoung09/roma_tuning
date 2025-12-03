#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeonSAT 벤치마크: DINOv2 vs DINOv3 성능 비교

기존 RoMa (DINOv2)와 DINOv3로 교체한 모델의 성능을 비교합니다.

사용법:
    python eval_neonsat_comparison.py --data_dir data_neonsat

DINOv3 모델 옵션:
    - sat-7b: facebook/dinov3-vit7b16-pretrain-sat493m (위성 전용 7B, 권장)
    - sat-large: facebook/dinov3-vitl16-pretrain-sat493m (위성 전용 Large)
"""

import os
import argparse
import json
import time
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


def run_comparison(data_dir, limit=None, dinov3_model="sat-7b", save_results=True):
    """
    DINOv2 vs DINOv3 성능 비교 실행
    """
    from romatch import roma_outdoor, roma_outdoor_dinov3
    from romatch.benchmarks.neonsat_benchmark import run_benchmark
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print("="*70)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'data_dir': data_dir,
        'limit': limit,
        'device': str(device),
    }
    
    # ========================================
    # 1. 기존 RoMa (DINOv2) 평가
    # ========================================
    print("\n" + "="*70)
    print("1. RoMa with DINOv2 (baseline)")
    print("="*70)
    
    try:
        print("Loading RoMa + DINOv2 model...")
        start_time = time.time()
        model_dinov2 = roma_outdoor(
            device=device,
            upsample_preds=True,
            use_custom_corr=False,  # 호환성을 위해
        )
        load_time_v2 = time.time() - start_time
        print(f"Model loaded in {load_time_v2:.2f}s")
        
        print("\nRunning benchmark...")
        results_v2 = run_benchmark(
            data_dir, 
            model_dinov2, 
            device=str(device), 
            limit=limit,
            visualize=True,
            save_results=False,
        )
        results['dinov2'] = results_v2
        results['dinov2']['load_time'] = load_time_v2
        
        # 메모리 정리
        del model_dinov2
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error with DINOv2: {e}")
        results['dinov2'] = {'error': str(e)}
    
    # ========================================
    # 2. RoMa + DINOv3 평가
    # ========================================
    print("\n" + "="*70)
    print(f"2. RoMa with DINOv3 ({dinov3_model})")
    print("="*70)
    
    try:
        print(f"Loading RoMa + DINOv3 model ({dinov3_model})...")
        start_time = time.time()
        model_dinov3 = roma_outdoor_dinov3(
            device=device,
            dinov3_model=dinov3_model,
            upsample_preds=True,
            use_custom_corr=False,
        )
        load_time_v3 = time.time() - start_time
        print(f"Model loaded in {load_time_v3:.2f}s")
        
        print("\nRunning benchmark...")
        results_v3 = run_benchmark(
            data_dir, 
            model_dinov3, 
            device=str(device), 
            limit=limit,
            visualize=True,
            save_results=False,
        )
        results['dinov3'] = results_v3
        results['dinov3']['load_time'] = load_time_v3
        results['dinov3']['model_name'] = dinov3_model
        
        # 메모리 정리
        del model_dinov3
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error with DINOv3: {e}")
        import traceback
        traceback.print_exc()
        results['dinov3'] = {'error': str(e)}
    
    # ========================================
    # 3. 결과 비교
    # ========================================
    print("\n" + "="*70)
    print("Comparison: DINOv2 vs DINOv3")
    print("="*70)
    
    if 'error' not in results.get('dinov2', {}) and 'error' not in results.get('dinov3', {}):
        v2 = results['dinov2']
        v3 = results['dinov3']
        
        print("\n┌" + "─"*68 + "┐")
        print("│{:^68}│".format("Performance Comparison"))
        print("├" + "─"*20 + "┬" + "─"*22 + "┬" + "─"*23 + "┤")
        print("│{:^20}│{:^22}│{:^23}│".format("Metric", "DINOv2", "DINOv3"))
        print("├" + "─"*20 + "┼" + "─"*22 + "┼" + "─"*23 + "┤")
        
        # Mean Error
        v2_err = v2.get('mean_error', float('inf'))
        v3_err = v3.get('mean_error', float('inf'))
        diff = v3_err - v2_err
        print("│{:^20}│{:^22}│{:^23}│".format(
            "Mean Error (px)", 
            f"{v2_err:.2f}", 
            f"{v3_err:.2f} ({diff:+.2f})"
        ))
        
        # Median Error
        v2_med = v2.get('median_error', float('inf'))
        v3_med = v3.get('median_error', float('inf'))
        diff = v3_med - v2_med
        print("│{:^20}│{:^22}│{:^23}│".format(
            "Median Error (px)", 
            f"{v2_med:.2f}", 
            f"{v3_med:.2f} ({diff:+.2f})"
        ))
        
        print("├" + "─"*20 + "┼" + "─"*22 + "┼" + "─"*23 + "┤")
        
        # PCK values
        for t in [1, 3, 5, 10, 20]:
            key = f'pck@{t}'
            v2_pck = v2.get(key, 0)
            v3_pck = v3.get(key, 0)
            diff = v3_pck - v2_pck
            
            # 90% 이상이면 체크마크
            v2_mark = "✓" if v2_pck >= 90 else " "
            v3_mark = "✓" if v3_pck >= 90 else " "
            
            print("│{:^20}│{:^22}│{:^23}│".format(
                f"PCK@{t}px", 
                f"{v2_mark} {v2_pck:.2f}%", 
                f"{v3_mark} {v3_pck:.2f}% ({diff:+.2f})"
            ))
        
        print("├" + "─"*20 + "┼" + "─"*22 + "┼" + "─"*23 + "┤")
        
        # Load time
        print("│{:^20}│{:^22}│{:^23}│".format(
            "Load Time (s)", 
            f"{v2.get('load_time', 0):.2f}", 
            f"{v3.get('load_time', 0):.2f}"
        ))
        
        print("└" + "─"*20 + "┴" + "─"*22 + "┴" + "─"*23 + "┘")
        
        # 결론
        print("\n" + "="*70)
        print("Summary")
        print("="*70)
        
        # 어느 모델이 더 좋은지 판단
        v3_better_count = 0
        for t in [1, 3, 5, 10, 20]:
            if v3.get(f'pck@{t}', 0) > v2.get(f'pck@{t}', 0):
                v3_better_count += 1
        
        if v3_better_count >= 3:
            print("✓ DINOv3가 대부분의 지표에서 우수합니다.")
        elif v3_better_count <= 2:
            print("✓ DINOv2가 대부분의 지표에서 우수합니다.")
        else:
            print("△ 두 모델의 성능이 비슷합니다.")
        
        # 90% 목표 달성 여부
        print("\n90% 정확도 목표 달성:")
        for model_name, res in [("DINOv2", v2), ("DINOv3", v3)]:
            achieved = False
            for t in [1, 3, 5, 10, 20]:
                if res.get(f'pck@{t}', 0) >= 90:
                    print(f"  {model_name}: ✓ PCK@{t}px에서 달성 ({res.get(f'pck@{t}', 0):.2f}%)")
                    achieved = True
                    break
            if not achieved:
                # 몇 픽셀에서 달성하는지 확인
                threshold_90 = res.get('threshold_for_90', 'N/A')
                print(f"  {model_name}: ✗ 미달성 (90% 달성 임계값: {threshold_90}px)")
    
    else:
        print("One or both models failed. Check errors above.")
    
    # ========================================
    # 4. 결과 저장
    # ========================================
    if save_results:
        output_file = os.path.join(data_dir, f'comparison_dinov2_vs_dinov3_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(output_file, 'w') as f:
            # numpy 타입을 JSON serializable하게 변환
            def convert(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(results, f, indent=2, default=convert)
        print(f"\nResults saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="NeonSAT Benchmark: DINOv2 vs DINOv3")
    parser.add_argument("--data_dir", type=str, default="data_neonsat",
                        help="Path to NeonSAT data directory")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of pairs to process (for quick testing)")
    parser.add_argument("--dinov3_model", type=str, default="sat-7b",
                        choices=["sat-7b", "sat-large", "web-large", "web-base"],
                        help="DINOv3 model to use")
    parser.add_argument("--no_save", action="store_true",
                        help="Don't save results to file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        return
    
    run_comparison(
        data_dir=args.data_dir,
        limit=args.limit,
        dinov3_model=args.dinov3_model,
        save_results=not args.no_save,
    )


if __name__ == "__main__":
    main()

