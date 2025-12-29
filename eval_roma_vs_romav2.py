#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeonSAT 벤치마크: RoMa vs RoMaV2 성능 비교

기존 RoMa (DINOv2 기반)와 차세대 RoMaV2의 성능을 비교합니다.
이 스크립트는 메모리 할당 오류(std::bad_alloc)를 방지하기 위해 
각 모델 평가를 별도의 서브프로세스에서 실행합니다.

사용법:
    python eval_roma_vs_romav2.py --data_dir data_neonsat
"""

import os
import sys
import argparse
import json
import time
import subprocess
import tempfile
from datetime import datetime
import numpy as np

# -----------------------------------------------------------------------------
# 메인 비교 로직
# -----------------------------------------------------------------------------

def run_subprocess_eval(model_type, data_dir, limit=None, romav2_path=None):
    """
    별도의 프로세스에서 단일 모델 평가를 실행하고 결과를 반환합니다.
    """
    print("\n" + "="*70)
    print(f"Running {model_type} evaluation in a subprocess...")
    print("="*70)
    
    # 현재 스크립트 경로
    script_path = os.path.abspath(__file__)
    
    # 임시 결과 파일 생성
    fd, temp_output = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    
    # 명령어 구성
    cmd = [
        sys.executable, script_path,
        '--mode', 'single',
        '--model', model_type,
        '--data_dir', data_dir,
        '--output', temp_output
    ]
    
    if limit is not None:
        cmd.extend(['--limit', str(limit)])
    
    if romav2_path is not None:
        cmd.extend(['--romav2_path', romav2_path])
        
    # 서브프로세스 실행
    start_time = time.time()
    try:
        # 환경 변수 상속 및 PYTHONPATH 추가
        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH", "")
        # 현재 디렉토리를 PYTHONPATH에 추가
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + current_pythonpath
        
        # 중요: MKL 및 OpenMP 충돌 방지, 메모리 할당 문제 해결 시도
        env["MKL_THREADING_LAYER"] = "GNU"
        env["OMP_NUM_THREADS"] = "8"  # 스레드 수 제한
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512" # 메모리 파편화 방지
        
        # 실시간 출력을 위해 subprocess.run 대신 Popen 사용 고려 가능하지만,
        # 간단하게 run을 사용하고 stdout을 상속받음
        result = subprocess.run(cmd, check=True, env=env)
        
        # 결과 로드
        if os.path.exists(temp_output):
            with open(temp_output, 'r') as f:
                eval_results = json.load(f)
            os.remove(temp_output)
            return eval_results
        else:
            print(f"Error: Output file not found for {model_type}")
            return {'error': 'Output file not found'}
            
    except subprocess.CalledProcessError as e:
        print(f"Error running {model_type} evaluation: {e}")
        return {'error': str(e)}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {'error': str(e)}

def run_comparison_main(data_dir, limit=None, save_results=True, romav2_path=None):
    """
    메인 비교 실행 함수
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'data_dir': data_dir,
        'limit': limit,
    }
    
    # 1. RoMaV2 평가 (서브프로세스) - 먼저 실행
    results['romav2'] = run_subprocess_eval('romav2', data_dir, limit, romav2_path)
    
    # 2. RoMa 평가 (서브프로세스)
    results['roma'] = run_subprocess_eval('roma', data_dir, limit, romav2_path)
    
    # 3. 결과 비교 및 출력
    print_comparison_table(results)
    
    # 4. 결과 저장
    if save_results:
        output_file = os.path.join(data_dir, f'comparison_roma_vs_romav2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

def print_comparison_table(results):
    """
    비교 결과 테이블 출력
    """
    print("\n" + "="*70)
    print("Comparison: RoMa vs RoMaV2")
    print("="*70)
    
    v1 = results.get('roma', {})
    v2 = results.get('romav2', {})
    
    if 'error' in v1 or 'error' in v2:
        print("One or both models failed. Check errors above.")
        if 'error' in v1: print(f"RoMa Error: {v1['error']}")
        if 'error' in v2: print(f"RoMaV2 Error: {v2['error']}")
        return

    print("\n┌" + "─"*68 + "┐")
    print("│{:^68}│".format("Performance Comparison"))
    print("├" + "─"*20 + "┬" + "─"*22 + "┬" + "─"*23 + "┤")
    print("│{:^20}│{:^22}│{:^23}│".format("Metric", "RoMa", "RoMaV2"))
    print("├" + "─"*20 + "┼" + "─"*22 + "┼" + "─"*23 + "┤")
    
    # Mean Error
    v1_err = v1.get('mean_error', float('inf'))
    v2_err = v2.get('mean_error', float('inf'))
    diff = v2_err - v1_err
    improvement = "▼" if diff < 0 else "▲" if diff > 0 else "="
    print("│{:^20}│{:^22}│{:^23}│".format(
        "Mean Error (px)", 
        f"{v1_err:.2f}", 
        f"{v2_err:.2f} ({improvement}{abs(diff):.2f})"
    ))
    
    # Median Error
    v1_med = v1.get('median_error', float('inf'))
    v2_med = v2.get('median_error', float('inf'))
    diff = v2_med - v1_med
    improvement = "▼" if diff < 0 else "▲" if diff > 0 else "="
    print("│{:^20}│{:^22}│{:^23}│".format(
        "Median Error (px)", 
        f"{v1_med:.2f}", 
        f"{v2_med:.2f} ({improvement}{abs(diff):.2f})"
    ))
    
    print("├" + "─"*20 + "┼" + "─"*22 + "┼" + "─"*23 + "┤")
    
    # PCK values
    for t in [1, 3, 5, 10, 20]:
        key = f'pck@{t}'
        v1_pck = v1.get(key, 0)
        v2_pck = v2.get(key, 0)
        diff = v2_pck - v1_pck
        
        v1_mark = "✓" if v1_pck >= 90 else " "
        v2_mark = "✓" if v2_pck >= 90 else " "
        
        improvement = "▲" if diff > 0 else "▼" if diff < 0 else "="
        print("│{:^20}│{:^22}│{:^23}│".format(
            f"PCK@{t}px", 
            f"{v1_mark} {v1_pck:.2f}%", 
            f"{v2_mark} {v2_pck:.2f}% ({improvement}{abs(diff):.2f})"
        ))
    
    print("├" + "─"*20 + "┼" + "─"*22 + "┼" + "─"*23 + "┤")
    
    # Load time
    v1_time = v1.get('load_time', 0)
    v2_time = v2.get('load_time', 0)
    diff = v2_time - v1_time
    improvement = "▼" if diff < 0 else "▲" if diff > 0 else "="
    print("│{:^20}│{:^22}│{:^23}│".format(
        "Load Time (s)", 
        f"{v1_time:.2f}", 
        f"{v2_time:.2f} ({improvement}{abs(diff):.2f})"
    ))
    
    print("└" + "─"*20 + "┴" + "─"*22 + "┴" + "─"*23 + "┘")

# -----------------------------------------------------------------------------
# 단일 모델 평가 로직 (서브프로세스용)
# -----------------------------------------------------------------------------

def evaluate_single_model(model_type, data_dir, output_file, limit=None, romav2_path=None):
    """
    단일 모델을 평가하고 결과를 JSON 파일로 저장합니다.
    """
    import torch
    from romatch.benchmarks.neonsat_benchmark import run_benchmark
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Worker Process - Device: {device}, Model: {model_type}")
    
    start_time = time.time()
    model = None
    
    try:
        if model_type == 'roma':
            from romatch import roma_outdoor
            model = roma_outdoor(
                device=device,
                upsample_preds=True,
                use_custom_corr=False,
            )
        
        elif model_type == 'romav2':
            # RoMaV2 경로 설정
            if romav2_path is not None:
                sys.path.insert(0, romav2_path)
            else:
                possible_paths = [
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RoMaV2', 'src'),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'RoMaV2', 'src'),
                ]
                for p in possible_paths:
                    if os.path.exists(p):
                        sys.path.insert(0, p)
                        break
            
            from romav2 import RoMaV2
            
            # RoMaV2 래퍼 클래스 정의
            class RoMaV2Wrapper:
                def __init__(self, model):
                    self.model = model
                
                def match(self, im_A, im_B, device='cuda', *args, **kwargs):
                    with torch.no_grad():
                        preds = self.model.match(im_A, im_B)
                    
                    warp = preds['warp_AB']
                    overlap = preds['overlap_AB']
                    
                    if warp.dim() == 4: warp = warp[0]
                    if overlap.dim() == 4: overlap = overlap[0]
                    if overlap.dim() == 3 and overlap.shape[-1] == 1: overlap = overlap.squeeze(-1)
                    
                    # Create grid (identity flow) to match RoMa output format (grid, warp) -> 4 channels
                    # RoMa benchmark expects warp[..., 2:] to be the predicted coordinates in normalized range [-1, 1]
                    H, W = warp.shape[:2]
                    
                    # Grid creation
                    grid_y, grid_x = torch.meshgrid(
                        torch.linspace(-1, 1, H, device=warp.device),
                        torch.linspace(-1, 1, W, device=warp.device),
                        indexing='ij'
                    )
                    grid = torch.stack([grid_x, grid_y], dim=-1) # (H, W, 2)
                    
                    # Concatenate grid and warp to get (H, W, 4)
                    warp_out = torch.cat([grid, warp], dim=-1)
                    
                    return warp_out, overlap

            romav2_model = RoMaV2()
            romav2_model.apply_setting("precise")
            romav2_model.to(device)
            romav2_model.eval()
            model = RoMaV2Wrapper(romav2_model)
            
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s")
        
        # 벤치마크 실행
        results = run_benchmark(
            data_dir, 
            model, 
            device=str(device), 
            limit=limit,
            visualize=True,
            save_results=False,
        )
        
        results['load_time'] = load_time
        
        # 결과 저장
        def convert(obj):
            if isinstance(obj, np.floating): return float(obj)
            elif isinstance(obj, np.integer): return int(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=convert)
            
    except Exception as e:
        print(f"Error in evaluate_single_model: {e}")
        import traceback
        traceback.print_exc()
        # 에러 정보를 파일에 씀
        with open(output_file, 'w') as f:
            json.dump({'error': str(e)}, f)

# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NeonSAT Benchmark: RoMa vs RoMaV2")
    parser.add_argument("--data_dir", type=str, default="data_neonsat")
    # limit 인자는 유지하되, 아래에서 강제로 덮어씌우거나 기본값을 변경
    parser.add_argument("--limit", type=int, default=None) 
    parser.add_argument("--romav2_path", type=str, default=None)
    parser.add_argument("--no_save", action="store_true")
    
    # 내부 사용 인자
    parser.add_argument("--mode", type=str, default="main", choices=["main", "single"])
    parser.add_argument("--model", type=str, choices=["roma", "romav2"])
    parser.add_argument("--output", type=str)
    
    args = parser.parse_args()

    limit_to_pass = args.limit

    if args.mode == "single":
        # 서브프로세스 모드
        evaluate_single_model(
            model_type=args.model,
            data_dir=args.data_dir,
            output_file=args.output,
            limit=limit_to_pass,
            romav2_path=args.romav2_path
        )
    else:
        # 메인 모드
        if not os.path.exists(args.data_dir):
            print(f"Error: Data directory not found: {args.data_dir}")
            return
            
        print(f"Running benchmark with limit={limit_to_pass}")
            
        run_comparison_main(
            data_dir=args.data_dir,
            limit=limit_to_pass,
            romav2_path=args.romav2_path,
            save_results=not args.no_save,
        )

if __name__ == "__main__":
    main()
