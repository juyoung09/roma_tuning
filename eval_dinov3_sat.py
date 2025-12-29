import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm

# ==========================================
# ⚙️ 설정 (Configuration)
# ==========================================
# 위성 특화 모델 ID
MODEL_ID = "facebook/dinov3-vitl16-pretrain-sat493m"

# 데이터가 있는 폴더 경로 (사용자 환경에 맞게 수정)
DATA_DIR = "./data_neonsat"

# 이미지 크기 (DINO 입력 크기, 14 또는 16의 배수여야 함)
IMG_SIZE = (1024, 1024) 
PATCH_SIZE = 16 # DINOv3-L은 보통 16, DINOv2-L은 14

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================

class SatDINOMatcher:
    def __init__(self, model_id, device=DEVICE):
        print(f"Loading Satellite Model: {model_id}...")
        self.device = device
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
        self.model.eval()
        
        # 전처리기는 모델 설정을 따르거나 기본 ImageNet 통계 사용
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        except:
            print("Warning: Could not load processor, using manual normalization.")
            self.processor = None

    def preprocess(self, img_path):
        image = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        
        if self.processor:
            # Processor가 이미지를 다시 리사이즈하지 않도록 설정
            inputs = self.processor(images=image, do_resize=False, do_center_crop=False, return_tensors="pt")
            return inputs["pixel_values"].to(self.device)
        else:
            # Fallback: Manual Normalization
            img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
            return ((img_tensor - mean) / std).unsqueeze(0)

    def extract_features(self, img_tensor):
        with torch.no_grad():
            outputs = self.model(pixel_values=img_tensor, output_hidden_states=True)
            # 마지막 레이어의 특징 사용
            last_hidden_state = outputs.last_hidden_state
            
            # CLS 토큰 등이 있는지 확인 (보통 첫 번째 토큰)
            # DINOv3/v2 ViT는 (B, N_patches + Prefix, Dim) 형태
            # 순수 패치만 추출
            grid_h = IMG_SIZE[0] // PATCH_SIZE
            grid_w = IMG_SIZE[1] // PATCH_SIZE
            num_patches = grid_h * grid_w
            
            seq_len = last_hidden_state.shape[1]
            if seq_len != num_patches + 1:
                # 디버깅: 토큰 길이 불일치 시 출력 (첫 1회만 출력하도록 제어 가능하나 여기선 매번 체크)
                # print(f"Debug: seq_len={seq_len}, num_patches={num_patches}")
                pass

            # 일반적인 경우: [CLS, P_1, ..., P_N] (길이 = N+1)
            # Register 토큰이 있는 경우: [CLS, P_1, ..., P_N, R_1, ..., R_4] (길이 = N+1+4) 혹은 [CLS, R..., P...]
            
            # Dinov2 with registers implementation often puts registers AFTER CLS, BEFORE patches, OR AFTER patches.
            # However, if we blindly take last num_patches, we might take registers if they are at the end.
            
            # shape 체크로 heuristic 적용
            if seq_len == num_patches + 1:
                features = last_hidden_state[:, 1:, :]
            elif seq_len == num_patches + 1 + 4: # Registers likely present
                 # facebook/dinov2-with-registers usually: CLS, PATCHES, REGISTERS
                 # But let's verify. usually registers are appended.
                 # If appended: [CLS, P...P, R...R]
                 features = last_hidden_state[:, 1:num_patches+1, :]
            else:
                # Fallback: 뒤에서부터 가져오기 (기존 로직) 그러나 위험함
                features = last_hidden_state[:, -num_patches:, :]
            
            # (B, N, C) -> (B, H, W, C)
            features = features.reshape(1, grid_h, grid_w, -1)
            return features

def parse_keypoints(txt_path):
    """
    텍스트 파일에서 키포인트 파싱
    형식: ID X_coord Y_coord
    """
    keypoints = {}
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3: continue
        
        # '', '1', '830.86...', '676.91...' 형태 처리
        try:
            # ID가 숫자인지 확인 (인덱스 위치 찾기)
            idx_start = 0
            for i, p in enumerate(parts):
                if p.replace(',', '').isdigit() and i > 0: # source 뒤에 오는 첫 숫자
                    idx_start = i
                    break
            
            kp_id = int(parts[idx_start])
            # 쉼표 제거 및 float 변환
            x = float(parts[idx_start+1].replace(',', ''))
            y = float(parts[idx_start+2].replace(',', ''))
            keypoints[kp_id] = (x, y)
        except ValueError:
            continue
            
    return keypoints

def evaluate_dataset():
    matcher = SatDINOMatcher(MODEL_ID)
    
    # 데이터 파일 쌍 찾기 (규칙에 따라 수정 필요)
    # 여기서는 .txt 파일들을 찾고 대응되는 이미지 매칭
    txt_files = sorted(glob.glob(os.path.join(DATA_DIR, "labels", "*.txt")))
    
    # K3A(Source 1)와 N01(Source 2) 파일 쌍을 구분해야 함
    # 파일명 예: coreg_00001_K3A....txt / coreg_00001_N01....txt / coreg_00021_K3....txt
    # 접두사(coreg_00001)로 그룹화
    pairs = {}
    for f in txt_files:
        basename = os.path.basename(f)
        # prefix = basename.split("_K3A")[0].split("_N01")[0] 
        # 더 강건한 접두사 추출 (coreg_XXXXX 형태 가정)
        parts = basename.split("_")
        if len(parts) >= 2:
            prefix = f"{parts[0]}_{parts[1]}"
        else:
            # fallback
            prefix = basename.split("_")[0]
        
        if prefix not in pairs:
            pairs[prefix] = {}
            
        if "K3" in basename: # K3, K3A 모두 매칭
            pairs[prefix]['A'] = f
        elif "N01" in basename:
            pairs[prefix]['B'] = f
            
    print(f"Found {len(pairs)} pairs.")
    
    errors = []
    pck_thresholds = [5, 10, 20] # pixels
    pck_hits = {t: 0 for t in pck_thresholds}
    total_kps = 0

    for prefix, files in tqdm(pairs.items(), desc="Evaluating"):
        if 'A' not in files or 'B' not in files:
            continue
            
        # 1. 경로 설정
        txt_a = files['A']
        txt_b = files['B']
        
        def get_image_path(label_path):
            dir_name = os.path.dirname(label_path)
            base_name = os.path.splitext(os.path.basename(label_path))[0]
            
            # labels -> images 폴더 변경
            if dir_name.endswith("labels"):
                img_dir = dir_name.replace("labels", "images")
            elif "labels" in dir_name:
                 img_dir = dir_name.replace("/labels/", "/images/")
            else:
                img_dir = dir_name
                
            png_path = os.path.join(img_dir, base_name + ".png")
            jpg_path = os.path.join(img_dir, base_name + ".jpg")
            
            if os.path.exists(png_path): return png_path
            if os.path.exists(jpg_path): return jpg_path
            return png_path
            
        img_a_path = get_image_path(txt_a)
        img_b_path = get_image_path(txt_b)
        
        if not os.path.exists(img_a_path) or not os.path.exists(img_b_path):
            print(f"\n[Skip] Missing images for {prefix}")
            if not os.path.exists(img_a_path):
                print(f"  - Not found A: {img_a_path}")
            if not os.path.exists(img_b_path):
                print(f"  - Not found B: {img_b_path}")
            continue

        # 2. GT 로드
        kps_a_dict = parse_keypoints(txt_a)
        kps_b_dict = parse_keypoints(txt_b)
        
        # 공통 ID 찾기
        common_ids = set(kps_a_dict.keys()) & set(kps_b_dict.keys())
        if not common_ids: continue
        
        # 3. 모델 특징 추출
        tensor_a = matcher.preprocess(img_a_path)
        tensor_b = matcher.preprocess(img_b_path)
        
        feats_a = matcher.extract_features(tensor_a) # (1, H, W, C)
        feats_b = matcher.extract_features(tensor_b) # (1, H, W, C)
        
        H_grid, W_grid = feats_a.shape[1], feats_a.shape[2]
        
        # Flatten for similarity calculation
        flat_feats_a = feats_a.reshape(-1, feats_a.shape[-1]) # (N_patches, C)
        flat_feats_b = feats_b.reshape(-1, feats_b.shape[-1])
        
        # 정규화 (Cosine Similarity)
        flat_feats_a = F.normalize(flat_feats_a, dim=-1)
        flat_feats_b = F.normalize(flat_feats_b, dim=-1)
        
        # 4. 매칭 및 에러 계산
        for kp_id in common_ids:
            total_kps += 1
            xa, ya = kps_a_dict[kp_id]
            xb_gt, yb_gt = kps_b_dict[kp_id]
            
            # A 좌표를 그리드 좌표로 변환
            grid_y = int(ya // PATCH_SIZE)
            grid_x = int(xa // PATCH_SIZE)
            
            # 범위 체크
            grid_y = min(max(grid_y, 0), H_grid - 1)
            grid_x = min(max(grid_x, 0), W_grid - 1)
            
            patch_idx_a = grid_y * W_grid + grid_x
            
            # 해당 패치의 특징 벡터 가져오기
            query_vec = flat_feats_a[patch_idx_a].unsqueeze(0) # (1, C)
            
            # B 이미지의 모든 패치와 유사도 비교
            # (1, C) @ (C, N) -> (1, N)
            sim_scores = torch.matmul(query_vec, flat_feats_b.T)
            best_match_idx = torch.argmax(sim_scores).item()
            
            # 예측된 패치 인덱스를 픽셀 좌표로 변환 (패치 중심)
            pred_grid_y = best_match_idx // W_grid
            pred_grid_x = best_match_idx % W_grid
            
            pred_x = pred_grid_x * PATCH_SIZE + (PATCH_SIZE / 2)
            pred_y = pred_grid_y * PATCH_SIZE + (PATCH_SIZE / 2)
            
            # 에러 계산 (L2 distance)
            dist = np.sqrt((pred_x - xb_gt)**2 + (pred_y - yb_gt)**2)
            errors.append(dist)
            
            for t in pck_thresholds:
                if dist <= t:
                    pck_hits[t] += 1

    # ==========================================
    # 📊 결과 출력
    # ==========================================
    if not errors:
        print("No matches evaluated.")
        return

    print("\n" + "="*50)
    print(f"Results using Raw Features from: {MODEL_ID}")
    print("Method: Nearest Neighbor Feature Matching (Patch-level)")
    print("="*50)
    print(f"Total Keypoints: {total_kps}")
    print(f"Mean Error: {np.mean(errors):.2f} px")
    print(f"Median Error: {np.median(errors):.2f} px")
    print("-" * 30)
    print("PCK Metrics (Note: Patch size limits precision):")
    for t in pck_thresholds:
        pck = (pck_hits[t] / total_kps) * 100
        print(f"  PCK @ {t}px: {pck:.2f}%")
    print("="*50)
    print("Note: Since this is patch-level matching without a refined head,")
    print(f"theoretical minimum error is limited by patch size ({PATCH_SIZE}px).")
    print("Values < 16px are extremely good for this method.")

if __name__ == "__main__":
    evaluate_dataset()