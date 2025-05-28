import os
import glob
import numpy as np
from PIL import Image

# ----- 사용자 설정 -----
SRC_DIR  = "/data/alc_jihan/silence_black_without_edge_imgae"
DST_DIR  = "/data/alc_jihan/morphology_thresholded"
THRESH   = 0.97    

# ----------------------
os.makedirs(DST_DIR, exist_ok=True)

png_files = sorted(glob.glob(os.path.join(SRC_DIR, "*.png")))
print(f"[INFO] {len(png_files)} files found.")

for path in png_files:
    fname = os.path.basename(path)                 # e.g. 0_0062014003_h_00.png
    # 1) 이미지 로드(그레이스케일, float32로 변환)
    img = Image.open(path).convert("L")            # 'L' = 8-bit gray
    arr = np.asarray(img, dtype=np.float32) / 255. # 0~1 범위

    # 2) thresholding
    bin_mask = (arr >= THRESH).astype(np.uint8)    # 1  또는 0
    bin_img  = Image.fromarray(bin_mask * 255)     # 8-bit (0/255)

    # 3) 저장
    out_path = os.path.join(DST_DIR, fname)
    bin_img.save(out_path, format="PNG")

    print(f"  > {fname:30s}  |  thresholded → {out_path}")

print("\n[Done] 모든 이미지에 thresholding 완료!")
