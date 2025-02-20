import os
import glob
import torch
import matplotlib.pyplot as plt

# 입력 및 출력 경로 설정
input_folder = '/data/alc_jihan/hubert_meta_disfluency_feature_fusion'
output_folder = '/home/ai/said/ensemble_1d_Swin_disfluency'
os.makedirs(output_folder, exist_ok=True)

# .pt 파일 목록 가져오기
pt_files = glob.glob(os.path.join(input_folder, '*.pt'))

# 각 파일의 T 길이를 저장할 리스트
t_values = []

for pt_file in pt_files:
    # 텐서 로드 (shape: [1, T, 1031])
    tensor = torch.load(pt_file)
    # 두번째 차원(T)의 길이를 추출
    T = tensor.shape[1]
    t_values.append(T)

# 히스토그램 그리기
plt.figure(figsize=(10, 6))
plt.hist(t_values, bins=50, edgecolor='black')
plt.xlabel("T (두번째 차원 길이)")
plt.ylabel("파일 수")
plt.title("각 .pt 파일의 T 길이 분포")
plt.grid(True)

# 이미지 저장
output_path = os.path.join(output_folder, "T_distribution_histogram.png")
plt.savefig(output_path)
plt.close()

print(f"히스토그램이 {output_path} 에 저장되었습니다.")
