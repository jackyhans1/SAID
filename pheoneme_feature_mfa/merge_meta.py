import pandas as pd

# 파일 경로 설정
mfa_features_path = "/data/alc_jihan/extracted_features_mfa/mfa_features.csv"
metadata_path = "/data/alc_jihan/extracted_features_whisper_disfluency/merged_data_disflency_meta_data.csv"
output_path = "/data/alc_jihan/extracted_features_mfa/final_mfa_features.csv"

# 기존 MFA feature CSV 파일 읽기
features_df = pd.read_csv(mfa_features_path)

# 메타데이터 CSV 파일 읽기 및 필요한 컬럼만 선택
metadata_df = pd.read_csv(metadata_path)
metadata_subset = metadata_df[['FileName', 'Class', 'Split', 'WEI', 'HEI', 'Task']]

# FileName을 기준으로 두 데이터프레임 병합 (왼쪽 기준으로 features_df에 맞게)
merged_df = pd.merge(features_df, metadata_subset, on='FileName', how='left')

# 병합된 결과를 최종 CSV 파일로 저장
merged_df.to_csv(output_path, index=False, encoding='utf-8')

print(f"Final merged features saved to {output_path}")
