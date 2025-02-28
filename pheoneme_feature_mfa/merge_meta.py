import pandas as pd

mfa_features_path = "/data/alc_jihan/extracted_features_mfa/mfa_features.csv"
metadata_path = "/data/alc_jihan/extracted_features_whisper_disfluency/merged_data_disflency_meta_data.csv"
output_path = "/data/alc_jihan/extracted_features_mfa/final_mfa_features.csv"

features_df = pd.read_csv(mfa_features_path)

metadata_df = pd.read_csv(metadata_path)
metadata_subset = metadata_df[['FileName', 'Class', 'Split', 'WEI', 'HEI', 'Task']]

# FileName을 기준으로 두 데이터프레임 병합 (왼쪽 기준으로 features_df에 맞게)
merged_df = pd.merge(features_df, metadata_subset, on='FileName', how='left')

merged_df.to_csv(output_path, index=False, encoding='utf-8')

print(f"Final merged features saved to {output_path}")
