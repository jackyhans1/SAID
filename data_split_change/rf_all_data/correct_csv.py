import pandas as pd

# 파일 경로 설정
feature_path = '/data/alc_jihan/extracted_features_mfa/final_for_use.csv'
split_path   = '/data/alc_jihan/split_index/merged_data_new_split.csv'
output_path  = '/data/alc_jihan/extracted_features_mfa/final_for_use_with_split.csv'

# 파일 읽기
df_feature = pd.read_csv(feature_path)
df_split   = pd.read_csv(split_path)

# 'FileName' 기준으로 merge
df_merged = pd.merge(df_feature, df_split[['FileName', 'Split']], on='FileName', how='left')

# 결과 저장
df_merged.to_csv(output_path, index=False)
