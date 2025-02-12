import pandas as pd

# 파일 경로 설정
emu_feature_path = '/data/alc_jihan/extracted_features_whisper_disfluency/emu_feature.csv'
put_together_path = '/data/alc_jihan/extracted_features_whisper_disfluency/whisper_meta_put_together.csv'
output_path = '/data/alc_jihan/extracted_features_whisper_disfluency/whisper_meta_emu_final.csv'

# CSV 파일 읽기
df_emu = pd.read_csv(emu_feature_path)
df_put = pd.read_csv(put_together_path)

# df_emu를 기준으로 병합(how='left')
merged_df = pd.merge(df_emu, df_put, on='Filename', how='left')

# 결과 CSV 파일로 저장
merged_df.to_csv(output_path, index=False, encoding='utf-8')

print(f"CSV 파일이 생성되었습니다: {output_path}")
