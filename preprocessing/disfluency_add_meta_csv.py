import pandas as pd

# 파일 경로 정의
speaext_path = "/data/ALC/TABLE/SPEAEXT.csv"
disfluency_path = "/data/alc_jihan/extracted_features_whisper_disfluency/merged_data_disflency_whisper.csv"
output_path = "/data/alc_jihan/extracted_features_whisper_disfluency/merged_data_disflency_meta_data.csv"

# SPEAEXT.csv 데이터 로드
speaext_df = pd.read_csv(speaext_path)

# Disfluency feature 데이터 로드
disfluency_df = pd.read_csv(disfluency_path)

# SCD와 SubjectID를 문자열로 변환하여 매칭이 정확히 되도록 처리
speaext_df["SCD"] = speaext_df["SCD"].astype(str)
disfluency_df["SubjectID"] = disfluency_df["SubjectID"].astype(str)

# SPEAEXT에서 필요한 컬럼만 선택 (SCD를 기준으로 조인)
columns_to_add = ["SEX", "AGE", "WEI", "HEI", "SMO", "DRH"]
speaext_filtered_df = speaext_df[["SCD"] + columns_to_add]

# 데이터 병합 (Left Join 방식 사용)
merged_df = disfluency_df.merge(speaext_filtered_df, left_on="SubjectID", right_on="SCD", how="left")

# 필요 없는 SCD 컬럼 제거
merged_df.drop(columns=["SCD"], inplace=True)

# 결과 저장
merged_df.to_csv(output_path, index=False)

print(f"파일이 성공적으로 저장되었습니다: {output_path}")
