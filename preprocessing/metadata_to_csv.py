import pandas as pd

# TBL 파일 경로 및 CSV 저장 경로
tbl_file_path = "/data/ALC/TABLE/SPEAEXT.TBL"
csv_file_path = "/data/ALC/TABLE/SPEAEXT.csv"

# 사용할 열 이름
columns_to_extract = ["SCD", "SEX", "AGE", "WEI", "HEI", "SMO", "DRH"]

# TBL 파일을 DataFrame으로 읽기 (탭 또는 공백 구분자를 자동 감지)
df = pd.read_csv(tbl_file_path, sep=r'\s+', usecols=columns_to_extract, engine='python')

# CSV 파일로 저장
df.to_csv(csv_file_path, index=False)

print(f"CSV 파일이 저장되었습니다: {csv_file_path}")
