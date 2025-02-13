import pandas as pd

# 입력 및 출력 파일 경로 설정
input_csv = '/data/alc_jihan/split_index/dataset_split_sliced.csv'
output_csv = '/data/alc_jihan/split_index/merged_data.csv'

# CSV 파일 읽기
df = pd.read_csv(input_csv)

# FileName 컬럼에서 마지막 '_'를 기준으로 자르기
df['FileName'] = df['FileName'].apply(lambda x: x.rsplit('_', 1)[0])

# 같은 FileName에 대해 나머지 값들이 동일하므로 중복 제거
df = df.drop_duplicates()

# 결과를 새로운 CSV 파일로 저장 (index 제외)
df.to_csv(output_csv, index=False)

print(f"Merged CSV saved to: {output_csv}")
