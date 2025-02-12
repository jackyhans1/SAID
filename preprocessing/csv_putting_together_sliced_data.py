import pandas as pd

# 파일 경로 설정
input_csv = '/data/alc_jihan/extracted_features_whisper_disfluency/all_data_Disfluency_and_metadata.csv'
output_csv = '/data/alc_jihan/extracted_features_whisper_disfluency/whisper_meta_put_together.csv'

# CSV 파일 읽기
df = pd.read_csv(input_csv)

# Filename에서 마지막 '_' 이후의 부분을 제거하여 기본 파일명(BaseFilename) 컬럼 생성
df['BaseFilename'] = df['Filename'].apply(lambda x: '_'.join(x.split('_')[:-1]))

# 합산할 숫자형 컬럼들 (각 그룹에서 합산)
numeric_cols = ['Len_S', 'Var_S', 'Avg_S', 'Max_S', 'Rat_S', 'Num_S', 'Num_PS', 'Avg_PS', 'Var_PS']

# 그룹 내에서 동일할 것으로 기대되는 기타 메타데이터 컬럼 (그룹당 아무거나 하나 선택)
const_cols = ['SubjectID', 'Class', 'Split', 'Task', 'SEX', 'AGE', 'WEI', 'HEI', 'SMO', 'DRH']

# 그룹별로 집계: 숫자형 컬럼은 합계, 나머지 컬럼은 첫번째 값을 사용
aggregations = {col: 'sum' for col in numeric_cols}
aggregations.update({col: 'first' for col in const_cols})

grouped_df = df.groupby('BaseFilename', as_index=False).agg(aggregations)

# BaseFilename 컬럼을 Filename으로 이름 변경
grouped_df = grouped_df.rename(columns={'BaseFilename': 'Filename'})

# 최종 컬럼 순서 지정
final_cols = ['Filename', 'SubjectID', 'Class', 'Split'] + numeric_cols + ['Task', 'SEX', 'AGE', 'WEI', 'HEI', 'SMO', 'DRH']
grouped_df = grouped_df[final_cols]

# 결과를 CSV 파일로 저장
grouped_df.to_csv(output_csv, index=False)

print(f"파일이 생성되었습니다: {output_csv}")
