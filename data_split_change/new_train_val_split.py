import pandas as pd
from sklearn.model_selection import train_test_split

# 경로 설정
input_path = "/data/alc_jihan/split_index/merged_data.csv"
output_path = "/data/alc_jihan/split_index/merged_data_new_split.csv"

# CSV 로드 및 'Split' 열 제거
df = pd.read_csv(input_path)
if 'Split' in df.columns:
    df = df.drop(columns=['Split'])

# 피실험자 ID 기준 분할
subject_ids = df["SubjectID"].unique()
train_subjects, temp_subjects = train_test_split(subject_ids, test_size=0.4, random_state=42)
val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=42)

# SubjectID에 따라 데이터 분리
def split_by_subject(df, train_subjects, val_subjects, test_subjects):
    train_data = df[df["SubjectID"].isin(train_subjects)].copy()
    val_data = df[df["SubjectID"].isin(val_subjects)].copy()
    test_data = df[df["SubjectID"].isin(test_subjects)].copy()
    return train_data, val_data, test_data

train_data, val_data, test_data = split_by_subject(df, train_subjects, val_subjects, test_subjects)

# Split 열 추가
train_data["Split"] = "train"
val_data["Split"] = "val"
test_data["Split"] = "test"

# 병합 및 저장
final_df = pd.concat([train_data, val_data, test_data], ignore_index=True)
final_df.to_csv(output_path, index=False)

# 통계 출력
print(f"Saved updated CSV with new 'Split' column to: {output_path}")
print(f"Train samples: {len(train_data)}")
print(f"Val samples  : {len(val_data)}")
print(f"Test samples : {len(test_data)}")
