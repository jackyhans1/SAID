import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 데이터 경로 설정
DATA_PATH = "/data/alc_jihan/h_wav_slided"  # 실제 데이터 경로로 수정하세요

# 파일 목록 가져오기
file_list = [f for f in os.listdir(DATA_PATH) if f.endswith(".wav")]

# 데이터 라벨 및 피실험자 ID 설정 (0: Sober, 1-4: Intoxicated)
data = []
for file_name in file_list:
    label = int(file_name.split('_')[0])  # 파일명 첫 번째 숫자로 라벨 결정
    subject_id = file_name.split('_')[1][:3]  # 피실험자 ID 추출 (예: 596)
    if label == 0:
        class_label = "Sober"
    elif label in [1, 2, 3, 4]:
        class_label = "Intoxicated"
    else:
        continue
    # 확장자 제거한 파일명 저장
    file_name_no_ext = os.path.splitext(file_name)[0]
    data.append((file_name_no_ext, subject_id, class_label))

# 데이터프레임 생성
df = pd.DataFrame(data, columns=["FileName", "SubjectID", "Class"])

# 피실험자 ID 기준으로 분리 (전체 ID 추출)
subject_ids = df["SubjectID"].unique()

# 피실험자 ID를 Train:Val:Test = 75:15:15 비율로 분리
train_subjects, temp_subjects = train_test_split(subject_ids, test_size=0.3, random_state=42)
val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=42)

# 피실험자 ID에 따라 데이터 분리
def split_by_subject(df, train_subjects, val_subjects, test_subjects):
    train_data = df[df["SubjectID"].isin(train_subjects)]
    val_data = df[df["SubjectID"].isin(val_subjects)]
    test_data = df[df["SubjectID"].isin(test_subjects)]
    return train_data, val_data, test_data

train_data, val_data, test_data = split_by_subject(df, train_subjects, val_subjects, test_subjects)

# 데이터프레임에 Split 컬럼 추가
train_data.loc[:, "Split"] = "train"
val_data.loc[:, "Split"] = "val"
test_data.loc[:, "Split"] = "test"

# 최종 데이터 통합
final_df = pd.concat([train_data, val_data, test_data])

# CSV 저장
output_csv_path = "/data/alc_jihan/split_index/dataset_split_slided.csv"  # 원하는 경로로 수정하세요
final_df.to_csv(output_csv_path, index=False, header=["FileName", "SubjectID", "Class", "Split"])

print(f"CSV 파일이 저장되었습니다: {output_csv_path}")
