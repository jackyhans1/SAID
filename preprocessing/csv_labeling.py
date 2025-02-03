import pandas as pd

# CSV 파일 불러오기
file_path = '/data/ALC/TABLE/SESSEXT.csv'
df = pd.read_csv(file_path)

# BAK 값을 기준으로 LAB 값 추가
def assign_lab(bak):
    if bak < 0.0005:
        return 0
    elif 0.0005 <= bak < 0.0008:
        return 1
    elif 0.0008 <= bak < 0.0010:
        return 2
    elif 0.0010 <= bak < 0.0012:
        return 3
    else:
        return 4

df['LAB'] = df['BAK'].apply(assign_lab)

# SES 값이 5000 미만인 데이터 필터링
filtered_df = df[df['SES'] < 5000]

# LAB 값의 갯수 출력
lab_counts = filtered_df['LAB'].value_counts()
print("SES 값이 5000 미만일 때 LAB 값의 갯수:")
for lab_value in sorted(lab_counts.index):
    print(f"LAB {lab_value}: {lab_counts[lab_value]}개")

# CSV 파일 저장
df.to_csv(file_path, index=False)
