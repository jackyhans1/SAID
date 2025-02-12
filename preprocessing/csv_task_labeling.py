import pandas as pd

# 파일 경로
input_file = "/data/alc_jihan/split_index/dataset_split_sliced.csv"
output_file = "/data/alc_jihan/split_index/dataset_split_sliced_task_indexed.csv"

# Task 매핑 정보
task_mapping = {
    "Intoxicated": {
        "monologue": {"005", "014", "018"},
        "dialogue": {"002", "010"},
        "number": {"001", "006", "009", "011", "015"},
        "read_command": {"021", "023", "024", "029"},
        "spontaneous_command": {"022", "025", "026", "027", "028"},
        "address": {"004", "008", "013", "017", "019", "030"},
        "tongue_twister": {"003", "007", "012", "016", "020"}
    },
    "Sober": {
        "monologue": {"005", "014", "018", "030", "038"},
        "dialogue": {"002", "010", "022", "025", "034"},
        "number": {"001", "006", "009", "011", "015", "021", "026", "029", "031", "035"},
        "read_command": {"041", "044", "050", "051", "052", "056", "057", "058", "059"},
        "spontaneous_command": {"042", "043", "045", "046", "047", "048", "049", "053", "054", "055"},
        "address": {"004", "008", "013", "017", "019", "024", "028", "033", "037", "039", "060"},
        "tongue_twister": {"003", "007", "012", "016", "020", "023", "027", "032", "036", "040"}
    }
}

# CSV 파일 읽기
df = pd.read_csv(input_file)

# Task 컬럼 추가
def get_task(file_name, speech_class):
    try:
        # 파일명에서 _ 기준으로 split 후 두 번째 데이터의 마지막 세 자리 숫자 추출
        last_three_digits = file_name.split("_")[1][-3:]
        # 매핑된 Task 찾기
        for task, numbers in task_mapping.get(speech_class, {}).items():
            if last_three_digits in numbers:
                return task
        return "unknown"
    except IndexError:
        return "unknown"

df["Task"] = df.apply(lambda row: get_task(row["FileName"], row["Class"]), axis=1)

df.to_csv(output_file, index=False)

print(f"새로운 CSV 파일이 생성되었습니다: {output_file}")
