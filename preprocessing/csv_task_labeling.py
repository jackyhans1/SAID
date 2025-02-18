import pandas as pd

# 파일 경로
input_file = "/data/alc_jihan/split_index/merged_data_without_task.csv"
output_file = "/data/alc_jihan/split_index/merged_data.csv"

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

def get_task(file_name):
    try:
        # 파일명에서 두 번째 부분 (언더바로 분리) 추출
        code_str = file_name.split("_")[1]
        # 마지막 세 자리 숫자 추출 (매핑 번호)
        last_three_digits = code_str[-3:]
        # 3번째부터 5번째 문자 추출하여 조건에 맞게 매핑 결정
        category_code = code_str[3:5]
        if category_code in {"10", "11", "30"}:
            mapping_category = "Intoxicated"
        elif category_code in {"20", "40"}:
            mapping_category = "Sober"
        else:
            return "unknown"
        
        # 선택된 매핑에서 해당하는 Task 찾기
        for task, numbers in task_mapping.get(mapping_category, {}).items():
            if last_three_digits in numbers:
                return task
        return "unknown"
    except IndexError:
        return "unknown"

# 기존 Class 컬럼 대신 FileName으로부터 Task 결정
df["Task"] = df["FileName"].apply(get_task)

# 새로운 CSV 파일 저장
df.to_csv(output_file, index=False)
print(f"새로운 CSV 파일이 생성되었습니다: {output_file}")
