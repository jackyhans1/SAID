import pandas as pd

input_file = "/data/ALC/TABLE/SESSEXT.TBL"
output_file = "/data/ALC/TABLE/SESSEXT.csv"

# .TBL 파일을 읽어서 CSV로 저장
def convert_tbl_to_csv(input_path, output_path):
    try:
        # 첫 줄이 헤더로 사용되므로 delim_whitespace로 공백 기준 분리
        df = pd.read_csv(input_path, delim_whitespace=True)

        # CSV로 저장
        df.to_csv(output_path, index=False)
        print(f"파일이 성공적으로 변환되었습니다: {output_path}")
    except Exception as e:
        print(f"오류 발생: {e}")

convert_tbl_to_csv(input_file, output_file)