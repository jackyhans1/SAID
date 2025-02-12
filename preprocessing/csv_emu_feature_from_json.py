import os
import json
import csv
import glob

# 입력 JSON 파일들이 있는 폴더 경로와 출력 CSV 파일 경로 설정
json_folder = '/data/alc_jihan/h_json/'
output_csv = '/data/alc_jihan/extracted_features_whisper_disfluency/whisper_meta_emu_feature.csv'

# json 폴더 내의 모든 JSON 파일 찾기
json_files = glob.glob(os.path.join(json_folder, '*.json'))

# CSV 파일 열기 (쓰기 모드)
with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    
    # CSV 파일의 헤더 작성
    writer.writerow([
        'Filename', 'Hesistation', 'P_less_1', 'P_more_1', 
        'Delayed', 'Pronunciation', 'Repetition', 'Repair', 
        'Interupted', 'P_Sum', 'Error_Sum'
    ])
    
    # 각 JSON 파일 처리
    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"JSON 파일 {json_path} 읽기 실패: {e}")
            continue

        # 파일명에서 '_annot.json' 제거
        base_filename = os.path.basename(json_path)
        if base_filename.endswith('_annot.json'):
            filename_value = base_filename[:-len('_annot.json')]
        else:
            filename_value = os.path.splitext(base_filename)[0]

        # JSON 내부에서 "irreg" 라벨의 값을 찾기
        irreg_value = None
        for level in data.get('levels', []):
            for item in level.get('items', []):
                for label in item.get('labels', []):
                    if label.get('name') == 'irreg':
                        irreg_value = label.get('value')
                        break
                if irreg_value is not None:
                    break
            if irreg_value is not None:
                break

        if irreg_value is None:
            print(f"'irreg' 값이 {json_path}에서 발견되지 않음.")
            continue

        # '|' 구분자로 분리 후, 첫 번째 숫자는 사용하지 않고 이후 순서대로 할당
        values = irreg_value.split('|')
        if len(values) < 9:
            print(f"예상보다 적은 숫자 ({irreg_value})가 {json_path}에 있습니다.")
            continue

        try:
            # values[0]은 사용하지 않음
            Hesistation   = int(values[1])
            P_less_1      = int(values[2])
            P_more_1      = int(values[3])
            Delayed       = int(values[4])
            Pronunciation = int(values[5])
            Repetition    = int(values[6])
            Repair        = int(values[7])
            Interupted    = int(values[8])
        except Exception as e:
            print(f"숫자 변환 에러 in {json_path}: {e}")
            continue

        # 계산: P_Sum = P_less_1 + P_more_1, Error_Sum = 모든 해당 숫자들의 합
        P_Sum = P_less_1 + P_more_1
        Error_Sum = Hesistation + P_less_1 + P_more_1 + Delayed + Pronunciation + Repetition + Repair + Interupted

        # CSV 파일에 한 줄 기록
        writer.writerow([
            filename_value, Hesistation, P_less_1, P_more_1, 
            Delayed, Pronunciation, Repetition, Repair, 
            Interupted, P_Sum, Error_Sum
        ])

print(f"CSV 파일이 생성되었습니다: {output_csv}")
