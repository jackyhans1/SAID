import os
import csv
import re
import subprocess

# 경로 설정
dictionary_path = "/home/ai/Documents/MFA/pretrained_models/dictionary/german_mfa.dict"
script_csv_path = "/data/alc_jihan/split_index/script3.csv"
g2p_model_path = "/home/ai/Documents/MFA/pretrained_models/g2p/german_mfa.zip"
missing_words_file = "/home/ai/said/pheoneme_feature_mfa/missing_words.txt"
missing_words_g2p_file = "/home/ai/said/pheoneme_feature_mfa/missing_words_g2p.txt"

def load_existing_dictionary(dict_path):
    """
    기존 사전 파일(german_mfa.dict)을 읽어 단어 목록(집합)으로 반환합니다.
    """
    existing_words = set()
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            # 사전에 있는 단어는 소문자로 처리
            word = parts[0].lower()
            existing_words.add(word)
    return existing_words

def collect_missing_words(script_csv_path, existing_words):
    """
    script3.csv에서 Prompt 열을 모두 읽어서,
    각 Prompt 내 단어 중 사전에 없는 단어만 골라 반환합니다.
    """
    missing_words = set()
    with open(script_csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)  # 헤더 스킵
        for row in reader:
            if len(row) < 3:
                continue
            # Prompt 열이 row[2] 이후로 이어져 있으므로
            prompt = ",".join(row[2:])
            # 정규식으로 단어 토큰화 (알파벳/숫자 조합)
            tokens = re.findall(r"\w+", prompt.lower())
            for token in tokens:
                # 이미 사전에 있는 단어는 제외
                if token not in existing_words:
                    missing_words.add(token)
    return missing_words

def run_mfa_g2p(missing_words, g2p_model_path, output_file):
    """
    MFA의 G2P 기능을 이용하여 missing_words 목록에 대한 발음을 생성하고,
    결과를 output_file에 저장합니다.
    
    MFA 3.x 버전에서는 인자 순서가:
    mfa g2p INPUT_PATH G2P_MODEL_PATH OUTPUT_PATH
    순으로 요구됩니다.
    """
    # missing_words를 임시 파일로 작성
    with open(missing_words_file, 'w', encoding='utf-8') as f:
        for w in sorted(missing_words):
            f.write(w + "\n")
    
    command = [
        "mfa", "g2p",
        missing_words_file,   
        g2p_model_path,       
        output_file           
    ]
    subprocess.run(command, check=True)

def append_g2p_to_dictionary(output_file, dict_path):
    """
    G2P 결과(output_file)를 읽어서 사전(dict_path) 끝에 추가합니다.
    동일 단어에 대해 여러 후보가 있을 경우, 첫 번째 후보(가장 적합한 후보)만 추가합니다.
    """
    best_candidates = {}
    with open(output_file, 'r', encoding='utf-8') as out_f:
        for line in out_f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            word = parts[0]
            # 만약 이미 해당 단어가 있으면 넘어감
            if word not in best_candidates:
                best_candidates[word] = line
    
    with open(dict_path, 'a', encoding='utf-8') as dict_f:
        for candidate_line in best_candidates.values():
            dict_f.write(candidate_line + "\n")

if __name__ == "__main__":
    # 기존 사전에 있는 단어 로드
    existing_words = load_existing_dictionary(dictionary_path)
    
    # script3.csv에서 사전에 없는 단어만 수집
    missing_words = collect_missing_words(script_csv_path, existing_words)
    if not missing_words:
        print("사전에 없는 새로운 단어가 없습니다.")
    else:
        print(f"{len(missing_words)}개의 단어에 대해 G2P 변환을 진행합니다.")
        
        # G2P 수행
        run_mfa_g2p(missing_words, g2p_model_path, missing_words_g2p_file)
        
        # 변환 결과를 기존 사전에 추가 (동일 단어에 대해 첫 번째 후보만)
        append_g2p_to_dictionary(missing_words_g2p_file, dictionary_path)
        
        print(f"G2P 변환 완료: {dictionary_path} 파일에 새로운 단어가 추가되었습니다.")
