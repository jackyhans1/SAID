import os
import shutil
import subprocess
import csv
from textgrid import TextGrid, IntervalTier, Interval

# GPU 사용을 위한 환경변수 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["MFA_LOG_LEVEL"] = "DEBUG"

def load_pronunciation_dict(dict_path):
    """
    발음 사전 파일을 파싱하여 단어-발음 딕셔너리를 생성합니다.
    파일은 각 줄이 "단어 phoneme1 phoneme2 ..." 형태라고 가정합니다.
    """
    pron_dict = {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            pronunciation = " ".join(parts[1:])
            pron_dict[word] = pronunciation
    return pron_dict

def add_canonical_tier(textgrid_path, pron_dict, output_path):
    """
    기존 TextGrid 파일에 표준 발음(canonical) tier를 추가하여 저장합니다.
    각 단어 구간("words" tier)을 참조하여, 해당 단어의 표준 발음을 사전에서 가져옵니다.
    """
    tg = TextGrid()
    tg.read(textgrid_path)
    
    try:
        words_tier = tg.getFirst("words")
    except Exception as e:
        print("TextGrid에서 'words' tier를 찾을 수 없습니다:", e)
        return
    
    canon_tier = IntervalTier("canonical", words_tier.minTime, words_tier.maxTime)
    
    for interval in words_tier.intervals:
        word = interval.mark.strip().lower()
        canon_text = pron_dict.get(word, "")
        new_interval = Interval(interval.minTime, interval.maxTime, canon_text)
        canon_tier.intervals.append(new_interval)
    
    tg.tiers.append(canon_tier)
    tg.write(output_path)
    print(f"표준 발음 tier가 추가된 TextGrid 파일이 저장되었습니다: {output_path}")

def load_script_dict(script_csv_path):
    """
    script.csv 파일을 읽어 (Class, Number)를 key로 Prompt를 매핑하는 딕셔너리를 생성합니다.
    Prompt에 쉼표가 포함된 경우, 첫 두 컬럼(Class, Number) 이후의 모든 값을 합쳐 하나의 Prompt로 인식합니다.
    """
    script_dict = {}
    with open(script_csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Header: Class, Number, Prompt
        for row in reader:
            if len(row) < 3:
                continue
            class_field = row[0].strip()
            number_field = row[1].strip()
            # row[2:]에 있는 모든 항목을 합쳐 Prompt로 사용
            prompt_field = ",".join(item.strip() for item in row[2:])
            key = (class_field, number_field)
            script_dict[key] = prompt_field
    return script_dict

def load_merged_data(merged_csv_path, allowed_tasks):
    """
    merged_data.csv 파일에서 Task가 허용된 파일들만 필터링하여 FileName 리스트로 반환합니다.
    """
    merged_files = []
    with open(merged_csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            task = row["Task"].strip()
            if task in allowed_tasks:
                merged_files.append(row["FileName"].strip())
    return merged_files

if __name__ == "__main__":
    # 경로 설정
    audio_dir = "/data/alc_jihan/h_wav_16K_merged"
    merged_csv_path = "/data/alc_jihan/split_index/merged_data.csv"
    script_csv_path = "/data/alc_jihan/split_index/script3.csv"
    
    # MFA에 사용할 발음 사전 및 음향 모델 경로
    dictionary_path = "/home/ai/Documents/MFA/pretrained_models/dictionary/german_mfa.dict"
    acoustic_model_path = "/home/ai/Documents/MFA/pretrained_models/acoustic/german_mfa.zip"
    
    # 결과를 저장할 디렉토리들
    mfa_output_dir = "/data/alc_jihan/phoneme_features_mfa/textgrid"
    canonical_output_dir = "/data/alc_jihan/phoneme_features_mfa/final_output"
    os.makedirs(mfa_output_dir, exist_ok=True)
    os.makedirs(canonical_output_dir, exist_ok=True)
    
    # CSV에서 사용할 Task 필터
    allowed_tasks = {"read_command", "address", "number", "tongue_twister"}
    
    # script.csv와 merged_data.csv 로드
    script_dict = load_script_dict(script_csv_path)
    merged_file_list = load_merged_data(merged_csv_path, allowed_tasks)
    
    # 발음 사전 로드 (canonical tier 추가용)
    pron_dict = load_pronunciation_dict(dictionary_path)
    
    # 여러 파일을 한 번에 처리하기 위한 corpus 디렉토리 생성
    corpus_dir = "/data/alc_jihan/phoneme_features_mfa/mfa_corpus"
    if os.path.exists(corpus_dir):
        shutil.rmtree(corpus_dir)
    os.makedirs(corpus_dir, exist_ok=True)
    
    processed_files = []
    # 각 파일에 대해 전사를 결정하고, corpus 디렉토리에 오디오 파일과 .lab 파일 생성
    for file_base in merged_file_list:
        audio_path = os.path.join(audio_dir, file_base + ".wav")
        if not os.path.exists(audio_path):
            print(f"오디오 파일이 존재하지 않습니다: {audio_path}")
            continue
        
        # 파일명 예: 0_0062014001_h_00 -> split 결과: ["0", "0062014001", "h", "00"]
        parts = file_base.split("_")
        if len(parts) < 2:
            print(f"파일명 형식 오류: {file_base}")
            continue
        num_str = parts[1]
        # num_str의 [3:5]가 "20" 또는 "40"이면 Sober, 아니면 Intoxicated로 처리
        if num_str[3:5] in {"20", "40"}:
            class_for_prompt = "Sober"
        else:
            class_for_prompt = "Intoxicated"
        number_key = num_str[-3:]
        key = (class_for_prompt, number_key)
        transcript = script_dict.get(key, "")
        if transcript == "":
            print(f"script.csv에서 매칭되는 Prompt를 찾지 못했습니다. (Key: {key})")
            continue
        
        # corpus 디렉토리에 오디오 파일 복사 및 전사(.lab) 파일 생성
        shutil.copy(audio_path, corpus_dir)
        lab_file = os.path.join(corpus_dir, file_base + ".lab")
        with open(lab_file, "w", encoding="utf-8") as f:
            f.write(transcript)
        
        processed_files.append(file_base)
        print(f"Added {file_base} to corpus with transcript: {transcript}")
    
    if not processed_files:
        print("처리할 파일이 없습니다.")
        exit(1)
    
    # 하나의 MFA 호출로 corpus 디렉토리의 모든 파일에 대해 forced alignment 수행 (GPU 사용)
    command = [
        "mfa",
        "align",
        corpus_dir,
        dictionary_path,
        acoustic_model_path,
        mfa_output_dir,
        "--clean",
        "--use_gpu",
        "--beam", "15",         # 기본 beam width보다 넓게 설정 (예시 값)
        "--retry_beam", "40"      # 재시도 beam width 조정 (예시 값)
    ]

    
    env = os.environ.copy()
    print("Starting MFA alignment in batch mode...")
    subprocess.run(command, check=True, env=env)
    print("MFA alignment completed.")
    
    # 각 파일별로 canonical tier를 추가하여 최종 TextGrid 생성
    for file_base in processed_files:
        textgrid_file = os.path.join(mfa_output_dir, file_base + ".TextGrid")
        canonical_textgrid_file = os.path.join(canonical_output_dir, file_base + ".TextGrid")
        if not os.path.exists(textgrid_file):
            print(f"TextGrid 파일이 생성되지 않았습니다: {textgrid_file}")
            continue
        add_canonical_tier(textgrid_file, pron_dict, canonical_textgrid_file)
