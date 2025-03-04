import os
import glob
import csv
import re

# 독일어 모음 (필요에 따라 추가 확장 가능)
VOWELS = {"a", "e", "i", "o", "u", "ä", "ö", "ü", "aː", "eː", "iː", "oː", "uː"}
# 독일어 마찰음 (fricatives)
FRICATIVES = {"f", "v", "s", "z", "ʃ", "ç", "x", "h"}
# 독일어 파열음 (plosives)
PLOSIVES = {"p", "b", "t", "d", "k", "g"}

def parse_textgrid(file_path):
    """
    TextGrid 파일을 간단히 파싱하여 tier별로 interval 정보를 dict로 반환.
    반환 형식: {tier_name: [ {"xmin":..., "xmax":..., "text":...}, ... ], ...}
    """
    tiers = {}
    current_tier = None
    in_interval = False
    current_interval = {}

    with open(file_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # tier 이름 추출 (예: name = "phones")
            if line.startswith("name ="):
                tier_name = line.split("=", 1)[1].strip().strip('"')
                current_tier = tier_name
                tiers[current_tier] = []
            # interval 시작 감지
            elif line.startswith("intervals ["):
                in_interval = True
                current_interval = {}
            elif in_interval and line.startswith("xmin ="):
                try:
                    current_interval['xmin'] = float(line.split("=", 1)[1].strip())
                except ValueError:
                    current_interval['xmin'] = 0.0
            elif in_interval and line.startswith("xmax ="):
                try:
                    current_interval['xmax'] = float(line.split("=", 1)[1].strip())
                except ValueError:
                    current_interval['xmax'] = 0.0
            elif in_interval and line.startswith("text ="):
                text_val = line.split("=", 1)[1].strip()
                if text_val.startswith('"') and text_val.endswith('"'):
                    text_val = text_val[1:-1]
                current_interval['text'] = text_val
                # interval 완료: tier에 추가 후 리셋
                if current_tier is not None:
                    tiers[current_tier].append(current_interval)
                in_interval = False
    return tiers

def filter_canonical_text(text):
    """
    canonical tier의 text에서 숫자(예: 0.99 등)를 모두 제외하고 문자 토큰만 반환.
    """
    tokens = text.split()
    filtered = [tok for tok in tokens if not re.match(r'^[0-9.]+$', tok)]
    return filtered

def compute_edit_distance(seq1, seq2):
    """
    두 토큰 리스트의 Levenshtein edit distance (삽입, 삭제, 대체 비용 1) 계산.
    """
    n = len(seq1)
    m = len(seq2)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,      # deletion
                           dp[i][j-1] + 1,      # insertion
                           dp[i-1][j-1] + cost) # substitution
    return dp[n][m]

def align_sequences(seq1, seq2):
    """
    두 토큰 리스트의 alignment을 구하여, gap은 None으로 채운 두 리스트를 반환.
    (Wagner-Fischer 알고리즘을 backtracking 방식으로 alignment 생성)
    """
    n = len(seq1)
    m = len(seq2)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,
                           dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)
    aligned1, aligned2 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if seq1[i-1] == seq2[j-1] else 1):
            aligned1.insert(0, seq1[i-1])
            aligned2.insert(0, seq2[j-1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            aligned1.insert(0, seq1[i-1])
            aligned2.insert(0, None)
            i -= 1
        else:
            aligned1.insert(0, None)
            aligned2.insert(0, seq2[j-1])
            j -= 1
    return aligned1, aligned2

def extract_features_from_file(file_path):
    tiers = parse_textgrid(file_path)
    
    # 전체 canonical과 phones의 phoneme sequence (빈 토큰은 제외)
    canonical_tokens = []
    for interval in tiers.get("canonical", []):
        if interval.get("text", "").strip():
            canonical_tokens.extend(filter_canonical_text(interval["text"]))
    
    phones_tokens = []
    for interval in tiers.get("phones", []):
        token_text = interval.get("text", "").strip()
        if token_text:
            # "spn" 토큰은 제외
            tokens = [tok for tok in token_text.split() if tok != "spn"]
            phones_tokens.extend(tokens)
    
    # 전체 edit distance 및 normalized 값 (phoneme 단위)
    overall_distance = compute_edit_distance(canonical_tokens, phones_tokens)
    norm_overall_distance = overall_distance / len(canonical_tokens) if len(canonical_tokens) > 0 else 0

    # word 단위 mispronunciation 및 추가적으로 모음, 마찰음, 파열음 분석
    mispronounced_words = 0
    total_words = 0
    total_mis_vowels = 0
    total_vowels = 0
    total_fricatives = 0
    mispronounced_fricatives = 0
    total_plosives = 0
    mispronounced_plosives = 0

    canonical_intervals = tiers.get("canonical", [])
    phones_intervals = tiers.get("phones", [])

    for c_int in canonical_intervals:
        c_text = c_int.get("text", "").strip()
        if c_text == "":
            continue  # 빈 단어 건너뜀
        total_words += 1
        c_tokens = filter_canonical_text(c_text)
        # 해당 canonical interval의 시간 범위 내의 phones interval 추출 (spn는 제외)
        c_xmin = c_int.get("xmin", 0)
        c_xmax = c_int.get("xmax", 0)
        p_tokens = []
        for p_int in phones_intervals:
            p_xmin = p_int.get("xmin", 0)
            p_xmax = p_int.get("xmax", 0)
            if p_xmin >= c_xmin and p_xmax <= c_xmax:
                token_text = p_int.get("text", "").strip()
                if token_text:
                    tokens = [tok for tok in token_text.split() if tok != "spn"]
                    p_tokens.extend(tokens)
        word_distance = compute_edit_distance(c_tokens, p_tokens)
        if word_distance > 0:
            mispronounced_words += 1

        # alignment을 통해 canonical과 phones의 토큰 비교 (모음, 마찰음, 파열음)
        aligned_c, aligned_p = align_sequences(c_tokens, p_tokens)
        for ac, ap in zip(aligned_c, aligned_p):
            # 모음 처리
            if ac in VOWELS:
                total_vowels += 1
                if ac != ap:
                    total_mis_vowels += 1
            # 마찰음 처리
            if ac in FRICATIVES:
                total_fricatives += 1
                if ac != ap:
                    mispronounced_fricatives += 1
            # 파열음 처리
            if ac in PLOSIVES:
                total_plosives += 1
                if ac != ap:
                    mispronounced_plosives += 1

    norm_mispronounced_words = mispronounced_words / total_words if total_words > 0 else 0
    norm_mis_vowels = total_mis_vowels / total_vowels if total_vowels > 0 else 0
    norm_mis_fricatives = mispronounced_fricatives / total_fricatives if total_fricatives > 0 else 0
    norm_mis_plosives = mispronounced_plosives / total_plosives if total_plosives > 0 else 0

    features = {
        "LevenshteinDistance": overall_distance,
        "NormalizedLevenshtein": norm_overall_distance,
        "MispronouncedWords": mispronounced_words,
        "NormalizedMispronouncedWords": norm_mispronounced_words,
        "VowelMispronunciations": total_mis_vowels,
        "NormalizedVowelMispronunciations": norm_mis_vowels,
        "FricativeMispronunciations": mispronounced_fricatives,
        "NormalizedFricativeMispronunciations": norm_mis_fricatives,
        "PlosiveMispronunciations": mispronounced_plosives,
        "NormalizedPlosiveMispronunciations": norm_mis_plosives
    }
    return features

def main():
    input_dir = "/data/alc_jihan/phoneme_features_mfa/final_output"
    output_csv = "/data/alc_jihan/extracted_features_mfa/mfa_features2.csv"

    files = glob.glob(os.path.join(input_dir, "*.TextGrid"))
    header = ["FileName", "LevenshteinDistance", "NormalizedLevenshtein", 
              "MispronouncedWords", "NormalizedMispronouncedWords", 
              "VowelMispronunciations", "NormalizedVowelMispronunciations",
              "FricativeMispronunciations", "NormalizedFricativeMispronunciations",
              "PlosiveMispronunciations", "NormalizedPlosiveMispronunciations"]

    with open(output_csv, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for file_path in files:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            features = extract_features_from_file(file_path)
            row = {"FileName": file_name}
            row.update(features)
            writer.writerow(row)
            print(f"Processed {file_name}")

if __name__ == "__main__":
    main()

