import os

# 데이터 폴더 정의
base_dir = "/data/alc_jihan/"
hlb_dir = os.path.join(base_dir, "hlb")
h_wav_dir = os.path.join(base_dir, "h_wav")
h_json_dir = os.path.join(base_dir, "h_json")
par_dir = os.path.join(base_dir, "par")
textgrid_dir = os.path.join(base_dir, "textgrid")
m_wav_dir = os.path.join(base_dir, "m_wav")
m_json_dir = os.path.join(base_dir, "m_json")

# hlb 파일 기준으로 파일 이름 목록 생성
hlb_files_h = set()
hlb_files_m = set()
for file in os.listdir(hlb_dir):
    if file.endswith(".hlb"):
        # 기본 이름 추출
        base_name_h = file.replace(".hlb", "")
        base_name_m = base_name_h.replace("_h_", "_m_")  # _h_를 _m_로 변환
        hlb_files_h.add(base_name_h)
        hlb_files_m.add(base_name_m)

# JSON 파일 전용 매칭 확인 함수
def is_json_file_matched(file_name, valid_files):
    # JSON 파일은 접미사 '_annot'를 추가하여 매칭 확인
    base_name = file_name.replace("_annot.json", "")
    return base_name in valid_files

# 매칭되지 않는 파일 삭제 함수
def delete_unmatched_files(target_dir, valid_files, is_json=False):
    for file_name in os.listdir(target_dir):
        if is_json:
            # JSON 파일의 매칭 여부를 확인
            if not is_json_file_matched(file_name, valid_files):
                file_path = os.path.join(target_dir, file_name)
                print(f"삭제 중: {file_path}")
                os.remove(file_path)
        else:
            # 일반 파일의 매칭 여부를 확인
            file_base, _ = os.path.splitext(file_name)  # 확장자 제거
            if file_base not in valid_files:
                file_path = os.path.join(target_dir, file_name)
                print(f"삭제 중: {file_path}")
                os.remove(file_path)

# h_wav, h_json, par, textgrid에서 매칭되지 않는 데이터 삭제 (hlb 기준)
delete_unmatched_files(h_wav_dir, hlb_files_h)
delete_unmatched_files(h_json_dir, hlb_files_h, is_json=True)  # JSON 파일 매칭
delete_unmatched_files(par_dir, hlb_files_h)
delete_unmatched_files(textgrid_dir, hlb_files_h)

# m_wav, m_json에서 매칭되지 않는 데이터 삭제 (hlb 기준의 _m_XX로 변환된 값 기준)
delete_unmatched_files(m_wav_dir, hlb_files_m)
delete_unmatched_files(m_json_dir, hlb_files_m, is_json=True)  # JSON 파일 매칭
