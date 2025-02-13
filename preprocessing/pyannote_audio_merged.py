import os
import shutil
from pydub import AudioSegment
from pyannote.audio import Pipeline
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def copy_file(src, dst):
    """
    파일을 복사하는 함수. 병렬 작업을 위해 사용.
    
    Parameters:
        src (str): 원본 파일 경로
        dst (str): 복사할 파일 경로
    """
    shutil.copy(src, dst)
    print(f"Copied original audio: {dst}")

def process_and_slice_audio(input_folder, output_folder, merge_threshold=3.0, min_duration=1.0, csv_path=None):
    """
    디렉터리 내의 .wav 파일을 처리하여 조건에 따라 슬라이싱하거나 원본 복사.
    slicing된 파일에 대한 구간 정보를 csv로 기록.
    
    수정된 부분: 여러 개로 쪼개진 슬라이스 구간을 하나의 오디오로 병합하여 저장.
    
    Parameters:
        input_folder (str): 입력 .wav 파일이 있는 디렉터리
        output_folder (str): 슬라이싱된 .wav 파일 또는 복사본을 저장할 디렉터리
        merge_threshold (float): 발화 구간 병합 기준(초)
        min_duration (float): 최종 결과에서 최소 발화 지속 시간(초)
        csv_path (str, optional): slicing 정보를 기록할 CSV 파일 경로
    """
    # Pyannote.audio 파이프라인 로드
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        # use_auth_token="",
    )
    pipeline.to(device)

    os.makedirs(output_folder, exist_ok=True)

    # CSV 파일이 주어졌다면, 한 번만 열고 헤더 작성
    csv_file = None
    csv_writer = None
    if csv_path is not None:
        csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        # 헤더
        csv_writer.writerow(["filename", "start/end segments (seconds)"])

    # 복사 작업을 병렬로 처리하기 위한 ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        # 입력 폴더의 모든 .wav 파일 처리
        for filename in os.listdir(input_folder):
            if filename.endswith(".wav"):
                # 파일명 조건 확인
                base_filename = os.path.splitext(filename)[0]
                parts = base_filename.split("_")
                
                if (
                    (parts[1][3:5] in ["10", "11", "30"] and parts[1][7:] in ["002", "010"]) or
                    (parts[1][3:5] in ["20", "40"] and parts[1][7:] in ["002", "010", "022", "025", "034"])
                ):
                    input_path = os.path.join(input_folder, filename)

                    # 화자 분리 수행
                    diarization = pipeline(input_path)

                    # 화자 분리 결과 저장
                    speaker_segments = []
                    for segment, _, speaker in diarization.itertracks(yield_label=True):
                        speaker_segments.append((speaker, segment.start, segment.end))

                    # 각 화자의 총 발화 시간 계산
                    speaker_durations = defaultdict(float)
                    for speaker, start, end in speaker_segments:
                        speaker_durations[speaker] += end - start

                    # 가장 긴 시간을 발화한 메인 화자 선택
                    main_speaker = max(speaker_durations, key=speaker_durations.get)

                    # 메인 화자의 발화 구간 추출
                    main_segments = [
                        seg for seg in speaker_segments if seg[0] == main_speaker
                    ]

                    # 시간 순서 정렬 (start 시간 기준)
                    main_segments.sort(key=lambda x: x[1])

                    # 메인 화자의 발화 구간 병합
                    merged_segments = []
                    if main_segments:
                        current_start, current_end = main_segments[0][1], main_segments[0][2]
                        for _, start, end in main_segments[1:]:
                            if start - current_end <= merge_threshold:  # 병합 조건
                                current_end = max(current_end, end)
                            else:
                                merged_segments.append((current_start, current_end))
                                current_start, current_end = start, end
                        # 마지막 병합된 구간 추가
                        merged_segments.append((current_start, current_end))

                    # min_duration(초) 미만의 구간 제거
                    final_segments = [
                        (start, end)
                        for start, end in merged_segments
                        if end - start >= min_duration
                    ]

                    if final_segments:
                        # 원본 오디오 로드
                        audio = AudioSegment.from_wav(input_path)

                        # CSV에 기록할 row 만들기
                        if csv_writer is not None:
                            row = [base_filename]
                            for seg_start, seg_end in final_segments:
                                row.append(round(seg_start, 2))
                                row.append(round(seg_end, 2))
                            csv_writer.writerow(row)

                        # 여러 슬라이스 구간을 하나의 오디오로 병합
                        combined_audio = AudioSegment.empty()
                        for start, end in final_segments:
                            start_ms = int(start * 1000)  # 초 -> 밀리초
                            end_ms = int(end * 1000)
                            combined_audio += audio[start_ms:end_ms]

                        output_filename = f"{base_filename}.wav"
                        output_path = os.path.join(output_folder, output_filename)
                        combined_audio.export(output_path, format="wav")
                        print(f"Saved merged sliced audio: {output_path}")
                    else:
                        # min_duration보다 긴 구간이 없다면, 원본 복사
                        output_filename = f"{base_filename}.wav"
                        output_path = os.path.join(output_folder, output_filename)
                        executor.submit(copy_file, input_path, output_path)
                else:
                    # 조건을 만족하지 않는 경우 원본 파일 복사
                    input_path = os.path.join(input_folder, filename)
                    output_filename = f"{base_filename}.wav"
                    output_path = os.path.join(output_folder, output_filename)
                    executor.submit(copy_file, input_path, output_path)

    # CSV 파일 닫기
    if csv_file is not None:
        csv_file.close()

    print("모든 파일 처리가 완료되었습니다!")
    if csv_path:
        print(f"slicing 정보가 '{csv_path}' 파일에 기록되었습니다.")

# 입력 및 출력 폴더 설정
input_folder = "/data/alc_jihan/h_wav_16K"        # 원본 .wav 파일이 위치한 폴더
output_folder = "/data/alc_jihan/h_wav_16K_merged"  # 슬라이싱된 파일 또는 복사본을 저장할 폴더
csv_path = "/data/alc_jihan/split_index/pyannote_conversation_split_merged.csv"  # slicing 구간 기록용 CSV

# 함수 실행
process_and_slice_audio(
    input_folder,
    output_folder,
    merge_threshold=3.0,
    min_duration=1.0,
    csv_path=csv_path
)
