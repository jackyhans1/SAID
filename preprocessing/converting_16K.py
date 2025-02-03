import os
from pydub import AudioSegment

# 입력 폴더와 출력 폴더 경로
input_folder = "/data/alc_jihan/h_wav"
output_folder = "/data/alc_jihan/h_wav_16K"

# 출력 폴더가 없으면 생성
os.makedirs(output_folder, exist_ok=True)

# 입력 폴더의 모든 파일 탐색
for filename in os.listdir(input_folder):
    # .wav 파일만 처리
    if filename.endswith(".wav"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            # 오디오 파일 로드
            audio = AudioSegment.from_file(input_path)

            # 16kHz로 변환
            audio_16k = audio.set_frame_rate(16000)

            # 변환된 파일 저장
            audio_16k.export(output_path, format="wav")
            print(f"변환 완료: {input_path} -> {output_path}")
        except Exception as e:
            print(f"오류 발생: {input_path} ({e})")

print("모든 파일 변환 완료!")
