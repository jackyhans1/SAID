from allosaurus.app import read_recognizer

# Allosaurus 모델 로드
model = read_recognizer()

# 음성 파일을 분석 (독일어 예제)
result = model.recognize("/data/alc_jihan/h_wav_16K/0_0062014001_h_00.wav")
print(result)
