import os
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor

def process_wav_file(
    file_name,
    input_dir,
    output_dir,
    chunk_length_sec=8,
    overlap_sec=2
):
    """단일 파일에 대해 8초 슬라이싱/패딩 작업을 수행."""
    file_path = os.path.join(input_dir, file_name)
    audio = AudioSegment.from_wav(file_path)
    duration_ms = len(audio)

    base_name = file_name[:-4]  # .wav 확장자 제거
    
    chunk_length_ms = chunk_length_sec * 1000
    overlap_ms = overlap_sec * 1000
    stride_ms = chunk_length_ms - overlap_ms  # 8초 - 2초 = 6초

    #----------------------------------------------
    # Case 1) 파일 길이가 8초 미만 → 무음 패딩 + "_0" 저장
    #----------------------------------------------
    if duration_ms < chunk_length_ms:
        padding_needed_ms = chunk_length_ms - duration_ms
        padded_audio = audio + AudioSegment.silent(duration=padding_needed_ms)
        out_path = os.path.join(output_dir, f"{base_name}_0.wav")
        padded_audio.export(out_path, format="wav")
        return

    #----------------------------------------------
    # Case 2) 파일 길이가 8초 이상 → 슬라이딩
    # 첫 chunk부터 "_1"로 시작
    #----------------------------------------------
    chunk_index = 1
    start_ms = 0

    while True:
        end_ms = start_ms + chunk_length_ms

        if end_ms <= duration_ms:
            # 정상적으로 8초 chunk 추출
            chunk_audio = audio[start_ms:end_ms]
            out_path = os.path.join(output_dir, f"{base_name}_{chunk_index}.wav")
            chunk_audio.export(out_path, format="wav")

            chunk_index += 1
            start_ms += stride_ms

            # 다음 chunk가 파일 길이를 넘어가는지 확인
            if start_ms + chunk_length_ms > duration_ms:
                # 마지막 조각 처리
                last_start_ms = duration_ms - chunk_length_ms
                if last_start_ms < start_ms:
                    # start_ms가 더 크면 → 그때부터 끝까지 잘라서 패딩
                    chunk_audio = audio[start_ms:duration_ms]
                    if len(chunk_audio) < chunk_length_ms:
                        silence_ms = chunk_length_ms - len(chunk_audio)
                        chunk_audio += AudioSegment.silent(duration=silence_ms)
                else:
                    # 마지막 구간은 last_start_ms부터 8초
                    chunk_audio = audio[last_start_ms:duration_ms]
                    if len(chunk_audio) < chunk_length_ms:
                        silence_ms = chunk_length_ms - len(chunk_audio)
                        chunk_audio += AudioSegment.silent(duration=silence_ms)

                out_path = os.path.join(output_dir, f"{base_name}_{chunk_index}.wav")
                chunk_audio.export(out_path, format="wav")
                break

        else:
            # 처음부터 8초 미만이 될 가능성이 있다는 뜻 → 마지막 구간 처리
            last_start_ms = duration_ms - chunk_length_ms

            if last_start_ms < start_ms:
                # start_ms부터 끝까지 + 패딩
                chunk_audio = audio[start_ms:duration_ms]
                if len(chunk_audio) < chunk_length_ms:
                    chunk_audio += AudioSegment.silent(duration=(chunk_length_ms - len(chunk_audio)))
            else:
                chunk_audio = audio[last_start_ms:duration_ms]
                if len(chunk_audio) < chunk_length_ms:
                    chunk_audio += AudioSegment.silent(duration=(chunk_length_ms - len(chunk_audio)))

            out_path = os.path.join(output_dir, f"{base_name}_{chunk_index}.wav")
            chunk_audio.export(out_path, format="wav")
            break


def slide_or_pad_wav_files(
    input_dir="/data/alc_jihan/h_wav_16K_sliced",
    output_dir="/data/alc_jihan/h_wav_slided",
    chunk_length_sec=8,
    overlap_sec=2
):
    """
    input_dir 내 모든 .wav 파일을 8초 단위로 슬라이싱(오버랩 2초) 후 output_dir에 저장.
    - 8초 미만이면 '_0'
    - 8초 이상이면 첫 조각부터 '_1'
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

    # 병렬 처리
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    max_workers = 16 
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file_name in files:
            futures.append(
                executor.submit(
                    process_wav_file,
                    file_name,
                    input_dir,
                    output_dir,
                    chunk_length_sec,
                    overlap_sec
                )
            )
        # 모든 작업 완료 대기
        for f in futures:
            f.result()

if __name__ == "__main__":
    slide_or_pad_wav_files()
