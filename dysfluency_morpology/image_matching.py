import os
import glob
import shutil

source_dir = '/data/alc_jihan/silence_black_without_edge_imgae'
dest_base = '/data/alc_jihan/matched_image'

sober_blocks = {20, 40}
intoxicated_blocks = {10, 11, 30}

# sober 상태에 해당하는 task 번호와 대응되는 intoxicated 상태 task 번호 매핑
sober_to_intoxicated = {
    30: 5,   14: 14,  38: 18,
    34: 2,   10: 10,
    1: 1,    26: 6,   29: 9,   31: 11,  15: 15,
    41: 21,  59: 23,  51: 24,  50: 29,
    42: 22,  46: 25,  55: 26,  48: 27,  49: 28,
    8: 8,    13: 13,  24: 17,  19: 19,  60: 30,
    32: 3,   23: 7,   12: 12,  16: 16,  20: 20
}
# intoxicated 상태 task 번호 → sober 상태 task 번호 (역매핑)
intoxicated_to_sober = {v: k for k, v in sober_to_intoxicated.items()}

matches = {}

png_files = glob.glob(os.path.join(source_dir, '*.png'))

for filepath in png_files:
    filename = os.path.basename(filepath)
    parts = filename.split('_')
    if len(parts) < 2:
        continue

    numeric_part = parts[1]
    if len(numeric_part) < 5:
        continue

    try:
        block_str = numeric_part[3:5]
        block_num = int(block_str)
        task_str = numeric_part[-2:]
        task_num = int(task_str)
    except ValueError:
        continue

    # block 번호에 따라 상태 결정 및 sober task 번호 추출
    if block_num in sober_blocks:
        state = 'sober'
        sober_task = task_num  # sober의 경우 task 번호 그대로 사용
    elif block_num in intoxicated_blocks:
        state = 'intoxicated'
        # intoxicated의 task 번호를 역매핑으로 sober의 task 번호로 변경
        if task_num not in intoxicated_to_sober:
            continue
        sober_task = intoxicated_to_sober[task_num]
    else:
        continue

    if sober_task not in matches:
        matches[sober_task] = {'sober': [], 'intoxicated': []}

    matches[sober_task][state].append(filepath)

for sober_task, state_files in matches.items():
    if len(state_files['intoxicated']) == 0:
        continue

    dest_sober_dir = os.path.join(dest_base, f'sober_{sober_task}')
    dest_intoxicated_dir = os.path.join(dest_base, f'intoxicated_{sober_task}')
    os.makedirs(dest_sober_dir, exist_ok=True)
    os.makedirs(dest_intoxicated_dir, exist_ok=True)

    for sober_file in state_files['sober']:
        dst_path = os.path.join(dest_sober_dir, os.path.basename(sober_file))
        shutil.copy(sober_file, dst_path)

    for intox_file in state_files['intoxicated']:
        original_name = os.path.basename(intox_file)
        name_no_ext, ext = os.path.splitext(original_name)
        parts = name_no_ext.split('_')
        
        if len(parts) >= 2:
            new_name = '_'.join(parts[1:]) + '_' + parts[0] + ext
        else:
            new_name = original_name
        dst_path = os.path.join(dest_intoxicated_dir, new_name)
        shutil.copy(intox_file, dst_path)

    print(f"Task {sober_task} 복사 완료: sober {len(state_files['sober'])}개, intoxicated {len(state_files['intoxicated'])}개.")
