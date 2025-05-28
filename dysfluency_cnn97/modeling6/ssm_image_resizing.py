import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# ───── GPU 설정 ───── #
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 필요 시 변경
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ───── 경로 설정 ───── #
SRC_DIR = Path("/data/alc_jihan/morphology_thresholded_97")
DST_DIR = Path("/data/alc_jihan/morphology_thresholded_97_resized")
DST_DIR.mkdir(parents=True, exist_ok=True)

# ───── Resize transform (Lanczos) ───── #
resize_transform = transforms.Compose([
    transforms.Resize((512, 512), interpolation=InterpolationMode.LANCZOS),
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()

# ───── 변환 실행 ───── #
image_files = list(SRC_DIR.glob("*.png"))
print(f"총 {len(image_files)}개의 이미지 변환 시작")

for img_path in tqdm(image_files):
    # Load image
    image = Image.open(img_path).convert("RGB")  # 채널 수 통일
    image_tensor = resize_transform(image).to(DEVICE)

    # 다시 PIL 이미지로 변환 후 저장
    resized_image = to_pil(image_tensor.cpu())
    save_path = DST_DIR / img_path.name
    resized_image.save(save_path)

print("모든 이미지가 512x512로 Lanczos 보간되어 저장되었습니다.")
