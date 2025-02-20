import torch
import os

def analyze_pt_file(file_path):
    """
    Load a .pt file and analyze its tensor dimensions.
    Args:
        file_path (str): Path to the .pt file.
    """
    tensor = torch.load(file_path)

    print(f"File: {file_path}")
    print(f"Tensor type: {type(tensor)}")
    print(f"Tensor shape: {tensor.shape}")  # [Batch, Sequence, Feature_Dimension]
    
    print("Sample values:")
    print(tensor[0, :5, :5])

FILE_PATH = "/data/alc_jihan/hubert_meta_disfluency_feature_fusion/0_0062014034_h_00.pt"
if os.path.exists(FILE_PATH):
    analyze_pt_file(FILE_PATH)
else:
    print(f"File not found: {FILE_PATH}")