import os
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from concurrent.futures import ThreadPoolExecutor

def extract_features(file_path, output_dir, model_name, device):
    
    file_name = os.path.basename(file_path)

    try:
        # Load the pre-trained HuBERT model and processor inside the function
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = HubertModel.from_pretrained(model_name).to(device)
        model.eval()

        # Load the audio file
        waveform, sample_rate = torchaudio.load(file_path)

        # Process the audio to extract input features
        input_values = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)

        # Extract features using the model
        with torch.no_grad():
            outputs = model(input_values)
            features = outputs.last_hidden_state

        # Ensure the extracted features are a tensor
        if isinstance(features, torch.Tensor):
            # Save the features to a .pt file
            output_file = os.path.join(output_dir, file_name.replace(".wav", ".pt"))
            torch.save(features.cpu(), output_file)
            print(f"Features extracted and saved to: {output_file}")
        else:
            raise TypeError("Extracted features are not a tensor.")

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

def extract_features_from_wav(input_dir, output_dir, model_name="facebook/hubert-large-ll60k", num_workers=8):
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all .wav files in the input directory
    wav_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".wav")]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(extract_features, file_path, output_dir, model_name, device) for file_path in wav_files]
        for i, future in enumerate(futures):
            try:
                future.result()
                print(f"[{i + 1}/{len(futures)}] Processing completed.")
            except Exception as e:
                print(f"[{i + 1}/{len(futures)}] Error: {e}")

# Paths to input and output directories
input_directory = "/data/alc_jihan/h_wav_16K_sliced"
output_directory = "/data/alc_jihan/extracted_features"

# Extract features with parallel processing
extract_features_from_wav(input_directory, output_directory, num_workers=8)
