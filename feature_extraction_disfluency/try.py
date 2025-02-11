from transformers import Wav2Vec2Tokenizer

model_path = "facebook/wav2vec2-large-xlsr-53"
save_path = "/home/ai/wav2vec2_model/wav2vec2-large-xlsr-53"

# 토크나이저 다운로드
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(save_path)

print("Tokenizer downloaded and saved successfully!")
