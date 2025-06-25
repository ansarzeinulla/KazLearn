import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import torch

model_path = "./models/whisper-base.kk"

processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)

audio, sr = librosa.load("New-Recording.wav", sr=16000)

# Force Kazakh transcription — no detection, no translation
inputs = processor(audio, sampling_rate=sr, return_tensors="pt", language="kk")

input_features = inputs.input_features

# Build attention mask: tensor of ones matching input_features shape (batch_size, seq_len)
attention_mask = torch.ones(input_features.shape[:-1], dtype=torch.long)

pred_ids = model.generate(input_features, attention_mask=attention_mask)

text = processor.batch_decode(pred_ids, skip_special_tokens=True)

print(text[0])
