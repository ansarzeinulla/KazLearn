import warnings
import logging
import os
import requests
import torch
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    VitsModel, AutoTokenizer,
    logging as hf_logging
)

# Suppress warnings/logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

def process_audio_pipeline(audio_path, whisper_model_path, tts_model_path, api_key, assistant_id, output_audio_path):
    ### Stage 1: Transcribe
    processor = WhisperProcessor.from_pretrained(whisper_model_path)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_model_path)

    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", language="kk")
    input_features = inputs.input_features
    attention_mask = torch.ones(input_features.shape[:-1], dtype=torch.long)

    pred_ids = model.generate(input_features, attention_mask=attention_mask)
    text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    print("🔤 Transcribed text:", text)

    ### Stage 2: Grammar Correction via API
    url = f'https://oylan.nu.edu.kz/api/v1/assistant/{assistant_id}/interactions/'
    headers = {
        'accept': 'application/json',
        'Authorization': f'Api-Key {api_key}',
    }
    data = {
        'content': f'ОСЫ СӨЙЛЕМДЕГІ ГРАММАТИКАЛЫҚ ҚАТЕЛЕРДІ ТАУЫП, ДҰРЫСТАП, ТҮСІНДІР: {text}',
        'assistant': str(assistant_id),
        'stream': 'false',
    }

    response = requests.post(url, headers=headers, data=data)
    if response.status_code not in (200, 201):
        raise RuntimeError(f"API error {response.status_code}: {response.text}")

    corrected_text = response.json()['response']['content']
    print("✅ Corrected response:", corrected_text)

    ### Stage 3: Text-to-Speech (VITS)
    vits_model = VitsModel.from_pretrained(tts_model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(tts_model_path, local_files_only=True)

    inputs = tokenizer(corrected_text, return_tensors="pt")
    with torch.no_grad():
        waveform = vits_model(**inputs).waveform

    scaled = (waveform.squeeze().numpy() * 32767).astype(np.int16)
    wavfile.write(output_audio_path, rate=vits_model.config.sampling_rate, data=scaled)

    print(f"🔊 Audio saved to: {output_audio_path}")
    return corrected_text

# Example usage:
if __name__ == "__main__":
    process_audio_pipeline(
        audio_path="New-Recording.wav",
        whisper_model_path="./models/whisper-base.kk",
        tts_model_path="./models/mms-tts-kaz",
        api_key="ezDqBEZN.5iVQlBt4QC0Ka7GBTcf8KIX5KBFyKA1P",
        assistant_id=1164,
        output_audio_path="final_output.wav"
    )
