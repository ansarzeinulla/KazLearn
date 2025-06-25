import os
import warnings
from transformers import logging as hf_logging
from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
import scipy.io.wavfile as wavfile

# Suppress transformers & PyTorch warnings
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

def synthesize(text, model_dir, output_file):
    model = VitsModel.from_pretrained(model_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform

    scaled = (waveform.squeeze().numpy() * 32767).astype(np.int16)
    wavfile.write(output_file, rate=model.config.sampling_rate, data=scaled)

# Kazakh
synthesize("сәлеметсіз бе, сізге не керек?", "./models/mms-tts-kaz", "kazakh_mms.wav")
