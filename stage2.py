from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# 1. Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load model and tokenizer from your folder
model_path = "models/mGPT-1.3B-kazakh"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
model.eval()

# 3. Tokenize your prompt (Kazakh text)
prompt = "Сәлем әлем!"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# 4. Generate output
outputs = model.generate(
    input_ids,
    max_length=100,
    min_length=50,
    eos_token_id=5,            # end‑of‑sentence token
    pad_token_id=1,            # padding token
    top_k=10,
    top_p=0.8,
    no_repeat_ngram_size=4,
    do_sample=True
)

# 5. Decode and print
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated)
