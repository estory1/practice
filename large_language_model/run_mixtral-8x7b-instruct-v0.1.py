# src: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1#lower-precision-using-8-bit--4-bit-using-bitsandbytes

#%%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from importlib import reload 
# reload(AutoModelForCausalLM)
# reload(AutoTokenizer)

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# https://huggingface.co/docs/huggingface_hub/v0.19.3/guides/download
from huggingface_hub import snapshot_download
snapshot_download(repo_id=model_id)#, force_download=True, resume_download=False)

#%%
#device cfg
# I'm on M1 Mac, so see: https://developer.apple.com/metal/pytorch/
if torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.set_default_dtype(torch.float32)  # works-around `TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.`?
# BUT, for Linux or Windows:
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print ("Neither Metal Performance Shaders (MPS) nor CUDA device found.")

#%%
import bitsandbytes
tokenizer = AutoTokenizer.from_pretrained(model_id)

# model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_id, 
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    # device_map={"mps": 0}
    device_map={"":"mps"}
    # torch_dtype='auto',
    # device_map='auto',
)
text = "Hello my name is"
inputs = tokenizer(text, return_tensors="pt").to(0)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# %%
