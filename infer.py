import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./qwen2.5-0.5b-receipt-json/final_adapter"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

sample_receipt = """
CORNER BAKERY
2X CROISSANT 5.00
1X COFFEE 3.50
SUBTOTAL 8.50
TAX 0.70
TOTAL 9.20
"""

messages = [
    {"role": "system", "content": "You are a receipt extraction system. Extract expense items, item prices, and total amount from the input text into clean JSON format."},
    {"role": "user", "content": sample_receipt}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
    
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)