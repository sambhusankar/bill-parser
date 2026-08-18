import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./qwen2.5-0.5b-receipt-json"

# 1. Dummy Dataset Creation (Replace this with your annotated dataset)
sample_data = [
    {
        "ocr_text": "GROCERY MART\nMILK 2.99\nBREAD 1.99\nBANANA 0.79\nTOTAL: 5.77",
        "json_output": json.dumps({
            "items": [
                {"item": "MILK", "price": 2.99},
                {"item": "BREAD", "price": 1.99},
                {"item": "BANANA", "price": 0.79}
            ],
            "total": 5.77
        })
    },
    {
        "ocr_text": "ORGANIC FOODS\nAPPLE JUICE 4.50\nCHICKEN BREAST 8.20\nTAX 0.00\nAMOUNT DUE: 12.70",
        "json_output": json.dumps({
            "items": [
                {"item": "APPLE JUICE", "price": 4.50},
                {"item": "CHICKEN BREAST", "price": 8.20}
            ],
            "total": 12.70
        })
    }
]

# Create HF Dataset
raw_dataset = Dataset.from_list(sample_data * 50)  # Replicated for dummy training loop

# 2. Tokenizer Setup
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

SYSTEM_PROMPT = (
    "You are a receipt extraction system. Extract expense items, item prices, "
    "and total amount from the input text into clean JSON format."
)

# 3. Format dataset into Qwen2.5 Chat Template
def format_prompts(example):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["ocr_text"]},
        {"role": "assistant", "content": example["json_output"]}
    ]
    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": formatted_text}

formatted_dataset = raw_dataset.map(format_prompts)

# 4. BitsAndBytes 4-bit Quantization Config (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True
)

# 5. Load Base Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)

# 6. Configure LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# 7. SFT & Trainer Arguments
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    max_seq_length=512,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    optim="paged_adamw_8bit",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    save_strategy="epoch",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    peft_config=peft_config,
    args=training_args,
)

# 8. Run Training & Save Fine-Tuned Adapter
trainer.train()
trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
print("Training complete! Adapter saved to:", f"{OUTPUT_DIR}/final_adapter")