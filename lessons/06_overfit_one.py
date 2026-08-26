import torch
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel
from bill_parser import config

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = DonutProcessor.from_pretrained(config.FINETUNED_REFERENCE_ID)
model = VisionEncoderDecoderModel.from_pretrained(config.FINETUNED_REFERENCE_ID).to(device)

model.config.pad_token_id = model.decoder.config.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s>")


ds = load_dataset(config.DATASET_ID, split="test[:1]")
pixel_values = processor(ds[0]["image"].convert("RGB"), return_tensors="pt").pixel_values.to(device)
target_str = "<s_cord-v2><s_total><s_total_price>10.000</s_total_price></s_total></s>"
labels = processor.tokenizer(target_str, add_special_tokens = False, return_tensors="pt").input_ids
labels[labels == processor.tokenizer.pad_token_id] =- 100
labels = labels.to(device)

opt = torch.optim.AdamW(model.parameters(), lr = 1e-5)
model.train()


for step in range(30):
  opt.zero_grad()
  loss = model(pixel_values=pixel_values, labels=labels).loss
  loss.backward()
  opt.step()
  print(f"step {step:2d}  loss {loss.item():.4f}")