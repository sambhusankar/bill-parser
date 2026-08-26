import json
import torch
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel

from bill_parser import config

MODEL_ID = config.FINETUNED_REFERENCE_ID

processor = DonutProcessor.from_pretrained(MODEL_ID)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

def count(m):
  return sum(p.numel() for p in m.parameters())

print(f"total   : {count(model)/1e6:8.1f}M")
print(f"encoder : {count(model.encoder)/1e6:8.1f}M")
print(f"decoder : {count(model.decoder)/1e6:8.1f}M")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable: {trainable/1e6:8.1f}M")


ds = load_dataset(config.DATASET_ID, split="test[:1]")
image = ds[0]["image"].convert("RGB")
target = json.loads(ds[0]["ground_truth"])["gt_parse"]
print("\ntarget json: ", json.dumps(target)[:200])

pixel_values = processor(image, return_tensors="pt").pixel_values

def json2token(obj, sort_json_key=True):
  if isinstance(obj, dict):
    if len(obj) == 1 and "text_sequence" in obj:
      return obj["text_sequence"]
    keys = sorted(obj.keys()) if sort_json_key else obj.keys()
    return "".join(
      f"<s_{k}>{json2token(obj[k], sort_json_key)}</s_{k}>" for k in keys
    )
  if isinstance(obj, list):
    return r"<sep/>".join(json2token(v, sort_json_key) for v in obj)
  return str(obj)

target_str = json2token(target)

labels = processor.tokenizer(
  target_str, add_special_tokens = False, return_tensors = "pt"
).input_ids
labels[labels == processor.tokenizer.pad_token_id] = -100

model.config.pad_token_id = model.decoder.config.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s>")

model.eval()
with torch.no_grad():
  out = model(pixel_values = pixel_values, labels = labels)

print("loss", out.loss.item())
print("logits", out.logits.shape)