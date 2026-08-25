import re
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
from datasets import load_dataset

ds = load_dataset("naver-clova-ix/cord-v2", split="test[:1]")
print(ds)
image: Image.Image = ds[0]["image"].convert("RGB")
print(f"PIL image size (W, H) = {image.size}")


MODEL_ID =  "naver-clova-ix/donut-base-finetuned-cord-v2"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | device = {device}")

print("loading processor...")
processor = DonutProcessor.from_pretrained(MODEL_ID, token=False)

print("loading model...")
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID, token=False).to(device)

print("evaluating model...")
model.eval()


pixel_values = processor(image, return_tensors="pt").pixel_values
print(f"tensor shape = {tuple(pixel_values)}")

task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(
  task_prompt, add_special_tokens = False, return_tensors = "pt"
).input_ids

with torch.no_grad():
  outputs = model.generate(
    pixel_values.to(device),
    decoder_input_ids = decoder_input_ids.to(device),
    max_length = model.decoder.config.max_position_embeddings,
    pad_token_id = processor.tokenizer.pad_token_id,
    eos_token_id = processor.tokenizer.eos_token_id,
    use_cache = True,
    bad_words_ids = [[processor.tokenizer.unk_token_id]],
    return_dict_in_generate = True
  )

seq = processor.batch_decode(outputs.sequences)[0]

seq = seq.replace(processor.tokenizer.eos_token, "").replace(
  processor.tokenizer.pad_token, ""
)

seq = re.sub(r"<.*?>", "", seq, count = 1).strip()