import argparse, json
from datasets import load_dataset
from bill_parser import config
from bill_parser.baseline.pipeline import image_to_bill
from bill_parser.data.cord import cord_to_bill
from bill_parser.eval.metrics import corpus_f1

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--split", default="test")
  ap.add_argument("--limit", type= int, default=None)
  args = ap.parse_args()

  split = f"{args.split}[:{args.limit}]" if args.limit else args.split
  ds = load_dataset(config.DATASET_ID, split=split)

  pairs, dump = [], []
  for i, row in enumerate(ds):
    gold = cord_to_bill(row["ground_truth"])
    pred = image_to_bill(row["image"])
    pairs.append((pred, gold))
    dump.append({"i":i, "pred": pred.model_dump(), "gold": gold.model_dump()})
    print(f"\r{i + 1}/{len(ds)}", end="", flush=True)

  out = config.OUTPUTS_DIR / "baseline_predictions.json"
  out.write_text(json.dumps(dump, indent=2))
  print("\n", json.dumps(corpus_f1(pairs), indent=2), sep="")
  print("wrote", out)

if __name__ == "__main__":
  main()