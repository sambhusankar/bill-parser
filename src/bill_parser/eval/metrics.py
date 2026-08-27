from __future__ import annotations
from collections import Counter
from bill_parser.schema import Bill


def _norm(v) -> str:
  if isinstance(v, float):
    return f"{v:.2f}"

  return " ".join(str(v).lower().split())

def flatten(b: Bill) -> Counter:
  c = Counter()
  for it in b.items:
    if it.name:
      c[f"item.name={_norm(it.name)}"] += 1
    if it.price is not None:
      c[f"item.price={_norm(it.price)}"] += 1
    if it.qty is not None:
      c[f"item.qty={_norm(it.qty)}"] += 1
  for k in ("subtotal", "tax", "total"):
    v = getattr(b, k)
    if v is not None:
      c[f"{k}={_norm(v)}"] += 1
  return c

def prf(pred: Bill, gold: Bill) -> tuple[int, int, int]:
  p,g = flatten(pred), flatten(gold)
  tp = sum((p & g).values())
  return tp, sum(p.values()), sum(g.values())

def corpus_f1(pairs: list[tuple[Bill, Bill]]) -> dict:
  tp = np_ = ng = 0
  for pred, gold in pairs:
    a, b, c = prf(pred, gold)
    tp, np_, ng = tp + a, np_ + b, ng + c
  precision = tp/ np_ if np_ else 0.0
  recall = tp / ng if ng else 0.0
  f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
  return {"precision": precision, "recall": recall, "f1": f1, "tp": tp}