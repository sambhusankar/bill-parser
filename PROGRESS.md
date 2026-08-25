# Receipt Parser — Progress

**Goal:** receipt image → JSON (items, prices, total). Learning-first, production-shaped.
**Approach:** fine-tune Donut (Swin encoder + BART decoder) on CORD. Chosen over
OCR+rules (brittle) and LayoutLMv3 (two-stage, error propagation).

## Phases
- [x] 0 — Environment: uv + repo + Colab, smoke test on pretrained donut-cord-v2 ✅
- [ ] 1 — Foundations: tensors, forward/backward, loss, fine-tuning  ← NEXT
- [ ] 2 — Baseline v0: OCR + heuristics (the number to beat)
- [ ] 3 — Donut architecture: attention, cross-attention, JSON-as-tokens
- [ ] 4 — Data: CORD loading, ground truth → token sequences
- [ ] 5 — Fine-tune: Trainer + one manual loop, loss curves
- [ ] 6 — Evaluate: field F1, tree-edit accuracy, error analysis
- [ ] 7 — Deploy: FastAPI + UI, latency, real-photo domain shift
- [ ] 8 — Perspective: compare vs. multimodal LLM API

## Environment
Python 3.11 · torch 2.x · transformers 5.x (NOT v4 — most blog tutorials are stale)
Local = code/tests/inference · Colab T4 = training only, clones this repo

## Open questions
- (none yet)