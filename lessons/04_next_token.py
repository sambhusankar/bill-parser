import torch
import torch.nn as nn

torch.manual_seed(0)

VOCAB = ["<s>", "a", "b", "c", "</s>"]
V = len(VOCAB)

seq = [0, 1, 2, 3, 4]

inputs = torch.tensor([seq[:-1]])
labels = torch.tensor([seq[1:]])

class TinyDecoder(nn.Module):
  def __init__(self, V, d=16):
    super().__init__()
    self.emb = nn.Embedding(V, d)
    self.rnn = nn.GRU(d, d, batch_first=True)
    self.head = nn.Linear(d, V)

  def forward(self, X):
    h, _ = self.rnn(self.emb(X))
    return self.head(h)

model = TinyDecoder(V)
opt = torch.optim.Adam(model.parameters(), lr = 0.05)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

for step in range(201):
  opt.zero_grad()
  logits = model(inputs)
  loss = loss_fn(logits.reshape(-1, V), labels.reshape(-1))
  loss.backward()
  opt.step()

  if step % 50 == 0:
    print(f"step {step:3d} loss {loss.item():.4f}")


model.eval()
ids = [0]

with torch.no_grad():
  for _ in range(10):
    logits = model(torch.tensor([ids]))
    next_id = logits[0, -1].argmax().item()
    ids.append(next_id)
    if next_id == 4:
      break

print("generated: ", [VOCAB[i] for i in ids])