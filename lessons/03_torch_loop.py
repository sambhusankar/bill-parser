import torch
import torch.nn as nn

torch.manual_seed(42)
X = torch.randn(100, 1) * 10
y_true = 3 * X + 2 + torch.randn(100, 1) * 0.5

model = nn.Linear(1, 1)

loss_fn = nn.MSELoss()

opt = torch.optim.SGD(model.parameters(), lr = 0.01)

for step in range(2000):
  opt.zero_grad()
  y_pred = model(X)
  loss = loss_fn(y_pred, y_true)
  loss.backward()
  opt.step()

  if step % 200 == 0:
    print(f"step {step:4d} loss {loss.item():8.4f}")

print(dict(model.named_parameters()))