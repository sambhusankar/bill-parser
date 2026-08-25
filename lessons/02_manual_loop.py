import torch

torch.manual_seed(42)

X = torch.rand(100, 1) * 10
y_true = 3 * X + 2 + torch.randn(100, 1) * 0.5

w = torch.zeros(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.01

for step in range(2000):
  y_pred = X @ w + b

  loss = ((y_pred - y_true) **2 ).mean()

  loss.backward()

  with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad


  w.grad.zero_()
  b.grad.zero_()

  if step % 200 == 0:
    print(f"step {step:4d} loss {loss.item():8.4f} w {w.item():.3f} b {b.item():.3f}")

print(f"learned: y = {w.item():.3f}x + {b.item():.3f}  (target: 3x + 2)")