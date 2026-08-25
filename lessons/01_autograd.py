import torch

x = torch.tensor(3.0, requires_grad = True)
print(x)

y = x**2 + 5 * x
print("y          :", y)
print("y.grad_fn", y.grad_fn)


y.backward()
print("x.grad   :", x.grad)

y2 = x**2 + 5 * x
y2.backward()
print("after 2nd: ", x.grad)

x.grad.zero_()
print("zeroed  :", x.grad)