import torch

x = torch.tensor(2.0, requires_grad=True)  # 미분 가능 설정

y = x ** 2  # y = x^2
y.backward()  # dy/dx 자동 계산

print(f" dx / dy = {x.grad} (from y=x^2)")  # dy/dx = 2x = 4.0
