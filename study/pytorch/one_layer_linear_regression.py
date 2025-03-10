import torch
import torch.nn as nn
from matplotlib import pyplot as plt

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1, bias=True) # y=ax+b 선형방정식.

    def forward(self, x):
        y = self.linear(x)
        return y

# 데이터 생성...
x = torch.FloatTensor(range(5)).unsqueeze(1)
y = 2*x +  torch.rand(5, 1) # 5x1 matrix

model = LinearRegression()
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_stack = []

for epoch in range(1001):

    optimizer.zero_grad()

    y_hat = model(x)
    loss = criterion(y_hat, y)

    loss.backward()
    optimizer.step()
    loss_stack.append(loss.item())

    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

with torch.no_grad():
    plt.plot(x, y, 'ro', label='Original data')
    plt.plot(x, model(x).detach(), label='Fitted line')
    plt.legend()
    plt.show()