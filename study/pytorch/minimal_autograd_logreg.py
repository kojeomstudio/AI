# minimal_autograd_logreg.py
import torch
torch.manual_seed(0)

# 1) 데이터: 2D 가우시안 두 덩어리(이진분류)
n_per_class = 200
mean0, mean1 = torch.tensor([0.0, 0.0]), torch.tensor([2.0, 2.0])
cov = torch.eye(2) * 0.7

x0 = torch.distributions.MultivariateNormal(mean0, cov).sample((n_per_class,))
x1 = torch.distributions.MultivariateNormal(mean1, cov).sample((n_per_class,))
X = torch.cat([x0, x1], dim=0)                  # (400, 2)
y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)])  # (400,)

# 2) 파라미터(직접 정의): W(2→1), b(1)
W = torch.randn(2, 1, requires_grad=True)  # (in_features, out_features)
b = torch.zeros(1, requires_grad=True)

# 3) 하이퍼파라미터
lr = 0.1
epochs = 500

# 4) 학습 루프
for epoch in range(1, epochs + 1):
    # 순전파: 로짓 z = XW + b
    z = X @ W + b  # (N,1)
    # BCEWithLogits: 시그모이드+크로스엔트로피 결합
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z.squeeze(1), y)

    # 역전파
    loss.backward()

    # 경사하강 수동 업데이트
    with torch.no_grad():
        W -= lr * W.grad
        b -= lr * b.grad
    W.grad.zero_()
    b.grad.zero_()

    if epoch % 100 == 0:
        with torch.no_grad():
            pred = (torch.sigmoid(z) > 0.5).float().squeeze(1)
            acc = (pred == y).float().mean().item()
        print(f"[{epoch:03d}] loss={loss.item():.4f} acc={acc:.3f}")