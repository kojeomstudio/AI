# torch_mlp_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)

# --- 1. 디바이스 자동 선택 ---
if torch.backends.mps.is_available():
    device = torch.device("mps")  # macOS M1/M2/M3 GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
# 1) 데이터(동일)
n_per_class = 400
mean0, mean1 = torch.tensor([0.0, 0.0]), torch.tensor([2.2, 2.2])
cov = torch.eye(2) * 0.9

x0 = torch.distributions.MultivariateNormal(mean0, cov).sample((n_per_class,))
x1 = torch.distributions.MultivariateNormal(mean1, cov).sample((n_per_class,))
X = torch.cat([x0, x1], dim=0)                  # (800, 2)
y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)])  # (800,)

# Train/Val 분할
idx = torch.randperm(X.size(0))
train_size = int(0.8 * X.size(0))
train_idx, val_idx = idx[:train_size], idx[train_size:]
X_train, y_train = X[train_idx].to(device), y[train_idx].to(device)
X_val, y_val = X[val_idx].to(device), y[val_idx].to(device)

# 2) 모델 정의: 간단한 MLP (2→16→16→1)
class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),  # 로짓 출력
        )
    def forward(self, x):
        return self.net(x).squeeze(1)  # (N,)

model = MLP().to(device)

# 3) 손실/옵티마이저
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# 4) 미니배치 훈련 루프 (Dataset/Dataloader 없이 수동 배치)
epochs = 200
batch_size = 64

def accuracy(logits, targets):
    preds = (torch.sigmoid(logits) > 0.5).float()
    return (preds == targets).float().mean().item()

for epoch in range(1, epochs + 1):
    model.train()
    # 셔플
    perm = torch.randperm(X_train.size(0), device=device)
    total_loss = 0.0

    for start in range(0, X_train.size(0), batch_size):
        idx = perm[start:start+batch_size]
        xb, yb = X_train[idx], y_train[idx]

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    # 검증
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_loss = criterion(val_logits, y_val).item()
        val_acc = accuracy(val_logits, y_val)

    if epoch % 20 == 0:
        print(f"[{epoch:03d}] train_loss={total_loss/X_train.size(0):.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")