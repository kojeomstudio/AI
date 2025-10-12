# dataset_dataloader_template.py
import torch
from torch.utils.data import Dataset, DataLoader
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
    
class ToyGaussian2C(Dataset):
    def __init__(self, n_per_class=1000):
        mean0, mean1 = torch.tensor([0.0, 0.0]), torch.tensor([2.5, 2.5])
        cov = torch.eye(2) * 1.0
        x0 = torch.distributions.MultivariateNormal(mean0, cov).sample((n_per_class,))
        x1 = torch.distributions.MultivariateNormal(mean1, cov).sample((n_per_class,))
        self.X = torch.cat([x0, x1], dim=0)
        self.y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)])
    def __len__(self):
        return self.X.size(0)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 데이터셋/로더
full_ds = ToyGaussian2C(n_per_class=1500)
n_total = len(full_ds)
n_train = int(0.8 * n_total)
n_val = n_total - n_train
train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=256)

# 모델(MLP)
class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

model = MLP().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-3)

def accuracy_from_logits(logits, targets):
    return ((torch.sigmoid(logits) > 0.5).float() == targets).float().mean().item()

# 학습 루프
epochs = 50
for epoch in range(1, epochs+1):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)

    model.eval()
    with torch.no_grad():
        vloss, vacc, vcount = 0.0, 0.0, 0
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            vloss += criterion(logits, yb).item() * xb.size(0)
            vacc  += accuracy_from_logits(logits, yb) * xb.size(0)
            vcount += xb.size(0)
    if epoch % 10 == 0:
        print(f"[{epoch:03d}] train_loss={running/n_train:.4f} "
              f"val_loss={vloss/vcount:.4f} val_acc={vacc/vcount:.3f}")