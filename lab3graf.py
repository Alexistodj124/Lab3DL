import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, hidden_layers, activation='tanh', num_classes=10):
        super().__init__()
        layers = []
        in_features = 28 * 28
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            if activation == 'relu':
                layers.append(nn.ReLU())
            else:  # tanh
                layers.append(nn.Tanh())
            in_features = h
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item() * X.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
val_size = 10000
train_size = len(full_train) - val_size
train_subset, val_subset = random_split(full_train, [train_size, val_size],
                                        generator=torch.Generator().manual_seed(42))
batch_size = 64
num_epochs = 5

learning_rates = [0.005, 0.01, 0.02]

history = {lr: {'train_loss': [], 'val_loss': [], 'val_acc': []} for lr in learning_rates}

for lr in learning_rates:
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_subset,   batch_size=1000,    shuffle=False, num_workers=0)
    model = MLP([256, 128], activation='tanh').to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, num_epochs+1):
        tl = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = evaluate(model, val_loader, criterion, device)
        history[lr]['train_loss'].append(tl)
        history[lr]['val_loss'].append(vl)
        history[lr]['val_acc'].append(va)

epochs = list(range(1, num_epochs+1))

plt.figure()
for lr in learning_rates:
    plt.plot(epochs, history[lr]['train_loss'], label=f"LR={lr}")
plt.title("Training Loss por LR")
plt.xlabel("Época")
plt.ylabel("Train Loss")
plt.legend()
plt.show()

plt.figure()
for lr in learning_rates:
    plt.plot(epochs, history[lr]['val_loss'], label=f"LR={lr}")
plt.title("Validation Loss por LR")
plt.xlabel("Época")
plt.ylabel("Val Loss")
plt.legend()
plt.show()

plt.figure()
for lr in learning_rates:
    plt.plot(epochs, history[lr]['val_acc'], label=f"LR={lr}")
plt.title("Validation Accuracy por LR")
plt.xlabel("Época")
plt.ylabel("Val Acc")
plt.legend()
plt.show()

