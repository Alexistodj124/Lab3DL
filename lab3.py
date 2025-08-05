import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import itertools

class MLP(nn.Module):
    def __init__(self, hidden_layers, activation='relu', num_classes=10):
        super().__init__()
        layers = []
        in_features = 28 * 28
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Activación {activation} no soportada")
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

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    val_size = 10000
    train_size = len(full_train) - val_size
    train_subset, val_subset = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    param_grid = {
        'lr':         [0.005, 0.01, 0.02],
        'batch_size': [64, 128],
        'hidden1':    [256],
        'hidden2':    [128],
        'activation': ['relu', 'tanh']
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []

    for combo in combinations:
        train_loader = DataLoader(
            train_subset,
            batch_size=combo['batch_size'],
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=1000,
            shuffle=False,
            num_workers=2
        )

        hidden_layers = [combo['hidden1'], combo['hidden2']]
        model = MLP(hidden_layers, activation=combo['activation']).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=combo['lr'], momentum=0.9)

        for _ in range(2):
            train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        results.append({
            'lr': combo['lr'],
            'batch_size': combo['batch_size'],
            'hidden1': combo['hidden1'],
            'hidden2': combo['hidden2'],
            'activation': combo['activation'],
            'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 4)
        })

    results.sort(key=lambda x: x['val_acc'], reverse=True)

    from pprint import pprint
    print("Resultados de Grid Search (ordenados por val_acc):")
    pprint(results)

    print("\nMejor combinación:")
    pprint(results[0])

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
