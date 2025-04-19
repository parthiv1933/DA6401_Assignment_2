import torch
import argparse
import wandb
import math
from torch import nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# --------- Argument Parsing ---------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="CNN Training Script")
    parser.add_argument('--wandb_project', type=str, default='DL_A2')
    parser.add_argument('--wandb_entity', type=str, default='DL_A2')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--activation', type=str, default='relu', choices=['selu', 'mish', 'leakyrelu', 'relu', 'gelu'])
    parser.add_argument('--filters', type=int, default=32)
    parser.add_argument('--filter_scheme', type=str, default='double', choices=['same', 'half', 'double'])
    parser.add_argument('--augment', type=str2bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batchnorm', type=str2bool, default=True)
    parser.add_argument('--dense_units', type=int, default=256)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--img_dim', type=int, default=256)
    parser.add_argument('--train_dir', type=str, default='./data/train')
    parser.add_argument('--test_dir', type=str, default='./data/val')
    return parser.parse_args()

# --------- Data Preparation ---------
def build_loaders(cfg, mode):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if mode == 'train':
        if cfg.augment:
            train_tfms = transforms.Compose([
                transforms.RandomHorizontalFlip(0.3),
                transforms.RandomRotation(12),
                transforms.Resize((cfg.img_dim, cfg.img_dim)),
                transforms.ToTensor(),
                norm
            ])
        else:
            train_tfms = transforms.Compose([
                transforms.Resize((cfg.img_dim, cfg.img_dim)),
                transforms.ToTensor(),
                norm
            ])
        dataset = datasets.ImageFolder(cfg.train_dir, transform=train_tfms)
        n = len(dataset)
        n_train = math.ceil(0.8 * n)
        n_val = n - n_train
        train_set, val_set = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
        return train_loader, val_loader
    else:
        test_tfms = transforms.Compose([
            transforms.Resize((cfg.img_dim, cfg.img_dim)),
            transforms.ToTensor(),
            norm
        ])
        test_set = datasets.ImageFolder(cfg.test_dir, transform=test_tfms)
        test_loader = DataLoader(test_set, batch_size=cfg.batch_size)
        return test_loader

# --------- Model Definition ---------
class ConvNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        filter_plan = self._make_filter_plan(cfg.filters, cfg.filter_scheme)
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.drop_layers = nn.ModuleList()
        self.act_fn = self._get_activation(cfg.activation)
        in_ch = 3
        size = cfg.img_dim
        for out_ch in filter_plan:
            self.conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=cfg.kernel_size))
            self.bn_layers.append(nn.BatchNorm2d(out_ch) if cfg.batchnorm else None)
            self.pool_layers.append(nn.MaxPool2d(2, 2))
            self.drop_layers.append(nn.Dropout(cfg.dropout))
            size = size - (cfg.kernel_size - 1)
            size = size // 2
            in_ch = out_ch
        flat_dim = size * size * filter_plan[-1]
        self.fc1 = nn.Linear(flat_dim, cfg.dense_units)
        self.fc1_act = self._get_activation(cfg.activation)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(cfg.dense_units, 10)
        self.out_act = nn.LogSoftmax(dim=1)

    def _get_activation(self, name):
        return {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'selu': nn.SELU(),
            'mish': nn.Mish()
        }[name]

    def _make_filter_plan(self, base, scheme):
        if scheme == 'same':
            return [base] * 5
        if scheme == 'double':
            return [base * (2 ** i) for i in range(5)]
        if scheme == 'half':
            return [max(base // (2 ** i), 1) for i in range(5)]

    def forward(self, x):
        for i in range(5):
            x = self.conv_layers[i](x)
            x = self.act_fn(x)
            if self.bn_layers[i] is not None:
                x = self.bn_layers[i](x)
            x = self.pool_layers[i](x)
            x = self.drop_layers[i](x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc1_act(x)
        x = self.fc1_drop(x)
        x = self.fc2(x)
        return self.out_act(x)

# --------- Training Loop ---------
def train_model():
    wandb.init()
    cfg = wandb.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvNet(cfg).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.NLLLoss()
    train_loader, val_loader = build_loaders(cfg, 'train')

    for epoch in range(cfg.epochs):
        model.train()
        train_loss, train_hits, train_count = 0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_hits += (preds.argmax(1) == lbls).sum().item()
            train_count += lbls.size(0)
        model.eval()
        val_loss, val_hits, val_count = 0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds = model(imgs)
                loss = loss_fn(preds, lbls)
                val_loss += loss.item()
                val_hits += (preds.argmax(1) == lbls).sum().item()
                val_count += lbls.size(0)
        wandb.log({
            "epoch": epoch + 1,
            "tr_accuracy": train_hits / train_count,
            "val_accuracy": val_hits / val_count,
            "tr_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader)
        })
        print(f"Epoch {epoch+1}: Train Acc {train_hits/train_count:.4f}, Val Acc {val_hits/val_count:.4f}")

    # Test phase
    test_loader = build_loaders(cfg, 'test')
    correct, total = 0, 0
    with torch.no_grad():
        model.eval()
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs)
            correct += (preds.argmax(1) == lbls).sum().item()
            total += lbls.size(0)
    print(f"Test Accuracy: {correct/total:.4f}")

# --------- Main Entry Point ---------
if __name__ == "__main__":
    args = parse_args()
    sweep_cfg = {
        "method": "grid",
        "name": "CNN Sweep",
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "parameters": {
            "augment": {"values": [args.augment]},
            "batchnorm": {"values": [args.batchnorm]},
            "filters": {"values": [args.filters]},
            "filter_scheme": {"values": [args.filter_scheme]},
            "dropout": {"values": [args.dropout]},
            "activation": {"values": [args.activation]},
            "batch_size": {"values": [args.batch_size]},
            "learning_rate": {"values": [args.learning_rate]},
            "epochs": {"values": [args.epochs]},
            "img_dim": {"values": [args.img_dim]},
            "kernel_size": {"values": [args.kernel_size]},
            "dense_units": {"values": [args.dense_units]},
            "train_dir": {"values": [args.train_dir]},
            "test_dir": {"values": [args.test_dir]},
        }
    }
    wandb.login()
    sweep_id = wandb.sweep(sweep_cfg, project=args.wandb_project)
    wandb.agent(sweep_id, function=train_model)
    wandb.finish()
