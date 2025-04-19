import torch
import wandb
import math
import argparse
from torch import nn
from torch.optim import Adam
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Training Script")
    parser.add_argument('--wandb_project', type=str, default='DL_A2', help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default='DL_A2', help='Wandb entity')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--strategy', type=str, default='k_freeze', choices=['all_freeze', 'k_freeze', 'no_freeze'], help='Layer freezing strategy')
    parser.add_argument('--layers_to_freeze', type=int, default=15, help='Number of layers to freeze (if k_freeze)')
    parser.add_argument('--train_data_dir', type=str, default='./data/train', help='Training data directory')
    parser.add_argument('--test_data_dir', type=str, default='./data/val', help='Testing data directory')
    return parser.parse_args()

def prepare_dataloaders(config, phase):
    if phase == 'train':
        augmentation = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomRotation(12),
            transforms.ColorJitter(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        dataset = datasets.ImageFolder(root=config['train_data_dir'], transform=augmentation)
        total = len(dataset)
        train_size = math.ceil(total * 0.8)
        val_size = total - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)
        return train_loader, val_loader
    else:
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        test_set = datasets.ImageFolder(root=config['test_data_dir'], transform=test_transform)
        test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)
        return test_loader

def configure_model(cfg):
    net = models.googlenet(pretrained=True)
    num_classes = 10
    if cfg.strategy == 'all_freeze':
        for name, param in net.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False
    elif cfg.strategy == 'k_freeze':
        layers = list(net.children())[:cfg.layers_to_freeze]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net

def run_training():
    wandb.init()
    params = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = configure_model(params)
    net = net.to(device)
    optimizer = Adam(net.parameters())
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader = prepare_dataloaders(params, 'train')

    for epoch in range(params.epochs):
        net.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = net(imgs)
            loss = criterion(preds, lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (preds.argmax(1) == lbls).sum().item()
            train_total += lbls.size(0)

        net.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = net(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == lbls).sum().item()
                val_total += lbls.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Train Acc {train_acc:.4f}, Val Loss {avg_val_loss:.4f}, Val Acc {val_acc:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "tr_accuracy": train_acc,
            "val_accuracy": val_acc,
            "tr_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

    # Testing phase
    test_loader = prepare_dataloaders(params, 'test')
    net.eval()
    test_correct, test_total = 0, 0
    all_targets, all_preds = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = net(imgs)
            preds = outputs.argmax(1)
            all_targets.extend(lbls.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            test_correct += (preds == lbls).sum().item()
            test_total += lbls.size(0)
    print(f"Test Samples: {test_total}, Correct: {test_correct}, Accuracy: {100 * test_correct / test_total:.2f}%")

if __name__ == "__main__":
    args = parse_args()
    sweep_conf = {
        "method": "grid",
        "name": "GoogLeNet Sweep",
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "parameters": {
            "batch_size": {"values": [args.batch_size]},
            "epochs": {"values": [args.epochs]},
            "strategy": {"values": [args.strategy]},
            "layers_to_freeze": {"values": [args.layers_to_freeze]},
            "train_data_dir": {"values": [args.train_data_dir]},
            "test_data_dir": {"values": [args.test_data_dir]}
        }
    }
    sweep_id = wandb.sweep(sweep_conf, project=args.wandb_project)
    wandb.agent(sweep_id, function=run_training)
    wandb.finish()
