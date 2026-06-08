import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import pytorch_lightning as pl
from model import MNISTResNet18
import os

def train():
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet standard
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load Dataset
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # MNIST train=True has 60,000 images. train=False has 10,000 images.
    # Total 70,000 images.
    # 60% = 42,000
    # 20% = 14,000
    # 20% = 14,000
    
    # We combine them first to get the exact split requested
    combined_dataset = torch.utils.data.ConcatDataset([full_dataset, test_dataset])
    total_size = len(combined_dataset)
    
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    
    train_set, val_set, test_set = random_split(combined_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=64, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=64, num_workers=2)

    # Model
    model = MNISTResNet18()

    # Trainer
    # Limit epochs for demonstration
    trainer = pl.Trainer(max_epochs=5, accelerator="auto", devices=1)
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    trainer.test(model, test_loader)

    # Save weights
    torch.save(model.state_dict(), 'model_weights.pth')
    print("Model saved to model_weights.pth")

if __name__ == '__main__':
    train()
