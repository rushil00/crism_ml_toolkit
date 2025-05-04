"""
CRISM Mineral Classification Training Script

Usage:
python train.py --data_path ./data --library_path ./libs --epochs 100 --batch_size 64

Features:
- Combined dataset from image labels and spectral libraries
- Spectral data augmentation
- Attention map visualization
- Metric tracking
- Model checkpointing
"""

import os
import argparse
from crism_ml.train import load_data
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------------------------- Configuration ----------------------------

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='CRISM Mineral Classifier Training')
parser.add_argument('--data_path', type=str, required=True,
                    help='Path to directory with training spectra and labels')
parser.add_argument('--library_path', type=str, default=None,
                    help='Path to spectral library data')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Input batch size for training')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate')
parser.add_argument('--hidden_size', type=int, default=128,
                    help='LSTM hidden state size')
parser.add_argument('--num_layers', type=int, default=2,
                    help='Number of LSTM layers')
parser.add_argument('--num_classes', type=int, required=True,
                    help='Number of mineral classes')
parser.add_argument('--num_bands', type=int, required=True,
                    help='Number of spectral bands in input data')
parser.add_argument('--output_dir', type=str, default='./results',
                    help='Output directory for results')
args = parser.parse_args()

# Create output directories
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'attention_maps'), exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------- Data Preparation ----------------------------

class CRISMDataset(Dataset):
    def __init__(self, spectra, labels, augment=True):
        self.spectra = spectra
        self.labels = labels
        self.augment = augment
        
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        spectrum = self.spectra[idx]
        label = self.labels[idx]
        
        if self.augment:
            spectrum = spectral_augmentation(spectrum)
            
        return spectrum, label

def spectral_augmentation(spectrum):
    """Apply random spectral transformations"""
    # Add Gaussian noise
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, 0.01, spectrum.shape)
        spectrum += noise
        
    # Random wavelength shift
    if np.random.rand() < 0.3:
        shift = np.random.randint(-3, 3)
        spectrum = np.roll(spectrum, shift)
        
    # Random scaling
    if np.random.rand() < 0.3:
        scale = 1 + np.random.uniform(-0.1, 0.1)
        spectrum *= scale
        
    return spectrum

# def load_data(data_path):
#     """Load numpy arrays of spectra and labels"""
#     spectra = np.load(os.path.join(data_path, 'spectra.npy'))
#     labels = np.load(os.path.join(data_path, 'labels.npy'))
#     return spectra, labels

def load_library_data(library_path):
    """Load spectral library data"""
    lib_spectra = []
    lib_labels = []
    for root, _, files in os.walk(library_path):
        for file in files:
            if file.endswith('.npy'):
                data = np.load(os.path.join(root, file))
                lib_spectra.append(data['spectra'])
                lib_labels.append(data['labels'])
    return np.concatenate(lib_spectra), np.concatenate(lib_labels)

# Load main dataset
spectra, labels, _ = load_data(args.data_path)
# spectra = (spectra - np.mean(spectra, axis=1, keepdims=True)) / \
#           (np.std(spectra, axis=1, keepdims=True) + 1e-8)

# Load spectral libraries if provided
datasets = [CRISMDataset(spectra, labels)]
if args.library_path:
    lib_spectra, lib_labels = load_library_data(args.library_path)
    lib_spectra = (lib_spectra - np.mean(lib_spectra, axis=1, keepdims=True)) / \
                 (np.std(lib_spectra, axis=1, keepdims=True) + 1e-8)
    datasets.append(CRISMDataset(lib_spectra, lib_labels, augment=False))

# Combine datasets
full_dataset = ConcatDataset(datasets)
train_idx, val_idx = train_test_split(range(len(full_dataset)), test_size=0.2)

# Create dataloaders
train_loader = DataLoader(
    torch.utils.data.Subset(full_dataset, train_idx),
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True
)
# 
val_loader = DataLoader(
    torch.utils.data.Subset(full_dataset, val_idx),
    batch_size=args.batch_size,
    pin_memory=True
)

# ---------------------------- Model Definition ----------------------------

class SpectralAttentionLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            batch_first=True
        )
        self.attention = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.Tanh(),
            nn.Linear(args.hidden_size, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Linear(args.hidden_size, args.num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(-1)  # Add channel dimension
        out, _ = self.lstm(x)
        attention_weights = self.attention(out).squeeze()
        context = torch.sum(out * attention_weights.unsqueeze(-1), dim=1)
        return self.fc(context), attention_weights

model = SpectralAttentionLSTM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()
# ---------------------------- Training Loop ----------------------------

def save_attention_maps(attention_weights, labels, epoch):
    """Save attention visualizations for each class"""
    epoch_dir = os.path.join(args.output_dir, 'attention_maps', f'epoch_{epoch+1:03d}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    unique_classes = np.unique(labels)
    for cls in unique_classes:
        cls_attention = attention_weights[labels == cls].mean(axis=0)
        plt.figure(figsize=(12, 6))
        plt.plot(range(args.num_bands), cls_attention)
        plt.title(f'Class {cls} Attention - Epoch {epoch+1}')
        plt.xlabel('Spectral Band')
        plt.ylabel('Attention Weight')
        plt.savefig(os.path.join(epoch_dir, f'class_{cls}_attention.jpg'))
        plt.close()

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    """Save training metrics visualization"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Classification Accuracy')
    plt.legend()
    
    plt.savefig(os.path.join(args.output_dir, 'training_metrics.jpg'))
    plt.close()

best_val_acc = 0.0
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(args.epochs):
    # Training phase
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    print("*"*80)
    print(f"EPOCH NUMBER : {epoch}")
    print("*"*80)
    for spectra, labels in train_loader:
        spectra = spectra.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(spectra)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = epoch_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    print("\nEpoch Metrics:")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_attention = []
    all_labels = []
    
    with torch.no_grad():
        for spectra, labels in val_loader:
            spectra = spectra.to(device)
            labels = labels.to(device)
            
            outputs, attention = model(spectra)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            all_attention.append(attention.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    # Print metrics at end of epoch
    print(f"Validation Loss: {val_loss:.4f}") 
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print("\n")
    print("*"*80)
    print("SAVING ATTENTION MAPS")
    print("*"*80)

    # Save attention maps
    save_attention_maps(
        np.concatenate(all_attention),
        np.concatenate(all_labels),
        epoch
    )
    
    # Save metrics plot
    plot_metrics(train_losses, val_losses, train_accs, val_accs)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'accuracy': val_acc,
        }, os.path.join(args.output_dir, 'checkpoints', 'best_model.pth'))
    
    # Print progress
    print(f'Epoch {epoch+1}/{args.epochs}')
    print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
    print(f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')
    print('-'*50)

print(f'Training complete. Best validation accuracy: {best_val_acc:.4f}')