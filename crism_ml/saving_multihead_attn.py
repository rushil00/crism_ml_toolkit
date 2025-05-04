import os
import argparse
from crism_ml.train import load_data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.io

# ---------------------------- Configuration ----------------------------
parser = argparse.ArgumentParser(description='CRISM Mineral Classifier Training')
parser.add_argument('--data_path', type=str, required=True, help='Path to .mat data file')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Input batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_size', type=int, default=128, help='LSTM hidden state size')
parser.add_argument('--num_classes', type=int, required=True, help='Number of mineral classes')
parser.add_argument('--num_bands', type=int, required=True, help='Number of spectral bands (will be padded/truncated)')
parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.output_dir, exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------- Data Loading & Processing ----------------------------
# Modified Dataset Class
class CRISMDataset(Dataset):
    def __init__(self, spectra, labels, augment=True, target_length=248):
        self.spectra = process_spectra(spectra, target_length)
        self.labels = labels
        self.augment = augment
        self.target_length = target_length
        
        # Normalize each spectrum individually
        self.spectra = (self.spectra - np.mean(self.spectra, axis=1, keepdims=True)) / \
                      (np.std(self.spectra, axis=1, keepdims=True) + 1e-8)
        
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        spectrum = self.spectra[idx]
        label = self.labels[idx]
        
        if self.augment:
            spectrum = spectral_augmentation(spectrum, self.target_length)
            
        return torch.tensor(spectrum), torch.tensor(label)

def load_matlab_data(file_path):
    """Load data from MATLAB .mat file"""
    mat = scipy.io.loadmat(file_path)
    spectra = mat['data']  # Adjust these keys based on your .mat structure
    labels = mat['labels'].squeeze()
    return spectra, labels

# ---------------------------- Model Architecture ----------------------------
class SpectralAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=args.hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.attention = nn.Sequential(
            nn.Linear(args.hidden_size*2, args.hidden_size),
            nn.Tanh(),
            nn.Linear(args.hidden_size, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Linear(args.hidden_size*2, args.num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(-1)  # Add channel dimension
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out).squeeze()
        context = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return self.fc(context), attention_weights
    
    
# ---------------------------- Attention Map Saving ----------------------------
def save_attention_maps(attention_weights, labels, epoch, mask=None):
    """Save attention visualizations with padding handling"""
    os.makedirs(os.path.join(args.output_dir, 'attention_maps', f'epoch_{epoch+1:03d}'), exist_ok=True)
    
    # Convert to numpy and handle padding
    attn_weights = attention_weights.cpu().numpy()
    if mask is not None:
        mask = mask.cpu().numpy()
        attn_weights = attn_weights * mask[:, None, :]  # Apply padding mask
    
    # Average across attention heads and batch
    attn_weights = attn_weights.mean(axis=(0,1))  # [seq_len]
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(args.num_bands), attn_weights[:args.num_bands])
    plt.title(f'Attention Weights - Epoch {epoch+1}')
    plt.xlabel('Spectral Band')
    plt.ylabel('Attention Weight')
    plt.savefig(os.path.join(args.output_dir, 'attention_maps', 
                            f'epoch_{epoch+1:03d}', 'combined_attention.jpg'))
    plt.close()

    # Class-specific attention
    unique_classes = np.unique(labels)
    for cls in unique_classes:
        cls_mask = labels == cls
        if np.sum(cls_mask) > 0:  # Only plot if class exists in batch
            cls_attention = attn_weights[cls_mask].mean(axis=0)
            plt.figure(figsize=(12, 6))
            plt.plot(range(args.num_bands), cls_attention[:args.num_bands])
            plt.title(f'Class {cls} Attention - Epoch {epoch+1}')
            plt.xlabel('Spectral Band')
            plt.ylabel('Attention Weight')
            plt.savefig(os.path.join(args.output_dir, 'attention_maps', 
                                   f'epoch_{epoch+1:03d}', f'class_{cls}_attention.jpg'))
            plt.close()

# ---------------------------- Metric Plotting ----------------------------
def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Accuracy Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig(os.path.join(args.output_dir, 'training_metrics.png'))
    plt.close()

# ---------------------------- Checkpointing ----------------------------
def save_checkpoint(epoch, model, optimizer, scheduler, val_acc, is_best):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_acc': val_acc
    }
    
    # Save regular checkpoint
    torch.save(state, os.path.join(args.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth'))
    
    # Save best separately
    if is_best:
        torch.save(state, os.path.join(args.output_dir, 'checkpoints', 'best_model.pth'))

def process_spectra(spectra, target_length=248):
    """Pad or truncate spectra to target length"""
    processed = []
    for spec in spectra:
        if len(spec) < target_length:
            # Pad with last value
            padded = np.pad(spec, (0, target_length - len(spec)), mode='edge')
            processed.append(padded)
        elif len(spec) > target_length:
            # Truncate to target length
            processed.append(spec[:target_length])
        else:
            processed.append(spec)
    return np.array(processed)

# Modified Spectral Augmentation
def spectral_augmentation(spectrum, target_length):
    """Augmentation that maintains spectrum length"""
    # Add Gaussian noise
    if np.random.rand() < 0.3:
        spectrum += np.random.normal(0, 0.01, spectrum.shape)
        
    # Random scaling
    if np.random.rand() < 0.3:
        scale = 1 + np.random.uniform(-0.1, 0.1)
        spectrum *= scale
        
    # Random shift (maintains length)
    if np.random.rand() < 0.3:
        max_shift = min(10, target_length//10)
        shift = np.random.randint(-max_shift, max_shift)
        spectrum = np.roll(spectrum, shift)
        
    return spectrum

# Custom Collate Function
def pad_collate(batch):
    """Handle variable-length sequences with padding"""
    from torch.nn.utils.rnn import pad_sequence
    spectra, labels = zip(*batch)
    
    # Convert to tensors and pad
    spectra_pad = pad_sequence(spectra, batch_first=True)
    labels = torch.stack(labels)
    
    return spectra_pad, labels

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# Modified Training Initialization
def initialize_training():
    # Load MAT file data
    spectra, labels, _ = load_data(args.data_path)
    
    # Create datasets
    train_spectra, val_spectra, train_labels, val_labels = train_test_split(
        spectra, labels, test_size=0.2, random_state=42
    )
    
    train_dataset = CRISMDataset(train_spectra, train_labels)
    val_dataset = CRISMDataset(val_spectra, val_labels, augment=False)
    
    # Create DataLoaders with custom collate
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pad_collate,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=pad_collate,
        pin_memory=True
    )
    
    # Rest of initialization remains the same...
    model = SpectralHybridModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr*10,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs
    )
    criterion = FocalLoss()
    
    return model, optimizer, scheduler, criterion, train_loader, val_loader

# ---------------------------- Modified Training Loop ----------------------------
def main():
    model, optimizer, scheduler, criterion, train_loader, val_loader = initialize_training()
    
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(args.epochs):
        # Training Phase
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        
        for spectra, labels in train_loader:
            spectra = spectra.to(device)
            labels = labels.to(device)
            
            # Create padding mask
            mask = (spectra != 0).any(dim=-1).float()  # Assuming 0-padding
            
            optimizer.zero_grad()
            outputs, attn_weights = model(spectra)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_attention, all_labels, all_masks = [], [], []
        
        with torch.no_grad():
            for spectra, labels in val_loader:
                spectra = spectra.to(device)
                labels = labels.to(device)
                mask = (spectra != 0).any(dim=-1).float()
                
                outputs, attn_weights = model(spectra)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_attention.append(attn_weights)
                all_labels.append(labels.cpu().numpy())
                all_masks.append(mask.cpu().numpy())
        
        # Process metrics
        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Save metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Save visualizations
        save_attention_maps(
            torch.cat(all_attention),
            np.concatenate(all_labels),
            epoch,
            mask=torch.cat(all_masks)
        )
        plot_metrics(train_losses, val_losses, train_accs, val_accs)
        
        # Checkpointing
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        save_checkpoint(epoch, model, optimizer, scheduler, val_acc, is_best)
        
        # Print progress
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        print(f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')
        print(f'Best Val Acc: {best_val_acc:.4f}')
        print('-'*50)

    print(f'\nTraining Complete. Best Validation Accuracy: {best_val_acc:.4f}')

class ResidualAttentionBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        attn_weights = self.attention(x)
        return self.norm(x + x * attn_weights)


# ---------------------------- Model Architecture Adjustments ----------------------------
class SpectralHybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN Pathway
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM Pathway
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=args.hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=args.hidden_size*2,
            num_heads=4,
            batch_first=True
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualAttentionBlock(args.hidden_size*2) for _ in range(3)]
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(64*(args.num_bands//4) + args.hidden_size*2, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, args.num_classes)
        )

    def forward(self, x):
        # CNN Features
        cnn_out = self.cnn(x.unsqueeze(1))  # [batch, channels, seq]
        cnn_feat = cnn_out.view(x.size(0), -1)  # Flatten
        
        # LSTM Features
        lstm_out, _ = self.lstm(x.unsqueeze(-1))  # [batch, seq, features]
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.res_blocks(attn_out)
        lstm_feat = torch.mean(attn_out, dim=1)
        
        # Combine features
        combined = torch.cat([cnn_feat, lstm_feat], dim=1)
        return self.fc(combined), attn_out

if __name__ == '__main__':
    main()