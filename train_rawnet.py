import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils import data
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

from compose_models import RawNetWithFC
from train_data_loader import RawNetDATAReader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model
binary_model = RawNetWithFC(embedding_dim=256, num_classes=2).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(binary_model.fc.parameters(), lr=0.001)  # Only optimize the FC layer


def calculate_eer(labels, scores):
    """
    Calculate the Equal Error Rate (EER).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    return eer, eer_threshold


def evaluate_model(model, val_loader):
    """
    Evaluate the model on the validation set and calculate AUC and EER.
    """
    model.eval()  # Set to evaluation mode
    total = 0
    correct = 0
    val_loss = 0.0
    all_labels = []
    all_scores = []

    progress_bar = tqdm(val_loader, desc="Evaluating", unit="batch", leave=False)

    with torch.no_grad():
        for data in progress_bar:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect scores and labels for metrics
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

            # Update progress bar with current metrics
            progress_bar.set_postfix({
                "Val Loss": f"{val_loss / total:.4f}",
                "Accuracy": f"{100 * correct / total:.2f}%"
            })

    accuracy = 100 * correct / total
    avg_loss = val_loss / total

    # Calculate AUC
    auc = roc_auc_score(all_labels, all_scores)

    # Calculate EER
    eer, eer_threshold = calculate_eer(all_labels, all_scores)

    return accuracy, avg_loss, auc, eer


# Training loop with best model saving
def train_fc_layer(model, train_loader, val_loader, num_epochs=10, save_path="best_model.pth"):
    best_auc = 0.0
    best_eer = 1.0  # Lower is better for EER
    best_model_path = save_path

    for epoch in range(num_epochs):
        model.train()  # Set to training mode
        epoch_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")
        for data in progress_bar:
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Backward pass
            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix({
                "Loss": f"{epoch_loss / total:.4f}",
                "Accuracy": f"{100 * correct / total:.2f}%"
            })

        # Evaluate on the validation set
        val_accuracy, val_loss, auc, eer = evaluate_model(model, val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {100 * correct / total:.2f}% - "
              f"Val Accuracy: {val_accuracy:.2f}% - Val Loss: {val_loss:.4f} - AUC: {auc:.4f} - EER: {eer:.4f}")

        # Save the best model based on AUC
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with AUC: {best_auc:.4f}")

        # Save the best model based on EER (lower is better)
        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with EER: {best_eer:.4f}")

    print(f"Training complete. Best AUC: {best_auc:.4f}, Best EER: {best_eer:.4f}")
    return best_model_path


# Main script
parser = argparse.ArgumentParser()
parser.add_argument('--root', '-rt', type=str, default='../DATASETS/DTIM', help='')
parser.add_argument('--nEpochs', '-epoch', type=int, default=20, help='')
parser.add_argument('--batch_size', '-b', type=int, default=32, help='')
parser.add_argument('--num_workers', '-w', type=int, default=16, help='')
parser.add_argument('--lr', '-lr', type=float, default=0.0001, help='')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0], help='')
args = parser.parse_args()
print(args)

# Example usage
train_dataset = RawNetDATAReader(args=args, split='TRAIN')
train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

dev_dataset = RawNetDATAReader(args=args, split='DEV')
dev_loader = data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

best_model_path = train_fc_layer(binary_model, train_loader, dev_loader, num_epochs=args.nEpochs, save_path="./weights/rawnet_3/best_rawnet3_fc.pth")
print(f"Best model saved at: {best_model_path}")
