import os
import argparse
import json

import yaml
from compose_models import RawNetWithFC
from models.aasist.AASIST import Model_ASSIST
from models.cnns.acnn1d import ACNN
from models.cnns.cnn1d import CNN
from models.cnns.crnn1d import CRNN
from models.rawboost.RawBoost import RawNet
from models.rawnet.RawNet2 import RawNet2
from models.resnet1d.multi_scale_ori import MSResNet
from models.resnet1d.resnet1d import ResNet1D
from models.rsm1d.RSM1D_large import DilatedNet_L, SSDNet1D_L
from models.rsm1d.RSM1D_small import DilatedNet_S, SSDNet1D_S
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils import data
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

from train_data_loader import RawNetDATAReader



scaler = torch.GradScaler()


def calculate_eer(labels, scores):
    """
    Calculate the Equal Error Rate (EER).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    return eer, eer_threshold


def evaluate_model(model, criterion, val_loader):
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
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                # outputs = outputs[1]         # outputs[1] for aassit
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
    all_scores = np.nan_to_num(all_scores, nan=0.0)


    # if len(all_scores) == 0 or len(np.unique(all_scores)) < 2:
    #     print("Insufficient valid data for AUC/EER calculation")
    #     auc = 0.0
    #     eer = 1.0
    # else:
        # Calculate AUC
    auc = roc_auc_score(all_labels, all_scores)
    # Calculate EER
    eer, eer_threshold = calculate_eer(all_labels, all_scores)

    return accuracy, avg_loss, auc, eer


# Training loop with best model saving
def train_model(model, criterion, optimizer , train_loader, val_loader, num_epochs, save_path,save_every):
    best_auc = 0.0
    best_eer = 1.0  # Lower is better for EER
    best_model_path = save_path
    # Loss function and optimizer


    for epoch in range(num_epochs):
        model.train()  # Set to training mode
        epoch_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")
        for audio_data, label in progress_bar:
            inputs, labels = audio_data.to(device), label.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):  # Mixed precision forward pass
                # Forward pass
                print(f"Input Size: {inputs.shape}")
                outputs = model(inputs)
                # outputs = outputs[1]         # outputs[1] for aassit
                print(f"Output shape: {outputs.shape}")
                print(f"Labels shape: {labels.shape}")

                loss = criterion(outputs, labels)      
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)  # Scaled optimizer step
            scaler.update()  # Update the scaler for next iteration
            # loss.backward()  # Backward pass
            # optimizer.step()

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
        val_accuracy, val_loss, auc, eer = evaluate_model(model, criterion, val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {100 * correct / total:.2f}% - "
              f"Val Accuracy: {val_accuracy:.2f}% - Val Loss: {val_loss:.4f} - AUC: {auc:.4f} - EER: {eer:.4f}")

        # Save the best model based on AUC
        if (epoch+1) % save_every == 0 or auc > best_auc and auc < 1:   # exceptional case, auc =1
            best_auc = auc
            best_eer = eer
            save_path_epoch = f"{save_path}_epoch_{epoch + 1}.pth"  # Save with epoch info

            torch.save(model.state_dict(), save_path_epoch)
            print(f"New best model saved with AUC: {best_auc:.4f}")

        # # Save the best model based on EER (lower is better)
        # if eer < best_eer:
        #     best_eer = eer
        #     torch.save(model.state_dict(), save_path)
        #     print(f"New best model saved with EER: {best_eer:.4f}")

    print(f"Training complete. Best AUC: {best_auc:.4f}, Best EER: {best_eer:.4f}")
    return best_model_path

def get_model(model_name, device):

    if model_name.lower() == 'aasist':
        # Instantiate the aasist model
        with open("./models/aasist/AASIST.conf", "r") as f_json:
            assist_config = json.loads(f_json.read())
        model_config = assist_config["model_config"]
        model = Model_ASSIST(model_config).to(device)
    elif model_name.lower() == 'rawnet3':
        # Instantiate the rawnet3 model
        model = RawNetWithFC(embedding_dim=256, num_classes=2).to(device)
    elif model_name.lower() == 'rawnet2':
        # Instantiate the rawnet2 model
        with open("./models/rawnet/RawNet2_config.yaml", 'r') as f_yaml:
            parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
        model = RawNet2(parser1['model'], device).to(device)
    elif model_name.lower() == 'rawboost':
        # Instantiate the rawboost model
        with open("./models/rawboost/model_config_RawNet.yaml", 'r') as f_yaml:
            parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
        model = RawNet(parser1['model'], device).to(device)
    elif model_name.lower() == 'ssdnet':
        # SSDNet Model
        model = SSDNet1D_L().to(device)    # SSDNet1D_S, DilatedNet_S, SSDNet1D_L, DilatedNet_L
    elif model_name.lower() == 'inc_ssdnet':
        # SSDNet Model
        model = DilatedNet_L().to(device)    # SSDNet1D_S, DilatedNet_S, SSDNet1D_L, DilatedNet_L
    elif model_name.lower() == 'resnet1d':
        # ResNet1D
        model = ResNet1D(in_channels = 1 , base_filters = 128, kernel_size = 5, stride=2, groups = 1, n_block = 3, n_classes = 2).to(device)
    elif model_name.lower() == 'msresnet':
        # MSResNet
        model = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=2).to(device)
    elif model_name.lower() == 'acnn1d':
        # ACNN1D
        model = ACNN(in_channels = 1 , out_channels = 1024, att_channels=256, n_len_seg = 256, n_classes = 2, device=device, verbose = False).to(device)

    return model

    # # CNN1D
    # model = CNN(in_channels = 1 , out_channels = 1024, n_len_seg = 256, n_classes = 2, device=device, verbose = False).to(device)

    # # RCNN1D
    # model = CRNN(in_channels = 1 , out_channels = 1024, n_len_seg = 256, n_classes = 2, device=device, verbose = False).to(device)

if __name__ == '__main__':

    # Main script
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-rt', type=str, default='../DATASETS/DTIM', help='')
    parser.add_argument('--nEpochs', '-epoch', type=int, default=50, help='')
    parser.add_argument('--batch_size', '-b', type=int, default=256, help='')
    parser.add_argument('--num_workers', '-w', type=int, default=4, help='')
    parser.add_argument('--lr', '-lr', type=float, default=0.001, help='')
    # parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0], help='')
    args = parser.parse_args()
    print(args)

    args.ratio = 8
    # Example usage
    train_dataset = RawNetDATAReader(args=args, split='TRAIN')
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    dev_dataset = RawNetDATAReader(args=args, split='TEST')
    dev_loader = data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    model_name = "inc_ssdnet" # models_list = "aasist", "rawnet3", "rawnet2", "rawboost", "ssdnet", "inc_ssdnet", "resnet1d", "msresnet", "acnn1d"
    save_every=10
    device_id = 3
    device = torch.device('cuda',device_id)
    model = get_model(model_name,device)

    os.makedirs(f"./weights/{model_name}",exist_ok=True)
    model_save_path = f"./weights/{model_name}/best_{model_name}_ratio_{args.ratio}"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # Only optimize the FC layer

    best_model_path = train_model(model, criterion, optimizer , train_loader, dev_loader, num_epochs=args.nEpochs, save_path=model_save_path,save_every=save_every)
    print(f"Best model saved at: {best_model_path}")
