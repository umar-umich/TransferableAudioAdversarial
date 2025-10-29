import argparse
import json

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils import data
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
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

from train_data_loader import RawNetDATAReader

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


    total_real = 0
    correct_real = 0
    total_fake = 0
    correct_fake = 0

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

            # Calculate separate accuracy for real and fake samples
            real_mask = labels == 0
            fake_mask = labels == 1

            total_real += real_mask.sum().item()
            correct_real += (predicted[real_mask] == 0).sum().item()

            total_fake += fake_mask.sum().item()
            correct_fake += (predicted[fake_mask] == 1).sum().item()

            # Collect scores and labels for metrics
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

            # Update progress bar with current metrics
            progress_bar.set_postfix({
                "Val Loss": f"{val_loss / total:.4f}",
                "Accuracy": f"{100 * correct / total:.2f}%"
            })

    accuracy = 100 * correct / total
    # avg_loss = val_loss / total
    accuracy_real = 100 * correct_real / total_real if total_real > 0 else 0.0
    accuracy_fake = 100 * correct_fake / total_fake if total_fake > 0 else 0.0

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

    return accuracy,accuracy_real,accuracy_fake, auc, eer

def get_model(model_name, device):
    if model_name.lower() == 'aasist':
        # Instantiate the aasist model
        with open("./models/aasist/AASIST.conf", "r") as f_json:
            assist_config = json.loads(f_json.read())
        model_config = assist_config["model_config"]
        model = Model_ASSIST(model_config)
    elif model_name.lower() == 'rawnet3':
        # Instantiate the rawnet3 model
        model = RawNetWithFC(embedding_dim=256, num_classes=2)
        check_point = torch.load("./weights/rawnet_3/best_rawnet3_8.pth", map_location=device, weights_only=True)
        model.load_state_dict(check_point)
    elif model_name.lower() == 'rawnet2':
        # Instantiate the rawnet2 model
        with open("./models/rawnet/RawNet2_config.yaml", 'r') as f_yaml:
            parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
        model = RawNet2(parser1['model'], device)
        check_point = torch.load("./weights/rawnet_2/best_rawnet2_4.pth", map_location=device, weights_only=True)
        model.load_state_dict(check_point)
    elif model_name.lower() == 'rawboost':
        # Instantiate the rawboost model
        with open("./models/rawboost/model_config_RawNet.yaml", 'r') as f_yaml:
            parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
        model = RawNet(parser1['model'], device)
        check_point = torch.load("./weights/rawboost/best_rawboost_4.pth", map_location=device, weights_only=True)
        model.load_state_dict(check_point)
    elif model_name.lower() == 'ssdnet':
        # SSDNet Model
        model = SSDNet1D_L()    # SSDNet1D_S, DilatedNet_S, SSDNet1D_L, DilatedNet_L
        check_point = torch.load("./weights/ssdnet/best_ssdnet_ratio_4_epoch_40.pth", map_location=device, weights_only=True)
        model.load_state_dict(check_point)
    elif model_name.lower() == 'ssdnet_small':
        model = SSDNet1D_S()
        check_point = torch.load("./weights/ssdnet/best_ssdnet_S.pth", map_location=device, weights_only=True)
        model.load_state_dict(check_point)
    elif model_name.lower() == 'ssdnet_large':
        model = SSDNet1D_L()
        check_point = torch.load("./weights/ssdnet/best_ssdnet_L.pth", map_location=device, weights_only=True)
        model.load_state_dict(check_point)
    elif model_name.lower() == 'inc_ssdnet':
        # SSDNet Model
        model = DilatedNet_L()    # SSDNet1D_S, DilatedNet_S, SSDNet1D_L, DilatedNet_L
        check_point = torch.load("./weights/inc_ssdnet/best_inc_ssdnet_ratio_4_epoch_40.pth", map_location=device, weights_only=True)
        model.load_state_dict(check_point)
    elif model_name.lower() == 'resnet1d':
        # ResNet1D
        model = ResNet1D(in_channels = 1 , base_filters = 128, kernel_size = 5, stride=2, groups = 1, n_block = 3, n_classes = 2)
        check_point = torch.load("./weights/resnet/best_resnet_8.pth", map_location=device, weights_only=True)
        model.load_state_dict(check_point)
    elif model_name.lower() == 'msresnet':
        # MSResNet
        model = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=2)
        check_point = torch.load("./weights/msresnet/best_msresnet_ratio_4_epoch_10.pth", map_location=device, weights_only=True)
        model.load_state_dict(check_point)
    elif model_name.lower() == 'acnn1d':
        # ACNN1D
        model = ACNN(in_channels = 1 , out_channels = 1024, att_channels=256, n_len_seg = 256, n_classes = 2, device=device, verbose = False)

    model.eval()
    return model.to(device)


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
    test_dataset = RawNetDATAReader(args=args, split='TEST')
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    model_name = "msresnet" # models_list = "aasist", "rawnet3", "rawnet2", "rawboost", "ssdnet", "inc_ssdnet", "resnet1d", "msresnet", "acnn1d"
    save_every=10
    device_id = 3
    device = torch.device('cuda',device_id)
    model = get_model(model_name,device)
    criterion = nn.CrossEntropyLoss()

    test_accuracy,accuracy_real,accuracy_fake, auc, eer = evaluate_model(model, criterion, test_loader)
    print(f"Test Accuracy: {test_accuracy:.2f}% - Real Accuracy: {accuracy_real:.2f}% -Fake Accuracy: {accuracy_fake:.2f}% - AUC: {auc:.2f} - EER: {eer:.4f}")
