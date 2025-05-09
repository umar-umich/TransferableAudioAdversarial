import datetime
import torch
from torch import nn
from torch.utils import data
# import timm
import argparse
import os
import numpy as np
from sklearn.metrics import *
from tqdm import tqdm
# from natsort import natsort_keygen
from compose_models import get_aasist, get_inc_ssdnet, get_rawboost, get_rawnet_2, get_rawnet_3, get_sentence_transformer, get_ssdnet, get_wav2vec2_model
from data_loader import DATAReader
from test_generator import GeneratorSimple
from utils import get_one_transciption_loss, get_one_transciption_sim, get_transciption_loss, transcribe_audio
from visualize import calculate_psnr, calculate_ssim
from wavefake_data_loader import WaveFakeDATAReader

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cal_acc(model,model_name, y, x, device):  # It should be model instead of model_name
    # outputs = inception(x)
    outputs = {}
    if model_name.lower() == 'aasist':
        outputs = model(x.squeeze(1))
        # print(f'Shape of AASIST output: {str(outputs[0].shape)}: {str(outputs[1].shape)}')
        predictions = outputs[1]
    elif model_name.lower() == 'rawnet3' or model_name.lower() == 'rawboost' or model_name.lower() == 'rawnet2':
        outputs = model(x.squeeze(1))
        # print(f'Shape of rawnet output: {str(outputs.shape)}')
        predictions = outputs    
    # exact name match....
    else: #if model_name.lower() == 'ssdnet' or model_name.lower() == 'inc_ssdnet':        
        outputs = model(x)
        # print(f'Shape of ssdnet output: {str(outputs.shape)}')
        predictions = outputs   

    predictions = nn.Softmax(dim=-1)(predictions)
    _, y_ = torch.max(predictions, 1)

    acc = accuracy_score(y.cpu().numpy(), y_.cpu().numpy())

    return acc

def get_model(model_name,device):

    if model_name.lower() == 'aasist':
        model = get_aasist(device)
    elif model_name.lower() == 'rawnet3':
        model = get_rawnet_3(device)    
    elif model_name.lower() == 'rawnet2':
        model = get_rawnet_2(device)
    elif model_name.lower() == 'ssdnet_original':
        model = get_ssdnet('original', device)
    elif model_name.lower() == 'ssdnet_small':
        model = get_ssdnet('small', device)
    elif model_name.lower() == 'ssdnet_large':
        model = get_ssdnet('large', device)
    elif model_name.lower() == 'inc_ssdnet_original':
        model = get_inc_ssdnet('original', device)
    elif model_name.lower() == 'inc_ssdnet_small':
        model = get_inc_ssdnet('small', device)
    elif model_name.lower() == 'inc_ssdnet_large':
        model = get_inc_ssdnet('large', device)
    elif model_name.lower() == 'rawboost':
        model = get_rawboost(device)

    return model.to(device)


def find_text_sim(forged, fake, t_processor, t_model,sentence_transformer, device):
    forged_transciption = transcribe_audio(forged,t_processor, t_model, device)
    attacked_transciption = transcribe_audio(fake,t_processor, t_model, device)

    text_sim = get_one_transciption_sim(forged_transciption, attacked_transciption, sentence_transformer)

    return text_sim


def find_text_mismatch(forged_batch, fake_batch, t_processor, t_model,sentence_transformer, G_2, save_dir_path, device):
    # generate transcriptions
    # If the transcription is not good for a single sample i.e., loss is high, save them to text file.
    # Also get the transcription for the same recorded sample from the model with T  ...  Just for automation. No more donkey work.
    # for each sample in batch
    for index, forged in enumerate(forged_batch):
        fake = fake_batch[index]
        forged_transciption = transcribe_audio(forged,t_processor, t_model, device)
        attacked_transciption = transcribe_audio(fake,t_processor, t_model, device)

        t1_loss = get_one_transciption_loss(forged_transciption, attacked_transciption, sentence_transformer)
        t_loss_threshold = 0.1


        if t1_loss > t_loss_threshold:
            print(f"T1_loss:{t1_loss}")
    
            fake2 = G_2(forged)
            attacked_transciption_2 = transcribe_audio(fake2,t_processor, t_model, device)
            # Write results to a text file
            os.makedirs(save_dir_path, exist_ok=True)  # Ensure the directory exists
            forged_file_path = os.path.join(save_dir_path, f"metrics_t_loss_{t_loss_threshold}_forged.txt")
            attacked_file_path = os.path.join(save_dir_path, f"metrics_t_loss_{t_loss_threshold}_attacked.txt")
            attacked_T_file_path = os.path.join(save_dir_path, f"metrics_t_loss_{t_loss_threshold}_attacked_T.txt")
            with open(forged_file_path, "a") as file:
                file.write("\n")
                file.write(f"{'Transcription'}: \t {forged_transciption}\n ")
            with open(attacked_file_path, "a") as file:
                file.write("\n")
                file.write(f"{'Transcription'}: \t {attacked_transciption}\n ")
            with open(attacked_T_file_path, "a") as file:
                file.write("\n")
                file.write(f"{'Transcription'}: \t {attacked_transciption_2}\n ")            

            print(f"Metrics saved to Text files")

        



def test(model_name, batch_size, num_workers, device):

    # test_dataset = DATAReader( split='TEST') # TEST, In_The_Wild, WaveFake
    test_dataset = WaveFakeDATAReader(split='WaveFake') # TEST, In_The_Wild, WaveFake
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

    # print('Device being used:', device)
    time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_dir_path = f"{time_now}_{model_name}_Test_"

    # G = nn.DataParallel(Generator()) 
    G = GeneratorSimple()
    checkpoint = torch.load("./CHECKPOINTS_2024-12-02-21-59-44_ssdnet_inc_ssdnet_0.0001_0.0001_0.01/generator_27.pth", map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict']
    # Remove 'module.' prefix from keys
    # new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    G.load_state_dict(state_dict)
    G = G.to(device)  # Move model to the appropriate device
    G.eval()

    # G = nn.DataParallel(Generator())  CHECKPOINTS_2024-12-06-13-35-33_ssdnet_inc_ssdnet_with_T_0.0001_0.0001_0.0001/generator_22.pth
    G_2 = GeneratorSimple()
    checkpoint_2 = torch.load("./CHECKPOINTS_2024-12-02-21-59-44_ssdnet_inc_ssdnet_0.0001_0.0001_0.01/generator_27.pth", map_location=device, weights_only=False)
    state_dict_2 = checkpoint_2['state_dict']
    # Remove 'module.' prefix from keys
    # new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    G_2.load_state_dict(state_dict_2)
    G_2 = G_2.to(device)  # Move model to the appropriate device
    G_2.eval()

    real_acc, fake_acc, af_acc = [], [], []
    progress_bar = tqdm(test_loader, desc="Testing", unit="batch", leave=True)
    model = get_model(model_name, device)

    t_processor, t_model = get_wav2vec2_model(device)
    sentence_transformer = get_sentence_transformer(device)
    t_model = t_model.to(device)

    best_psnr = 0
    best_ssim = 0
    best_text_sim = 0


    for test_sample in progress_bar:
        real = test_sample[0].unsqueeze(1).to(device, dtype=torch.float)
        forged = test_sample[2].unsqueeze(1).to(device, dtype=torch.float)

        y_real =  torch.zeros(real.shape[0]).to(device, dtype=torch.float)
        y_fake =  torch.ones(forged.shape[0]).to(device, dtype=torch.float)

        fake = G(forged)

        forged_sample = forged[0].detach().squeeze().cpu().numpy()
        fake_sample = fake[0].detach().squeeze().cpu().numpy()
        # print(f"Shapes: {forged_sample.shape}: {fake_sample.shape}")
        # find_text_mismatch(forged, fake, t_processor, t_model,sentence_transformer, G_2, save_dir_path, device)
        text_sim = find_text_sim(forged, fake, t_processor, t_model,sentence_transformer, device)
        psnr = calculate_psnr(forged_sample, fake_sample)
        ssim = calculate_ssim(forged_sample, fake_sample)
        # text_sim = np.mean(text_sim)
        if text_sim > best_text_sim:            
            best_text_sim = text_sim
        if psnr > best_psnr:            
            best_psnr = psnr
        if ssim > best_ssim:            
            best_ssim = ssim
        
        real_acc.append(cal_acc(model, model_name, y_real, real, device))
        fake_acc.append(cal_acc(model, model_name, y_fake, forged, device))
        af_acc.append(cal_acc(model, model_name, y_fake, fake, device))

    progress_bar.close()  # Ensure tqdm closes cleanly when done
    return 100*np.mean(real_acc), 100*np.mean(fake_acc), 100*np.mean(af_acc), best_psnr, best_ssim, best_text_sim

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='')
    parser.add_argument('--num_workers', '-w', type=int, default=16, help='')

    args = parser.parse_args()
    print(args)
    device_id = 1
    device = torch.device('cuda', device_id)

    model_name = 'ssdnet_original'  # aasist, ssdnet_original, ssdnet_small, ssdnet_large, inc_ssdnet_original, inc_ssdnet_small, inc_ssdnet_large, rawnet2, rawboost
    print(f"Model: {model_name}")
    r_acc, f_acc, af_acc, best_psnr, best_ssim, best_text_sim = test(model_name, args.batch_size, args.num_workers, device)
    print('[Test] [[Acc: %.2f, %.2f, %.2f]] Quantitative [ %.2f, %.2f, %.2f]'% (r_acc, f_acc, af_acc, best_psnr, best_ssim, best_text_sim))

    # Example usage
    # binary_classifier = get_rawnet3_binary_classifier()

    # # Input tensor: batch of 16 samples
    # x = torch.randn(1, 1, 64600).to(device)

    # # Get binary class logits
    # outputs = binary_classifier(x.squeeze(1))
    # print(f"Shape of binary classifier output: {outputs.shape}")
