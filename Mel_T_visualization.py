import os
import torch
from torch import nn
from torch.utils import data
# import timm
import argparse

import numpy as np
from sklearn.metrics import *
from tqdm import tqdm
# from natsort import natsort_keygen
from compose_models import get_aasist, get_inc_ssdnet, get_rawboost, get_rawnet2, get_rawnet_3, get_ssdnet
from data_loader import DATAReader
from test_generator import GeneratorSimple
# from opgan import Generator
from unet import Generator
from utils import transcribe_audio
from visualize import compare_audio_samples

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
        model = get_rawnet2(device)
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


def test(model_name, batch_size, num_workers, device):

    test_dataset = DATAReader( split='TEST') # TEST, In_The_Wild, WaveFake
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

    # print('Device being used:', device)

    # G = nn.DataParallel(Generator())
    # G = GeneratorSimple()
    G = Generator()
    # G = nn.DataParallel(Generator()).to(device)

    # real_acc, fake_acc, af_acc = [], [], []
    progress_bar = tqdm(test_loader, desc="Testing", unit="batch", leave=True)
    # model = get_model(model_name, device)

    for sample_index,test_sample in enumerate(progress_bar):

        generator_ids = [1]#,2,3,4,5,6]
        for generator_id in generator_ids:
            checkpoint = torch.load(f"/data/Umar/Repos/Collaborative_Learning_Interspeech/Attacks/unet.pth", map_location=device, weights_only=False)
            state_dict = checkpoint['state_dict']
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            G.load_state_dict(new_state_dict)
            G = G.to(device)  # Move model to the appropriate device
            G.eval()

            real = test_sample[0].unsqueeze(1).to(device, dtype=torch.float)
            forged = test_sample[2].unsqueeze(1).to(device, dtype=torch.float)

            # y_real =  torch.zeros(real.shape[0]).to(device, dtype=torch.float)
            # y_fake =  torch.ones(forged.shape[0]).to(device, dtype=torch.float)

            fake = G(forged)

            # forged_transciption1 = transcribe_audio(forged,self.t_processor_1, self.t_model_1, self.device)
            # attacked_transciption1 = transcribe_audio(fake,self.t_processor_1, self.t_model_1, self.device)

            for index in range(10):
                forged_audio = forged[index].detach()  # Select first sample of forged audio
                attacked_audio = fake[index].detach()   # Corresponding generated audio

                # self.s1_w, self.s2_w, self.t1_w
                # Plot and compare
                wav_dir_path = 'Wav_Plot_'+'Mel'#self.save_dir_path
                os.makedirs(wav_dir_path, exist_ok=True)
                compare_audio_samples(forged_audio, attacked_audio, "","",sample_index,generator_id,index,wav_dir_path, sr=16000)



    #     real_acc.append(cal_acc(model, model_name, y_real, real, device))
    #     fake_acc.append(cal_acc(model, model_name, y_fake, forged, device))
    #     af_acc.append(cal_acc(model, model_name, y_fake, fake, device))

    # progress_bar.close()  # Ensure tqdm closes cleanly when done
    # return 100*np.mean(real_acc), 100*np.mean(fake_acc), 100*np.mean(af_acc)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='')
    parser.add_argument('--num_workers', '-w', type=int, default=16, help='')

    args = parser.parse_args()
    print(args)
    device_id = 1
    device = torch.device('cuda', device_id)

    model_name = 'inc_ssdnet_large'  # RawNet3, AASIST, ssdnet_original, ssdnet_small, ssdnet_large, inc_ssdnet_original, inc_ssdnet_small, inc_ssdnet_large, rawnet3, rawnet2, rawboost
    print(f"Model: {model_name}")
    test(model_name, args.batch_size, args.num_workers, device)
    # r_acc, f_acc, af_acc = test(model_name, args.batch_size, args.num_workers, device)
    # print('[Test] [[Acc: %.2f, %.2f, %.2f]'% (r_acc, f_acc, af_acc))

    # Example usage
    # binary_classifier = get_rawnet3_binary_classifier()

    # # Input tensor: batch of 16 samples
    # x = torch.randn(1, 1, 64600).to(device)

    # # Get binary class logits
    # outputs = binary_classifier(x.squeeze(1))
    # print(f"Shape of binary classifier output: {outputs.shape}")
