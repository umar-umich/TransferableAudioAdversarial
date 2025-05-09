from DF_dataloader import DF_DATAReader
import torch
from torch import nn
from torch.utils import data
# import timm
import argparse

import numpy as np
from sklearn.metrics import *
from tqdm import tqdm
# from natsort import natsort_keygen
from compose_models import get_aasist, get_acnn, get_cnn, get_crnn, get_inc_ssdnet, get_msresnet, get_psanet, get_rawboost, get_rawnet_2, get_rawnet_3, get_resnet, get_ssdnet
from data_loader import DATAReader
from test_generator import GeneratorSimple
from wavefake_data_loader import WaveFakeDATAReader

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cal_acc(model,model_name, y, x,index, device):  # It should be model instead of model_name
    # outputs = inception(x)
    outputs = {}
    if model_name.lower() == 'aasist':
        model = model.to(device)
        outputs = model(x.squeeze(1))
        # print(f'Shape of AASIST output: {str(outputs[0].shape)}: {str(outputs[1].shape)}')
        predictions = outputs[1]
    elif model_name.lower() in ('rawnet3', 'rawboost', 'rawnet2', 'resnet', 'msresnet', 'acnn','cnn','crnn'):
        model = model.to(device)
        outputs = model(x.squeeze(1))
        # print(f'Shape of rawnet output: {str(outputs.shape)}')
        predictions = outputs    
    # exact name match....
    elif model_name.lower() == 'psanet':
        x = x.squeeze(1)
        x_numpy = x.cpu().detach().numpy().astype(np.float32)
        prediction = model.predict(x_numpy)
        print(prediction)

        # Decide if real or deepfake based on the prediction
        # Assuming 0 indicates real and 1 indicates deepfake
        if prediction <= 0.1:  # You might need to adjust this threshold based on your model's output
            prediction = [1]  # Real
        else:
            prediction = [0]  # Deepfake
        acc = accuracy_score(y.cpu().numpy(), prediction)

        return acc, prediction[0]


    else: #if model_name.lower() == 'ssdnet' or model_name.lower() == 'inc_ssdnet':        
        outputs = model(x)
        # print(f'Shape of ssdnet output: {str(outputs.shape)}')
        predictions = outputs   
    

    predictions = nn.Softmax(dim=-1)(predictions)
    _, y_ = torch.max(predictions, 1)
    print(f"Chunk# {index+1}, Predicted label: {'Fake' if 1 == y_.cpu().numpy()[0] else 'Real'}")

    acc = accuracy_score(y.cpu().numpy(), y_.cpu().numpy())

    return acc,  y_.cpu().numpy()[0] 

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
    elif model_name.lower() == 'resnet':
        model = get_resnet(device)
    elif model_name.lower() == 'msresnet':
        model = get_msresnet(device)
    elif model_name.lower() == 'acnn':
        model = get_acnn(device)
    elif model_name.lower() == 'cnn':
        model = get_cnn(device)
    elif model_name.lower() == 'crnn':
        model = get_crnn(device)
    elif model_name.lower() == 'psanet':
        model = get_psanet()        

    return model

def test(model_name, device):
      # Example path to audio file
    test_sample_name= 'Ulendo_test_sample.wav'
    dataset_name = test_sample_name
    print(f"Dataset Name: {dataset_name}")

    # test_dataset = DATAReader( split=dataset_name) # TEST, In_The_Wild, WaveFake
    # # test_dataset = WaveFakeDATAReader(split=dataset_name) # TEST, In_The_Wild, WaveFake

    test_audio_path = './Ulendo_test_sample.wav'
    data_reader = DF_DATAReader(test_audio_path)    
    data_loader = data.DataLoader(data_reader, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True, persistent_workers=True)
    
    # progress_bar = tqdm(test_loader, desc="Testing", unit="batch", leave=True)
    model = get_model(model_name, device)

    real_acc = []
    real_count = 0
    fake_count = 0

    with torch.no_grad():
        for index, audio_chunk in enumerate(data_loader):
            with torch.autocast(device_type='cuda', dtype=torch.float16):  # Enable Mixed Precision
                audio_sample = audio_chunk.unsqueeze(1).to(device, dtype=torch.float16)

                # y_real =  torch.zeros(audio_sample.shape[0]).to(device, dtype=torch.float16)
                y_fake =  torch.ones(audio_sample.shape[0]).to(device, dtype=torch.float16)

                acc, label = cal_acc(model, model_name, y_fake, audio_sample,index, device)
                if label == 1:
                    fake_count += 1
                else:
                    real_count  +=1

                real_acc.append(acc)

    # progress_bar.close()  # Ensure tqdm closes cleanly when done
    return 100*np.mean(real_acc), fake_count, real_count

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size', '-b', type=int, default=128, help='')
    # parser.add_argument('--num_workers', '-w', type=int, default=16, help='')

    # args = parser.parse_args()
    # print(args)
    device_id = 1
    device = torch.device('cuda', device_id)

    model_name = 'psanet'  # RawNet3, aasist, ssdnet_original, ssdnet_small, ssdnet_large, inc_ssdnet_original, inc_ssdnet_small, inc_ssdnet_large, psanet
    # rawnet3, rawnet2, rawboost,resnet, msresnet, acnn, crnn, cnn
    # print(f"Model: {model_name}")
    r_acc, fake_count, real_count = test(model_name, device)
    print('[Test] [[Confidence: %.2f, # of Fake chunks: %d, # of Real chunks:  %d]'% (r_acc, fake_count, real_count))