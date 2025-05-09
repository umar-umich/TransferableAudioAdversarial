

import json
from models.cnns.acnn1d import ACNN
from models.cnns.cnn1d import CNN
from models.cnns.crnn1d import CRNN
from models.resnet1d.multi_scale_ori import MSResNet
from models.resnet1d.resnet1d import ResNet1D
from models.rsm1d.RSM1D_large import DilatedNet_L, SSDNet1D_L
from models.rsm1d.RSM1D_small import DilatedNet_S, SSDNet1D_S
import torch
from torch import nn
import yaml
from models.aasist.AASIST import Model_ASSIST
from models.rawboost.RawBoost import RawNet
from models.rawnet.RawNet2 import RawNet2
from models.rawnet.RawNet3 import RawNet3
from models.rawnet.RawNetBasicBlock import Bottle2neck
from models.rsm1d.RSM1D import DilatedNet, SSDNet1D
import tensorflow as tf

# Function to make prediction using the loaded model and preprocessed audio
def get_psanet():
    print("Loading TF model")
    # Load the model
    model_path = './weights/psanet/mybestmodel.hdf5'
    model = tf.keras.models.load_model(model_path)
    print("TF model loaded")
    return model


def get_rawnet3():
    rawnet_model = RawNet3(
        Bottle2neck,
        model_scale=8,
        context=True,
        summed=True,
        encoder_type="ECA",
        nOut=256,
        out_bn=False,
        sinc_stride=10,
        log_sinc=True,
        norm_sinc="mean",
        grad_mult=1,
    )
    # rawnet_model.load_state_dict(
    #     torch.load(
    #         "./weights/rawnet_3/model.pt",
    #         map_location=lambda storage, loc: storage, weights_only=True
    #     )["model"]
    # )
    # rawnet_model = rawnet_model.to(device)  # Move model to the appropriate device

    # rawnet_model.eval()
    # print("RawNet3 initialised & weights loaded!")

    return rawnet_model

# Define the modified RawNet with a trainable FC layer
class RawNetWithFC(nn.Module):
    def __init__(self, embedding_dim=256, num_classes=2):
        super(RawNetWithFC, self).__init__()
        self.rawnet = get_rawnet3()
        # for param in self.rawnet.parameters():  # Freeze all RawNet layers
        #     param.requires_grad = False
        self.fc = nn.Linear(embedding_dim, num_classes)  # Trainable FC layer

    def forward(self, x):
        x = self.rawnet(x)  # Get embeddings
        x = self.fc(x)  # Pass through FC layer
        return x
    


def get_aasist(device):
    # Load the AASIST model
    with open("./models/aasist/AASIST.conf", "r") as f_json:
        assist_config = json.loads(f_json.read())
    model_config = assist_config["model_config"]
    # model_config = config["model_config"]

    # print(f'ASSIST Conf: {str(model_config)}')
    assist_model = Model_ASSIST(model_config)
    assist_model.load_state_dict(torch.load("./weights/AASIST.pth", map_location=device, weights_only=True))
    # assist_model = assist_model.to(device)  # Move model to the appropriate device

    num_total_learnable_params = sum(i.numel() for i in assist_model.parameters() if i.requires_grad)
    print('Number of learnable params: {}.'.format(num_total_learnable_params))

    assist_model.eval()  # Set the model to evaluation mode
    return assist_model


def get_ssdnet(model_type,device):
    if model_type == 'original':
        ssdnet_model = SSDNet1D()
        check_point = torch.load("./weights/ssdnet/best_ssdnet_Combined_epoch_25.pth", map_location=device, weights_only=True)
        ssdnet_model.load_state_dict(check_point)
    elif model_type == 'small':
        ssdnet_model = SSDNet1D_S()
        check_point = torch.load("./weights/ssdnet/best_ssdnet_S.pth", map_location=device, weights_only=True)
        ssdnet_model.load_state_dict(check_point)
    elif model_type == 'large':
        ssdnet_model = SSDNet1D_L()
        check_point = torch.load("./weights/ssdnet/best_ssdnet_L.pth", map_location=device, weights_only=True)
        ssdnet_model.load_state_dict(check_point)

    num_total_learnable_params = sum(i.numel() for i in ssdnet_model.parameters() if i.requires_grad)
    print(f'Number of learnable params for {model_type}: {num_total_learnable_params}.')

    # ssdnet_model = ssdnet_model.to(device)  # Move model to the appropriate device
    ssdnet_model.eval()
    return ssdnet_model

def get_inc_ssdnet(device):
    inc_ssdnet = DilatedNet()
    num_total_learnable_params = sum(i.numel() for i in inc_ssdnet.parameters() if i.requires_grad)
    print('Number of learnable params: {}.'.format(num_total_learnable_params))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    check_point = torch.load("./weights/ssdnet/ssdnet_1.09.pth", map_location=device, weights_only=True)
    inc_ssdnet.load_state_dict(check_point['model_state_dict'])
    inc_ssdnet = inc_ssdnet.to(device)  # Move model to the appropriate device
    inc_ssdnet.eval()
    return inc_ssdnet


def get_inc_ssdnet(model_type,device):
    if model_type == 'original':
        inc_ssdnet = DilatedNet()
        check_point = torch.load("./weights/ssdnet/ssdnet_1.09.pth", map_location=device, weights_only=True)
        inc_ssdnet.load_state_dict(check_point['model_state_dict'])
    elif model_type == 'small':
        inc_ssdnet = DilatedNet_S()
        check_point = torch.load("./weights/ssdnet/best_inc_ssdnet_S.pth", map_location=device, weights_only=True)
        inc_ssdnet.load_state_dict(check_point)
    elif model_type == 'large':
        inc_ssdnet = DilatedNet_L()
        check_point = torch.load("./weights/ssdnet/best_inc_ssdnet_L.pth", map_location=device, weights_only=True)
        inc_ssdnet.load_state_dict(check_point)
    
    num_total_learnable_params = sum(i.numel() for i in inc_ssdnet.parameters() if i.requires_grad)
    print('Number of learnable params: {}.'.format(num_total_learnable_params))

    # inc_ssdnet = inc_ssdnet.to(device)  # Move model to the appropriate device
    inc_ssdnet.eval()
    return inc_ssdnet


def get_rawboost(device):
    with open("./models/rawboost/model_config_RawNet.yaml", 'r') as f_yaml:
        parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
    rawboost_model = RawNet(parser1['model'], device)
    rawboost_model.load_state_dict(torch.load("./weights/rawboost/best_rawboost_8.pth", map_location=device, weights_only=True))
    # rawboost_model = rawboost_model.to(device)  # Move model to the appropriate device
    rawboost_model.eval()
    return rawboost_model

def get_resnet(device):
    resnet_model = ResNet1D(in_channels = 1 , base_filters = 128, kernel_size = 5, stride=2, groups = 1, n_block = 3, n_classes = 2)
    check_point = torch.load("./weights/resnet/best_resnet_8.pth", map_location=device, weights_only=True)
    resnet_model.load_state_dict(check_point)
    resnet_model.eval()
    return resnet_model

def get_msresnet(device):
    ms_resnet_model = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=2)   # ./weights/resnet/best_msresnet_ratio_4_epoch_10.pth
    check_point = torch.load("./weights/msresnet/best_msresnet_ratio_8_epoch_25.pth", map_location=device, weights_only=True)
    ms_resnet_model.load_state_dict(check_point)
    ms_resnet_model.eval()
    return ms_resnet_model


def get_acnn(device):
    acnn_model = ACNN(in_channels = 1 , out_channels = 1024, att_channels=256, n_len_seg = 256, n_classes = 2, device=device, verbose = False).to(device)   # ./weights/resnet/best_msresnet_ratio_4_epoch_10.pth
    check_point = torch.load("./weights/cnns/best_acnn_ratio_4_epoch_5.pth", map_location=device, weights_only=True)
    acnn_model.load_state_dict(check_point)
    acnn_model.eval()
    return acnn_model

def get_cnn(device):
    cnn_model = CNN(in_channels = 1 , out_channels = 256, n_len_seg = 128, n_classes = 2, device=device, verbose = False).to(device)   # ./weights/resnet/best_msresnet_ratio_4_epoch_10.pth
    check_point = torch.load("./weights/cnns/best_cnn_ratio_2_epoch_5.pth", map_location=device, weights_only=True)
    cnn_model.load_state_dict(check_point)
    cnn_model.eval()
    return cnn_model

def get_crnn(device):
    crnn_model = CRNN(in_channels = 1 , out_channels = 1024, n_len_seg = 256, n_classes = 2, device=device, verbose = False).to(device)   # ./weights/resnet/best_msresnet_ratio_4_epoch_10.pth
    check_point = torch.load("./weights/cnns/best_crnn_ratio_4_epoch_5.pth", map_location=device, weights_only=True)
    crnn_model.load_state_dict(check_point)
    crnn_model.eval()
    return crnn_model

def get_rawnet_3(device):
    # Instantiate the model
    rawnet_model = RawNetWithFC(embedding_dim=256, num_classes=2)#.to(device)
    rawnet_model.load_state_dict(torch.load("./weights/rawnet_3/best_rawnet3_6.pth", map_location=device, weights_only=True))
    # rawnet_model = rawnet_model.to(device)  # Move model to the appropriate device
    rawnet_model.eval()
    return rawnet_model

# def get_rawnet_2(device):
#     with open("./models/rawnet/RawNet2_config.yaml", 'r') as f_yaml:
#         parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
#     rawnet_model = RawNet2(parser1['model'], device)
#     rawnet_model.load_state_dict(torch.load("./weights/rawnet_2/best_rawnet2_6.pth", map_location=device, weights_only=True))
#     # rawnet_model = rawnet_model.to(device)  # Move model to the appropriate device
#     rawnet_model.eval()
#     return rawnet_model

def get_rawnet2(device):
    with open("./models/rawnet/RawNet2_config.yaml", 'r') as f_yaml:
        parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
    rawnet_model = RawNet2(parser1['model'], device)
    rawnet_model.load_state_dict(torch.load("./weights/rawnet_2/best_rawnet2_6.pth", map_location=device, weights_only=True))
    rawnet_model = rawnet_model.to(device)  # Move model to the appropriate device
    rawnet_model.eval()
    return rawnet_model


def get_wav2vec2_model(device):
    # load model and processor
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")  # facebook/wav2vec2-base-960h
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    # processor = processor.to(device)
    # model = model.to(device)
    model.eval()  # Set to evaluation mode
    return processor, model



def get_speech_to_text_model(device):
    # load model and processor
    from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
    model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")

    # processor = processor.to(device)
    # model = model.to(device)
    model.eval()  # Set to evaluation mode
    return processor, model

def get_sentence_transformer(device):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')   # Pretrained semantic model
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    return model
