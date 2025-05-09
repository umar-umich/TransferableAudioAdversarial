from compose_models import get_cnn, get_msresnet, get_rawboost, get_rawnet2, get_resnet, get_sentence_transformer, get_ssdnet, get_inc_ssdnet, get_wav2vec2_model

def load_audio_models(models_list, device):
    # Mapping model names to corresponding functions
    model_functions = {
        "ssdnet": lambda: get_ssdnet('original', device),
        "inc_ssdnet": lambda: get_inc_ssdnet('original', device),
        "rawboost": lambda: get_rawboost(device),
        "rawnet2": lambda: get_rawnet2(device),
        "resnet": lambda: get_resnet(device),
        "msresnet": lambda: get_msresnet(device),
        "cnn": lambda: get_cnn(device)
    }

    # Dynamically populate the dictionary based on models_list
    cl_models = {name: model_functions[name]() for name in models_list if name in model_functions}

    # Load transcription and sentence transformer models
    t_processor_1, t_model_1 = get_wav2vec2_model(device)
    sentence_transformer = get_sentence_transformer(device)

    return cl_models, t_processor_1, t_model_1, sentence_transformer
