import torch
from my_model import create_nn_model

def load_my_checkpoint(path, gpu = False):
    checkpoint = torch.load(path, map_location="cuda" if gpu else "cpu")

    my_arch = checkpoint['architecture']
    hidden = checkpoint['hidden']
    lr = checkpoint['learning_rate']

    model, criterion, optimizer = create_nn_model(my_arch, hidden, lr, gpu)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.class_to_idx  = checkpoint['class_to_idx']
    

    return model, optimizer, criterion