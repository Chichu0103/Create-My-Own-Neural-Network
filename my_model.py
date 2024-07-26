from torchvision import models
from torch import nn, optim


def create_nn_model(model_name, hidden, lr, gpu=False):
    model = None

    print(model_name)
    print(hidden)
    print(lr)

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False
        
        model.classifier = nn.Sequential(nn.Linear(25088, hidden),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.2),
                                         nn.Linear(hidden, 102),
                                         nn.LogSoftmax(dim=1))

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained = True)

        for param in model.parameters():
            param.requires_grad = False

        model.classifier = nn.Sequential(nn.Linear(2048, hidden),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.2),
                                         nn.Linear(hidden, 102),
                                         nn.LogSoftmax(dim=1))        
    
    else:
        print("No such model, select either 'vgg' or 'resnet'.")
        exit()


    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),  lr)

    if gpu == True:
        model.to("cuda")
        criterion.to("cuda")

    return model, criterion, optimizer