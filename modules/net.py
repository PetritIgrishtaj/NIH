import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms


def get_model(num_classes):
    model = models.vgg19(pretrained=True, progress=True)

    # change the last linear layer
    num_ftrs = model.fc.in_features
    #num_ftrs = model.classifier.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    #model.classifier = nn.Linear(num_ftrs, num_classes)
    return model
