import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from efficientnet_pytorch import EfficientNet

def get_model(num_classes):
    #model = models.resnet50(pretrained=True, progress=True)
    model = EfficientNet.from_pretrained('efficientnet-b8')

    # change the last linear layer
    num_ftrs = model.fc.in_features
    #num_ftrs = model.classifier.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    #model.classifier = nn.Linear(num_ftrs, num_classes)
    return model
