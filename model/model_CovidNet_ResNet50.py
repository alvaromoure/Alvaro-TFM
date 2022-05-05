import torch
import torch.nn as nn
from torchvision import models


class CovidNet_ResNet50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        model = models.resnet50(pretrained=True)
        layer_list = list(model.children())[:-2]
        self.pretrained_model = nn.Sequential(*layer_list)

        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.dense1 = nn.Linear(2048, 256)  # equivalent to Dense in keras
        self.Dropout = nn.Dropout(0.2)
        self.classifer = nn.Linear(256, n_classes)
        self.n_classes = n_classes

    def forward(self, x):
        #x = torch.squeeze(x, dim=0)
        features = self.pretrained_model(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        #regularization = self.Dropout(pooled_features)
        dense = self.dense1(pooled_features)
        regularization = self.Dropout(dense)
        #flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]#En el paper original se procesan varios slices de CT
        #print(flattened_features.shape)
        output = self.classifer(regularization)
        return output
