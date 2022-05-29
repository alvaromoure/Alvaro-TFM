import torchvision.models as models
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
import torch


def addDropout(net, p=0.1):
    for name in net.features._modules.keys():
        if name != "conv0":
            net.features._modules[name] = addDropoutRec(net.features._modules[name], p=p)
    net.classifier = addDropoutRec(net.classifier, p=p)
    return net


def addDropoutRec(module, p):
    if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
        return nn.Sequential(module, nn.Dropout(p))
    for name in module._modules.keys():
        module._modules[name] = addDropoutRec(module._modules[name], p=p)

    return module


class myDenseNet(nn.Module):
    """
    see https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
    """

    def __init__(self, n_classes=3, in_features=1024):
        super(myDenseNet, self).__init__()
        net = models.densenet121(pretrained=True)
        self.features = net.features
        self.classifier = nn.Sequential(Linear(in_features=in_features, out_features=n_classes),
                                        nn.Sigmoid())  # Linear(in_features=in_features, out_features=n_classes)#

    def forward(self, x):
        activations = []
        for feat in self.features:
            x = feat(x)
            activations.append(x)

        out = F.relu(x, inplace=True)
        activations.append(out)
        out = F.avg_pool2d(out, kernel_size=7, stride=1)
        # out = F.max_pool2d(out, kernel_size=14, stride=1)
        out = out.view(x.size(0), -1)
        out = self.classifier(out)
        activations.append(out)
        return activations


class myDenseNet_v2(nn.Module):
    """
    see https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
    """

    def __init__(self, n_classes=3, in_features=1024):
        super().__init__()
        model = models.densenet121(pretrained=True)
        self.pretrained_model = model
        self.features = model.classifier.out_features
        self.dense1 = nn.Linear(self.features, 256)  # equivalent to Dense in keras
        self.Dropout = nn.Dropout(0.2)
        self.classifer = nn.Linear(256, n_classes)
        self.n_classes = n_classes

    def forward(self, x):
        features = self.pretrained_model(x)
        dense = nn.ReLU()(self.dense1(features))
        regularization = self.Dropout(dense)
        output = self.classifer(regularization)
        return output


def NewDenseNet(n_classes=3, p=0.2):
    DenseNet = myDenseNet(n_classes=n_classes)
    if p > 0:
        DenseNet = addDropout(DenseNet, p=p)

    return DenseNet
