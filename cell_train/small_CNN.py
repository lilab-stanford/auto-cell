import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.autograd import Function

class Customresnet(nn.Module):
    def __init__(self, num_classes=4):
        super(Customresnet, self).__init__()
        self.feature = timm.create_model('resnet18', pretrained=True, num_classes=0)
        self.linear = nn.Linear(512, num_classes)
        self.domain_classifier=nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 2)
        )
    def forward(self, x,alpha):
        feature = self.feature(x)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.linear(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output,domain_output

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


