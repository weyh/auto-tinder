from functools import lru_cache

import torch
from torch import nn
import torchvision.models as models

IMG_HEIGHT = 240
IMG_WIDTH = 240

NORM_VECS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


@lru_cache(maxsize=1)
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False

        # Replace final layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)