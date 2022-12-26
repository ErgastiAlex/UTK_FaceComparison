from torchvision import models
import torch
from Model.BaseClassifier import BaseClassifier

class ResNetClassifier(BaseClassifier):
    """Single ResNet that takes two images as input and outputs who is older"""

    def forward(self, x):
        img1,img2=torch.split(x,1,1) # [x, 2, 3, h, w] -> [x, 1, 3, h, w] [x, 1, 3, h, w]

        img1=img1.squeeze(1) # [x, 1, 3, h, w] -> [x, 3, h, w]
        img2=img2.squeeze(1) # [x, 1, 3, h, w] -> [x, 3, h, w]

        images=torch.cat((img1,img2),1) # [x, 3, h, w] [x, 3, h, w] -> [x, 6, h, w]
        feature_map=self.resnet(images)
        feature_map=feature_map.view(feature_map.size(0), -1) # [x, y, 1, 1] -> [x, y]
        return self.fc(feature_map)

    def implement_classifier(self,resnet_type):
        temporal_resnet=resnet_type()
        temporal_resnet.conv1=torch.nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet = torch.nn.Sequential(*(list(temporal_resnet.children())[:-1]))
        self.feature_map_dim=temporal_resnet.fc.in_features

    def get_feature_map_dim(self):
        return self.feature_map_dim
