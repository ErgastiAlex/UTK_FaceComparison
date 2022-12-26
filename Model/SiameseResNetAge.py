from torchvision import models
import torch
from Model.BaseClassifier import BaseClassifier


class SiameseResNetAge(BaseClassifier):

    def forward(self, x):
        img1,img2=torch.split(x,1,1) # [x, 2, 3, h, w] -> [x, 1, 3,h, w] [x, 1, 3,h, w]

        img1=img1.squeeze(1) # [x, 1, 3, h, w] -> [x, 3, h, w]
        img2=img2.squeeze(1) # [x, 1, 3, h, w] -> [x, 3, h, w]

        feature_map1=self.resnet_1(img1)
        feature_map1=feature_map1.view(feature_map1.size(0), -1) # [x, 2048, 1, 1] -> [x, 2048]

        feature_map2=self.resnet_1(img2)
        feature_map2=feature_map2.view(feature_map2.size(0), -1) # [x, 2048, 1, 1] -> [x, 2048]

        feature_map=torch.cat((feature_map1,feature_map2),1) # [x, 2048] [x, 2048] -> [x, 4096]

        return self.fc(feature_map)

    def implement_classifier(self,resnet_type):
        temporal_resnet=resnet_type()
        self.resnet_1 = torch.nn.Sequential(*(list(temporal_resnet.children())[:-1]))
        self.feature_map_dim=2*temporal_resnet.fc.in_features

    def get_feature_map_dim(self):
        return self.feature_map_dim


def main():
    m=SiameseResNet(hidden_layers=[],resnet_type=models.resnet18,use_dropout=False,dropout_p=0.5)
    print(m)


if __name__== "__main__":
    main()
