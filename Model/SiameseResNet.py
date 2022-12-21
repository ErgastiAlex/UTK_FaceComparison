from torchvision import models
import torch

class SiameseResNet(torch.nn.Module):
    def __init__(self,hidden_layer_dim=100):
        super().__init__() 


        #create two siamese resnet and remove the last not used layer
        temporal_resnet=models.resnet18()
        self.resnet_1 = torch.nn.Sequential(*(list(temporal_resnet.children())[:-1]))

        feature_map_dim=2*temporal_resnet.fc.in_features
        
        self.fc=torch.nn.Sequential(
            torch.nn.Linear(feature_map_dim,100,bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim,1,bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        img1,img2=torch.split(x,1,1) # [x, 2, 3, 224, 224] -> [x, 1, 3, 224, 224] [x, 1, 3, 224, 224]

        img1=img1.squeeze(1) # [x, 1, 3, 224, 224] -> [x, 3, 224, 224]
        img2=img2.squeeze(1) # [x, 1, 3, 224, 224] -> [x, 3, 224, 224]

        feature_map1=self.resnet_1(img1)
        feature_map1=feature_map1.view(feature_map1.size(0), -1) # [2, 512, 1, 1] -> [2, 512]

        feature_map2=self.resnet_1(img2)
        feature_map2=feature_map2.view(feature_map2.size(0), -1) # [2, 512, 1, 1] -> [2, 512]

        feature_map=torch.cat((feature_map1,feature_map2),1)

        return self.fc(feature_map)


def main():
    m=SiameseResNet()
    print(m)


if __name__== "__main__":
    main()
