from torchvision import models
import torch

class DoubleResNet(torch.nn.Module):
    def __init__(self,hidden_layer_dim=100):
        super().__init__() 


        #create two siamese resnet and remove the last not used layer
        self.resnet_1=models.resnet18()
        feature_map_dim=2*self.resnet_1.fc.in_features
        
        del self.resnet_1.fc

        self.fc=torch.nn.Sequential(
            torch.nn.Linear(feature_map_dim,100,bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim,1,bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        img1=x[0]
        img2=x[1]

        feature_map1=self.resnet_1(img1)
        feature_map2=self.resnet_1(img2)
        
        feature_map=torch.cat((feature_map1,feature_map2),1)

        return self.fc(feature_map)


def main():
    m=DoubleResNet()
    print(m)


if __name__== "__main__":
    main()
