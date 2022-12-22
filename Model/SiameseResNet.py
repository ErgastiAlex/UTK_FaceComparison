from torchvision import models
import torch

class SiameseResNet(torch.nn.Module):
    def __init__(self,hidden_layers=[100], resnet_type=models.resnet50, use_dropout=False, dropout_p=0.5):
        super().__init__() 


        #create two siamese resnet and remove the last not used layer
        temporal_resnet=resnet_type()
        self.resnet_1 = torch.nn.Sequential(*(list(temporal_resnet.children())[:-1]))

        feature_map_dim=2*temporal_resnet.fc.in_features


        if len(hidden_layers)!=0:
            fc_layers=[torch.nn.Linear(feature_map_dim,hidden_layers[0],bias=True)]
            if use_dropout:
                fc_layers.append(torch.nn.Dropout(p=dropout_p))
            fc_layers.append(torch.nn.ReLU())


            for i in range(len(hidden_layers)-1):
                fc_layers.append(torch.nn.Linear(hidden_layers[i],hidden_layers[i+1],bias=True))
                if use_dropout:
                    fc_layers.append(torch.nn.Dropout(p=dropout_p))
                fc_layers.append(torch.nn.ReLU())


            fc_layers.append(torch.nn.Linear(hidden_layers[-1],1,bias=True))
            fc_layers.append(torch.nn.Sigmoid())
        else:
            fc_layers=[torch.nn.Linear(feature_map_dim,1,bias=True)]
            fc_layers.append(torch.nn.Sigmoid())

        
        self.fc=torch.nn.Sequential(*fc_layers)


    def forward(self, x):
        img1,img2=torch.split(x,1,1) # [x, 2, 3, 200, 200] -> [x, 1, 3, 200, 200] [x, 1, 3, 200, 200]

        img1=img1.squeeze(1) # [x, 1, 3, 200, 200] -> [x, 3, 200, 200]
        img2=img2.squeeze(1) # [x, 1, 3, 200, 200] -> [x, 3, 200, 200]

        feature_map1=self.resnet_1(img1)
        feature_map1=feature_map1.view(feature_map1.size(0), -1) # [x, 2048, 1, 1] -> [x, 2048]

        feature_map2=self.resnet_1(img2)
        feature_map2=feature_map2.view(feature_map2.size(0), -1) # [x, 2048, 1, 1] -> [x, 2048]

        feature_map=torch.cat((feature_map1,feature_map2),1) # [x, 2048] [x, 2048] -> [x, 4096]

        return self.fc(feature_map)


def main():
    m=SiameseResNet(hidden_layers=[100,100])
    print(m)


if __name__== "__main__":
    main()
