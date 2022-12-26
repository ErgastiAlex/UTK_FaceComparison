from torchvision import models
import torch

class BaseClassifier(torch.nn.Module):
    def __init__(self,hidden_layers=[100], resnet_type=models.resnet50, use_dropout=False, dropout_p=0.5):
        super().__init__() 


        self.implement_classifier(resnet_type)
        feature_map_dim=self.get_feature_map_dim()

        fc_layers=[]

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
        raise NotImplementedError

    def implement_classifier(self,resnet_type):
        raise NotImplementedError

    def get_feature_map_dim(self):
        raise NotImplementedError