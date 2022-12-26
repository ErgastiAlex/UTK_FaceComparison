from torchvision import models
import torch

class ResNetAgeClassifier(torch.nn.Module):
    def __init__(self, resnet_type=models.resnet50):
        super().__init__() 

        self.resnet = resnet_type()

        fc_layers=[torch.nn.Linear(self.resnet.fc.in_features,1,bias=True)]
        # Use ReLU activation function to predict age
        fc_layers.append(torch.nn.ReLU())

        self.resnet.fc=torch.nn.Sequential(*fc_layers)


    def forward(self, x):
        outputs=self.resnet(x)

        return outputs

    def load_model():
        pass
