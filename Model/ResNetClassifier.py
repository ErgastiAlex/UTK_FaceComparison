from torchvision import models
import torch

class ResNetClassifier(torch.nn.Module):
    def __init__(self,hidden_layers=[100], resnet_type=models.resnet50, use_dropout=False, dropout_p=0.5):
        super().__init__() 


        self.resnet = resnet_type()

        feature_map_dim=self.resnet.fc.in_features

        print(self.resnet)

        #Override the first layer to accept 6 channels instead of 3
        self.resnet.conv1=torch.nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

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

        self.resnet.fc=torch.nn.Sequential(*fc_layers)


    def forward(self, x):
        img1,img2=torch.split(x,1,1) # [x, 2, 3, h, w] -> [x, 1, 3, h, w] [x, 1, 3, h, w]

        img1=img1.squeeze(1) # [x, 1, 3, h, w] -> [x, 3, h, w]
        img2=img2.squeeze(1) # [x, 1, 3, h, w] -> [x, 3, h, w]

        images=torch.cat((img1,img2),1) # [x, 3, h, w] [x, 3, h, w] -> [x, 6, h, w]
        outputs=self.resnet(images)

        return outputs



def main():
    m=ResNetClassifier(hidden_layers=[],resnet_type=models.resnet18,use_dropout=False,dropout_p=0.5)
    print(m)
    #check if the model is working
    x=torch.rand(1,2,3,224,224)
    m(x)



if __name__== "__main__":
    main()
