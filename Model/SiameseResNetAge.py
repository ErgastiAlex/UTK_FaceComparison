from torchvision import models
import torch
from Model.BaseClassifier import BaseClassifier
from Model.ResNetAgeClassifier import ResNetAgeClassifier
import os
import glob
import Utility.utility as utility

class SiameseResNetAge(BaseClassifier):
    """ Siamese ResNet that takes two images as input and outputs who is older \r\n
        It uses ResNetAgeClassifier as a base classifier and the model .pth must be in <checkpoint_path>/<resnet_type>AgeClassifier/<timestamp>.pth"""
    def forward(self, x):

        img1,img2=torch.split(x,1,1) # [x, 2, 3, h, w] -> [x, 1, 3,h, w] [x, 1, 3,h, w]

        img1=img1.squeeze(1) # [x, 1, 3, h, w] -> [x, 3, h, w]
        img2=img2.squeeze(1) # [x, 1, 3, h, w] -> [x, 3, h, w]

        feature_map1=img1 
        # Exclude FC layer
        for layer in list(self.ageClassifier.resnet.children())[:-1]:
            feature_map1=layer(feature_map1)
        
        feature_map1=feature_map1.view(feature_map1.size(0), -1) # [x, 2048, 1, 1] -> [x, 2048]

        feature_map2=img2
        for layer in list(self.ageClassifier.resnet.children())[:-1]:
            feature_map2=layer(feature_map2)
        
        feature_map2=feature_map1.view(feature_map2.size(0), -1) # [x, 2048, 1, 1] -> [x, 2048]

        feature_map=torch.cat((feature_map1,feature_map2),1) # [x, 2048] [x, 2048] -> [x, 4096]

        return self.fc(feature_map)

    def implement_classifier(self,resnet_class):
        self.ageClassifier = ResNetAgeClassifier(resnet_type=resnet_class)

        self.feature_map_dim=2*self.ageClassifier.resnet.fc[0].in_features
    

    def load_model(self, checkpoint_path, resnet_type):
        """Load the internal resnet model from the last saved model in the folder <checkpoint_path>/<resnet_type>AgeClassifier"""

        model_path=os.path.join(checkpoint_path, resnet_type+'AgeClassifier' )
        saved_models= glob.glob(model_path + '/*.pth') 
    
        last_model= max(saved_models, key=os.path.getctime)

        state_dict=torch.load(last_model)

        self.ageClassifier.load_state_dict(state_dict)

        for param in self.ageClassifier.parameters():
            param.requires_grad = False



    def get_feature_map_dim(self):
        return self.feature_map_dim

