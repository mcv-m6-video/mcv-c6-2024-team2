import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import sqlite3
from sqlite3 import Error


class FeatureExtractor:
    
    def __init__(self, model_name='resnet50', pretrained=True):


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = getattr(models, model_name)(pretrained=pretrained)



        # Modify the model to remove the last fully connected layer
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model = self.model.to(self.device)
        self.model.eval()




        self.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    



    def extract_features(self, frame):
        # input_tensor = torch.from_numpy(frame)
        input_tensor = self.transform(frame)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # # If GPU is available, move tensor to GPU
        # if torch.cuda.is_available():
        #     input_tensor = input_tensor.cuda()
        
        # Perform forward pass to get the feature vector
        with torch.no_grad():
            output = self.model(input_tensor)
            
        # Extract feature vector
        feature_vector = output.squeeze().cpu().numpy()
        
        return feature_vector



