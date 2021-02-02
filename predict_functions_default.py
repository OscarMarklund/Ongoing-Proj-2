import numpy as np
import torch.nn.functional as F

import argparse
import json
import time
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import os
from collections import OrderedDict
import PIL
from PIL import Image

############## LOAD THE USER'S CHECKPOINT BY DEFAULT OR ANOTHER CHECKPOINT OF THEIR CHOICE WITH ASSOCIATED DIRECTORY

def load_checkpoint(ch_path):
    checkpoint = torch.load(ch_path)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    epochs = checkpoint['epochs']
    model.load_state_dict = (checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

model = load_checkpoint('/home/workspace/ImageClassifier/checkpoint.pth')


#INFERENCING
# Process image via transforms
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''  
    image = Image.open(image)
    
    preprocess = transforms.Compose([transforms.Resize(255), #255x255
                                     transforms.CenterCrop(224), #cropped centrally to 224x224
                                     transforms.ToTensor(), #makes tensor and transposition is not necessary for it is not an np.array
                                     transforms.Normalize([0.485, 0.456, 0.406], #normalises via mean
                                                          [0.229, 0.224, 0.225])]) #normalises via std deviations
    
    image = preprocess(image)
    return image

############# USER INPUTS IMAGE OF THEIR CHOICE
process_image('/home/workspace/ImageClassifier/flowers/test/21/image_06805.jpg')


########## USER INPUTS GPU OR CPU
gpu = True
if gpu is True:
    if torch.cuda.is_available() is True:
        device = torch.device('cuda')
        print('GPU will gladly inference')
    else:
        device = torch.device('cpu')
        print('GPU was not found so CPU is being used')
else:
    device = torch.device('cpu')
    print('Inferencing will commence on CPU')
        
        
#Class Prediction
####### USER DECIDES TOPK
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model with USER CHOICE of category names.
    '''
        
    model.to(device)
    image = process_image(image_path)
    image = image.to(device)
    image_classes_dict = {v: k for k, v in model.class_to_idx.items()}
    model.eval()
    
    with torch.no_grad():
        image.unsqueeze_(0)
        output = model.forward(image)
        ps = torch.exp(output)
        probs, classes = ps.topk(topk)
        probs, classes = probs[0].tolist(), classes[0].tolist()
        
        return_classes = []
        for c in classes:
            return_classes.append(image_classes_dict[c])
        ############ USER DECIDES JSON FILE    
        #with open('cat_to_name.json', 'r') as f:
        #    cat_to_name = json.load(f)
    
        #for x in return_classes:
            #flower_classes.append(cat_to_name[x])
            
        return probs, return_classes
  

probs, classes = predict('/home/workspace/ImageClassifier/flowers/test/21/image_06805.jpg', model, 5)
print(probs)
print(classes)


    

#(cat_to_name[classes[n_classes-1]])