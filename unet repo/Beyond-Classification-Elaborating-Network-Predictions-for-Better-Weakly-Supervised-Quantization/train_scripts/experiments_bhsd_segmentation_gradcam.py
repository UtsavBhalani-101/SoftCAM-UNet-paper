# the file for draw gradcam images
from sklearn.metrics import roc_auc_score , roc_curve

from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from rdnet.rdnet import *
import warnings, torch
import pandas as pd
warnings.filterwarnings('ignore')
import random
import copy
import os, importlib
from omegaconf import OmegaConf
from torchvision import models, transforms 
from torchvision.transforms import Compose, Normalize, ToTensor
from torchsummary import summary
import numpy as np
import cv2
import requests

from ich_dataset_bhsd_normalize_middle import ICH_B_ALL_Dataset
import shutil
from test_se.test_se_rdnet import SegRDNet

import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget,BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from PIL import Image

from absl import app, flags

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



class ClassNet(torch.nn.Module):
    def __init__(self, model, index):
        super(ClassNet, self).__init__()
        self.model = model
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.index = index
        
    def forward(self, x):
        final = self.model(x)
        
        return final[:,self.index]

def make_label(mask):
    #{1:'edh',2:'ich',5:'ivh',4:'sah',3:'sdh'}
    l_set = np.unique(mask)
    transfer_dict = {1:5, 2:1, 5:2, 4:3, 3:4} # key: label for BHSD dataset, val: label for our model
    label = np.array(6*[0])
    for i in range(1,6):
        if i in l_set:
            t_i = transfer_dict[i]
            label[t_i] = 1
    if np.sum(label) >=1:
        label[0] = 1
    return label
        

if __name__ == "__main__":
    

    model = SegRDNet(num_classes=6) 

  
    checkpoint = torch.load('./output/train/20250907-114136-segrd-512/checkpoint-16.pth.tar' )
    model.load_state_dict(checkpoint['state_dict'])
    print("load checkpoint for the backbone")
    
    

    model.cuda() 
    cam_path = "./experiments_segrd_bhsd_segmentation"
    os.makedirs(cam_path,exist_ok= True)
    

    
    dataset_eval = ICH_B_ALL_Dataset( img_size = 512,mean = [0.485, 0.456, 0.406] ,std = [0.229, 0.224, 0.225],mode='val')
    count = 10
    L = dataset_eval.__len__()
    thresholds = [-1.0854573, -2.9508734, -2.0274892, -3.6829848, -2.9302106, -4.741706] # thresholds cal with max (TP -FP) on the validation dataset
    for i in range(L):

        input_tensor, mask, img_path, _ = dataset_eval.__getitem__(i)
        input_tensor = input_tensor.unsqueeze(0).cuda()
        
        output = model(input_tensor)
        label = make_label(mask)
        if np.sum(label) >= 1:
            for curr_index in range(6):
              if label[curr_index] > 0 or output[0,curr_index]> thresholds[curr_index]:
                try:
                    classifier = ClassNet(model,curr_index)
                    targets = [BinaryClassifierOutputTarget(1)]
                
                    target_layers = [classifier.model.lse3.check] # layer from the heatmap generation
                    with GradCAM(model=classifier, target_layers=target_layers) as cam:
                        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
            
                    cam = np.uint8(255*grayscale_cams[0, :])

                
                  
                    heatmap = cv2.resize(cam, dsize=(512,512),
                                             interpolation=cv2.INTER_CUBIC)
                
               
                    
                    filename = os.path.join(cam_path,img_path.split('/')[-1].replace('.png', '_'+ str(curr_index)+ '.png'))
         
                
                    cv2.imwrite(filename, heatmap)
                except Exception as ee:
                    print(ee)

                         


            count -= 1

      







