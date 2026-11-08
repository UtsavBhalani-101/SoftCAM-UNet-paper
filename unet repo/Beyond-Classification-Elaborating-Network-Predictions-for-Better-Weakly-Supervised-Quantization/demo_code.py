# the file for draw gradcam images
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
import warnings, torch
import pandas as pd
warnings.filterwarnings('ignore')
import random
import copy
import os, importlib
from torchvision import models, transforms 
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torchvision.datasets import ImageFolder

from torchsummary import summary
import numpy as np
import cv2
import requests
import shutil
from models.model_segrdnet import SegRDNet

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

FLAGS = flags.FLAGS


flags.DEFINE_string("model_checkpoint", "/home/chihchieh/rdnet/output/train/20250907-114136-segrd-512/checkpoint-16.pth.tar", help="model checkpoint path")
flags.DEFINE_string("png_dir", "./demo_dir", help="input_directory") 
flags.DEFINE_string("cam_path", "./experiments_segrd_segmentation_out", help="output_directory")

class ich_dataset(nn.Module):

    def __init__(self, png_dir, transform, key = '.png'):
        super().__init__()
        self.img_list = [os.path.join(png_dir, x) for x in os.listdir(png_dir) if key in x]
        self.transform = transform
        print( 'number of images', len(self.img_list))
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        img = np.array(Image.open(self.img_list[i]).convert('RGB'))
        img = self.transform(img)
        return img, self.img_list[i]


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
        
def main(argv):

    print(
        "model_checkpoint, png_dir, cam_path:",
        FLAGS.model_checkpoint,
        FLAGS.png_dir,
        FLAGS.cam_path,
    )


    model = SegRDNet(num_classes=6) 

  
    checkpoint = torch.load(FLAGS.model_checkpoint )
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    print("load checkpoint for the backbone")
    

    cam_path = FLAGS.cam_path
    png_dir  = FLAGS.png_dir
    assert os.path.exists(png_dir)
    os.makedirs(cam_path,exist_ok= True)
    
    dataset_eval = ich_dataset(png_dir, transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((512,512)), transforms.Normalize(mean= [0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])]) ) 

    L = dataset_eval.__len__()
    thresholds = [-1.0854573, -2.9508734, -2.0274892, -3.6829848, -2.9302106, -4.741706] # thresholds cal with max (TP -FP) on the validation dataset, corresponds to ICH, IPH, IVH, SAH, SDH, EDH
    label_dict = {0 : 'ICH', 1:'IPH', 2:'IVH', 3:'SAH', 4: 'SDH', 5:'EDH'}
    for i in range(L):

        input_tensor, img_path = dataset_eval.__getitem__(i)
        input_tensor = input_tensor.unsqueeze(0).cuda()
        
        output = model(input_tensor)
        
        for curr_index in range(6):
            if output[0,curr_index]> thresholds[curr_index]:
                try:
                    classifier = ClassNet(model,curr_index)
                    targets = [BinaryClassifierOutputTarget(1)]
                
                    target_layers = [classifier.model.lse3.check] # layer from the heatmap generation
                    with GradCAM(model=classifier, target_layers=target_layers) as cam:
                        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
            
                    cam = np.uint8(255*grayscale_cams[0, :])
  
                    heatmap = cv2.resize(cam, dsize=(512,512),
                                             interpolation=cv2.INTER_CUBIC)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)                
                    filename = os.path.join(cam_path,img_path.split('/')[-1].replace('.png', '_'+ str(label_dict[curr_index])+ '.png'))
         
                
                    cv2.imwrite(filename, heatmap)
                except Exception as ee:
                    print(ee)


if __name__ == "__main__":
      
    app.run(main)






