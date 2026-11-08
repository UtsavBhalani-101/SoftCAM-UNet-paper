import bisect, os 
import numpy as np
import albumentations
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset
import json
import cv2
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop, 
    Blur,   
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    Rotate,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    RandomGamma,
    ShiftScaleRotate     
) 

from torchvision import transforms as T



class ImagePathsMICH(Dataset):
    def __init__(self, paths, size=None, mode='train',mean =0.5, std =0.5 ,json_file ="/home/chihchieh/rsna/src/select_ich.json"):
        print('MICH ImagePathsMICH')
        self.size = size

        txt_file = open(paths,'rb')
        path_list = [line.decode('utf8').rstrip('\n') for line in txt_file]
        assert os.path.exists(path_list[0])
        self.labels = dict() 
        self.labels["file_path_"] = path_list
        self._length = len(path_list)
        self.json = json.load(open(json_file,'rb'))
        self.mean = mean
        self.std = std
        print(mean, std)

        self.transform = {
        'train': T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomAffine( degrees=(-30, 30),  translate=(0.1, 0.1),  scale=(0.8, 1.2),  shear=(-10, 10) ),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
            T.Resize((size, size),antialias=False)
        ]),
        'val': T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
            T.Resize((size, size),antialias=False)

        ]),
    }
        self.mode = mode
    def __len__(self):

        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = self.transform[self.mode](image)


    def __getitem__(self, index):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][index])
   
        curr = self.labels["file_path_"][index]
        curr_list = curr.split('/')

        if curr_list[-3] == 'final_train_ich_png_3channels':
            example['classification'] = np.array([max(self.json[curr_list[-2]][curr_list[-1]])]+ self.json[curr_list[-2]][curr_list[-1]] )

        else:
            assert 'final_train_ich_png_3channels' not in curr
            example['classification'] = np.array(6*[0])

        
        return example['image'], example['classification']

