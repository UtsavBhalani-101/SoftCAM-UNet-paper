# BHSD dataset for evaluation
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import os, numpy as np
from PIL import Image
from natsort import natsorted
import PIL,cv2,json 
from einops import repeat, rearrange


class ICH_B_ALL_Dataset(Dataset):
    def __init__(self,  root_dir="/home/chihchieh/BHSD/output",img_size=512,mode ='train',nb_classes=6, mean =0.5, std =0.5): 
        
        print('root_dir',root_dir) # dir for BHSD dataset
        self.img_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'labels')
        self.img_size = img_size
        self.case_stat = json.load(open('/home/chihchieh/BHSD/stat_report.json', 'rb'))
        self.slice_stat = json.load(open('/home/chihchieh/BHSD/slice_stat.json', 'rb'))
        self.id_to_label = {1:'edh',2:'ich',5:'ivh',4:'sah',3:'sdh'}
        self.nb_classes = nb_classes
        self.mode = mode
       
        self.key_ratio = round(255/(self.nb_classes -1)) 
        
        self.dir_list = [x for x in os.listdir(self.img_dir) ]
        
        self.preprocessing()
        print('dir length: ', len(self.dir_list))
        self.mean = mean
        self.std = std
        print('mean an std: ', self.mean, self.std)
    
    def preprocessing(self):
        self.final_list = []
        for dirname in self.dir_list:
            dirpath = os.path.join(self.img_dir, dirname)
            for i_name in os.listdir(dirpath):
                final_dict = {}
                img_name = os.path.join(dirpath,i_name)
                init_image = Image.open(img_name)
                label_name = img_name.replace(self.img_dir,self.mask_dir)
                targetmask = (Image.open(label_name).convert('L'))

                if init_image.size[0] != self.img_size or init_image.size[1] != self.img_size:
                
                    init_image=init_image.resize((self.img_size, self.img_size))
                    targetmask = targetmask.resize((self.img_size, self.img_size),resample =  PIL.Image.NEAREST)  

                init_image = np.array(init_image)
            
                targetmask = np.array(targetmask)

                targetmask = np.round((targetmask*(1/self.key_ratio)))

                final_dict['filename'] =  img_name
                final_dict['image'] =  init_image 
                final_dict['mask'] = targetmask 
                final_dict['stat'] = self.slice_stat[i_name]
                self.final_list.append(final_dict)

    def __len__(self):
        
        print('current len', len(self.final_list))
        
        return len(self.final_list)

    def __getitem__(self, index):
        final_dict = self.final_list[index]
        init_image = final_dict['image']
        targetmask = final_dict['mask']
        image_name = final_dict['filename']
        stat = final_dict['stat']
        stat[0] = max(stat[1:])

            
        tr = transforms.Compose([   transforms.ToTensor(), transforms.Normalize(mean= self.mean, std =self.std)    ])
        return tr(init_image), targetmask,  image_name, np.array(stat)


        

    

