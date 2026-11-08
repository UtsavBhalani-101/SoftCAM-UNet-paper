import cv2, os, PIL, shutil
import numpy as np, nibabel as nib
from PIL import Image
from pydicom import dcmread

def nii_to_png_3ch(filename,output_dir,ww = 130,wc = 25):
        
    ds = nib.load(filename)
        
        
    np_data = np.array(ds.dataobj)
    lower_bound = wc - 0.5*ww
    upper_bound = wc + 0.5*ww

    np_data = np.where(np_data < lower_bound, lower_bound, np_data )
    np_data = np.where(np_data > upper_bound, upper_bound, np_data)

    np_data = 255/(upper_bound - lower_bound)*(np_data - lower_bound)
    np_data = np_data.astype('uint8')

    l = np_data.shape[2] 
        
    for i in range(l):
        path = os.path.join(output_dir,filename.split('/')[-1].split('.')[0]+"_"+str(i)+'.png')
        prev = np.flip(np.swapaxes(np_data[:,:,max(i-1,0)],0,1))
        curr = np.flip(np.swapaxes(np_data[:,:,i],0,1))
        pro = np.flip(np.swapaxes(np_data[:,:,min(i+1,l-1)],0,1))
        image = np.stack([prev,curr,pro],axis= 2)
            
        im = Image.fromarray(image)
        im.save(path) 
        


def dcm_to_np(filename):
    ds = dcmread(filename)

    try:
        slope=ds.RescaleSlope
        intercept=ds.RescaleIntercept
    except:
        slope =1 
        intercept =0

    np_data = ds.pixel_array*slope + intercept

    return np_data

 def dcm_to_png_3ch(dcm_dir,output_dir, ww = 130, wc = 25):
     lower_bound = wc - 0.5*ww
     upper_bound = wc + 0.5*ww
     
     in_dict = {}
     img_list = [os.path.join(dcm_dir,x) for x in os.listdir(dcm_dir) if '.dcm' in x]
     for f in img_list:
        ds = dcmread(f)
        in_dict[f] = ds.get('InstanceNumber', -100000)
        assert in_dict[f] > -100000
     
     sorted_list = [(k, v) for k, v in sorted(in_dict.items(), key=lambda item: item[1])]
     
     l = len(sorted_list):
     for i in range(l):
         curr_index = i
         prev_index = max(i-1, 0)
         pro_index = min(i+1, l-1)
         
         curr = dcm_to_np((sorted_list[curr_index]))
         pre =  dcm_to_np((sorted_list[pre_index]))
         pro =  dcm_to_np((sorted_list[pro_index]))
 
         np_data = np.stack([prev,curr,pro],axis= 2)
         np_data = np.where(np_data < lower_bound, lower_bound, np_data )
         np_data = np.where(np_data > upper_bound, upper_bound, np_data)
         np_data = 255/(upper_bound - lower_bound)*(np_data - lower_bound)
         np_data = np_data.astype('uint8')
         
         im = Image.fromarray(np_data)
         path = os.path.join(output_dir, sorted_list[curr_index].split('/')[-1])
        
         im.save(path)  
