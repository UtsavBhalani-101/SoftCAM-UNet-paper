import cv2
import torch
from torchvision import transforms

def basic_xchest_transform(cfg, p=0.5):
    # used for the XAI challenge
    normalization = [
        transforms.Resize((cfg.data.input_size)), 
        transforms.ToTensor(),
        transforms.CenterCrop(cfg.data.input_size),
        transforms.Normalize(cfg.data.mean, cfg.data.std)
    ]

    train_preprocess = transforms.Compose([
        transforms.RandomHorizontalFlip(p=p),
        transforms.RandomVerticalFlip(p=p),
        transforms.RandomRotation(degrees=30, fill=0),
        transforms.ColorJitter(contrast=(0.8, 1.4), brightness=(0.8, 1.1)),
        *normalization ])
    
    val_preprocess = transforms.Compose([ *normalization])
    
    return train_preprocess, val_preprocess