# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 21:51:12 2019

@author: Zhao Jianjie
"""
#%%

from torch.utils.data import Dataset
import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image
import linecache as lc

#%% generate dataloader 
class loaddataset(Dataset):
    def __init__(self, path, transforms):

        self.path = path
        self.label_ = os.listdir(self.path)
        self.transforms = transforms
    
    def __len__(self):
        return len(os.listdir(self.path))
    
    def __getitem__(self, idx):
        line = lc.getline('list_attr_celeba.txt', idx+2) # get label from the file 
        line = line.rstrip('\n')
        file = line.split() 

        image_name = os.path.join(self.path,
                                self.label_[idx])
        image = Image.open(image_name)

        image = self.transforms(image)
        # save the label to respective variables
        iAttractive = []
        iEyeGlasses = []
        iMale = []
        iMouthOpen = []
        iSmiling = []
        iYoung = []
        ibrownhair = []
        ihat = []
        ilipstick = []
        iovalface = []

        iAttractive = np.asarray([0 if float(file[3])==-1 else 1])
        iEyeGlasses = np.asarray([0 if float(file[16])==-1 else 1])
        iMale = np.asarray([0 if float(file[21])==-1 else 1])
        iMouthOpen = np.asarray([0 if float(file[22])==-1 else 1])
        iSmiling = np.asarray([0 if float(file[32])==-1 else 1])
        iYoung = np.asarray([0 if float(file[40])==-1 else 1])
        ibrownhair = np.asarray([0 if float(file[23])==-1 else 1])
        ihat = np.asarray([0 if float(file[36])==-1 else 1])
        ilipstick = np.asarray([0 if float(file[37])==-1 else 1])
        iovalface = np.asarray([0 if float(file[26])==-1 else 1])
        sample = {'image': image, 'Attractive': iAttractive, 'EyeGlasses': iEyeGlasses, 'Male': iMale, \
                  'MouthOpen': iMouthOpen, 'Smiling': iSmiling, 'Young': iYoung , 'brown hair':ibrownhair,'hat':ihat ,'lipstick':ilipstick,'oval face':iovalface}       
        
        return sample


























