#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import skimage.io as io
import random
import cv2
import SimpleITK as sitk
import math
from collections import namedtuple
import sys

# river at https://www.kaggle.com/datasets/gvclsu/water-segmentation-dataset
class riverDataset (Dataset):
    def __init__(self, dataset, imgnpy, size = 256, imgRange = None, transform = None, arg = None, tp = 'train', iter = None):
        self.dataset = dataset # image folder
        self.imgnpy = imgnpy # name list of images
        self.size= size # resize image to this size
        self.imgRange = imgRange # range of used images in imgmpy
        self.transform = transform # transformer to input
        self.arg = arg # potential arg for polygon and filter generator
        self.tp = tp # 'train' for perfect mask; 'box', 'poly', 'filter', 'scribble; NEED CHANGE if require other operations
        self.iter = iter # name of previous mask folder, read from origin mask if None, else read from previous generated mask folder

    def __getitem__(self, i):
        imgList = np.load('{}/{}'.format(self.dataset, self.imgnpy), allow_pickle = True)
        if self.imgRange:
            idx = i + self.imgRange[0]
        else:
            idx = i
        imgFile = imgList[idx]
        if imgFile[0] == 'A':
            imgFolder = "ADE20K"
        else:
            imgFolder = "river_segs"

        imgPath = r"{}/JPEGImages/{}/{}".format(self.dataset, imgFolder, imgFile)
        img = Image.open(imgPath)

        if not self.iter:
            maskPath = r"{}/Annotations/{}/{}.png".format(self.dataset, imgFolder, imgFile[:-4])
        else:
            maskPath = r"{}/{}.jpg".format(self.iter, imgFile[:-4])
        mask = Image.open(maskPath)
        
        if self.size:
            img = img.resize((self.size,self.size))
            img = np.array(img)
            mask = mask.resize((self.size,self.size))
            mask = np.array(mask)
        else:
            img = np.array(img)
            mask = np.array(mask)

        if not self.iter:
            mask = mask[:,:,0]
            if self.tp == 'poly':
                # if it's poly, arg stands for sensitivity of algorithm (the maximum ratio between max distance over line length)
                # details: if AB is a line while a ACB is the curve from origin mask, and C is the furthest point away from line AB
                #          d is the distance from point C to line AB;
                #          if d is smaller than arg * length of AB, than the curve ACB will be degraded to line AB;
                #          otherwise, other point B will be chosen 
                
                # higher arg leads to worse mask (lower iou wrt origin mask) 
                assert(self.arg and self.arg < 1)
                mask = create_poly(mask, self.arg)
            elif self.tp == 'filter':
                # if it's filter, arg is the kernal size of average filter
                # larger kernal size leads to more coarse mask
                # maybe NEED CHANGE, the average filter will be applied 10 times, you may want to change it
                # in trainprocess.py create_filter function, in while loop parameter
                assert(self.arg and self.arg % 2 == 1)
                mask = create_filter(mask, self.arg)
            elif self.tp == 'box':
                mask = create_box(mask)
            elif self.tp == 'scribble':
                mask = create_scribble(mask)
            elif self.tp != 'train':
                print('unknown anno type')
                # maybe you want to raise a error here

        else:
            # do something to modify updated mask
            pass

            
            
        maskShape = mask.shape
        temp = [0 if it == 0 else 1 for it in mask.flatten()]
        mask = np.array(temp).reshape(maskShape)
        Img = torch.tensor(img)
        Img = Img.permute((2, 0, 1))
        Mask = torch.tensor(mask)

        return Img.float(), Mask.long(), imgFile[:-4]
    
    def __len__(self):
        imgList = np.load('{}/{}'.format(self.dataset, self.imgnpy), allow_pickle = True)
        if self.imgRange:
            return self.imgRange[1] - self.imgRange[0]
        else:
            return len(imgList)

# brain segmentation at https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation
class bratDataset (Dataset):
    def __init__(self, dataset, imgnpy, size = 256, imgRange = None, transform = None, arg = None, tp = 'train', iter = None):
        self.dataset = dataset # image folder
        self.imgnpy = imgnpy # name list of images
        self.size= size # resize image to this size
        self.imgRange = imgRange # range of used images in imgmpy
        self.transform = transform # transformer to input
        self.arg = arg # potential arg for polygon and filter generator
        self.tp = tp # 'train' for perfect mask; 'box', 'poly', 'filter', 'scribble; NEED CHANGE if require other operations
        self.iter = iter # name of previous mask folder, read from origin mask if None, else read from previous generated mask folder

    def __getitem__(self, i):
        imgList = np.load('{}/{}'.format(self.dataset, self.imgnpy), allow_pickle = True).item()
        if self.imgRange:
            idx = i + self.imgRange[0]
        else:
            idx = i

        imgFile, imgSlice = imgList[idx]
        imgSlice = int(imgSlice)
        

        path = r"{}/BraTS20_Training_{}".format(self.dataset, imgFile)
        imgType = ['flair', 't1', 't1ce', 't2']
        # maybe NEED CHANGE
        # maybe you want to try to use only one style instead of putting 4 styles together
        imgs = []
        for ttp in imgType:
            name = r"BraTS20_Training_{}_{}.nii".format(imgFile, ttp)
            img = sitk.ReadImage(os.path.join(path, name))
            img = sitk.GetArrayFromImage(img)
            img = img[imgSlice]
            if self.size:
                img = Image.fromarray(img)
                img = img.resize((self.size, self.size))
                img = np.array(img)
            imgs.append(img)
        
        if not self.iter:
            maskName = r"BraTS20_Training_{}_seg.nii".format(imgFile)
            mask = sitk.ReadImage(os.path.join(path, maskName))
            mask = sitk.GetArrayFromImage(mask)
            mask = mask[imgSlice]
            if self.size:
                mask = Image.fromarray(mask)
                mask = mask.resize((self.size,self.size))
                mask = np.array(mask)
        else:
            # name of saved mask is here, in format file name + slice index
            maskPath = r"{}/{}.jpg".format(self.iter, imgFile[:-4] + str(imgSlice))
            mask = Image.open(maskPath)
            if self.size:
                mask = mask.resize((self.size,self.size))
                mask = np.array(mask)
                
        if not self.iter:
            if self.tp == 'poly':
                assert(self.arg and self.arg < 1)
                mask = create_poly(mask, self.arg)
            elif self.tp == 'filter':
                assert(self.arg and self.arg % 2 == 1)
                mask = create_filter(mask, self.arg)
            elif self.tp == 'box':
                mask = create_box(mask)
            elif self.tp == 'scribble':
                mask = create_scribble(mask)
            elif self.tp != 'train':
                print('unknown anno type')

        else:
            # do something to modify updated mask
            pass

            
            
        maskShape = mask.shape      
        temp = [0 if it == 0 else 1 for it in mask.flatten()]
        mask = np.array(temp).reshape(maskShape)


        img = np.dstack((imgs[0], imgs[1], imgs[2], imgs[3]))
        Img = torch.tensor(img)
        Img = Img.permute(2, 0, 1)
        Mask = torch.tensor(mask)

        return Img.float(), Mask.long(), imgFile[:-4] + str(imgSlice)

    def __len__(self):
        imgList = np.load('{}/{}'.format(self.dataset, self.imgnpy), allow_pickle = True).item()
        if self.imgRange:
            return self.imgRange[1] - self.imgRange[0]
        else:
            return len(imgList)
        
# liver at https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation
class liverDataset (Dataset):
    def __init__(self, dataset, imgnpy, size = 256, imgRange = None, transform = None, arg = None, tp = 'train', iter = None):
        self.dataset = dataset # image folder
        self.imgnpy = imgnpy # name list of images
        self.size= size # resize image to this size
        self.imgRange = imgRange # range of used images in imgmpy
        self.transform = transform # transformer to input
        self.arg = arg # potential arg for polygon and filter generator
        self.tp = tp # 'train' for perfect mask; 'box', 'poly', 'filter', 'scribble; NEED CHANGE if require other operations
        self.iter = iter # name of previous mask folder, read from origin mask if None, else read from previous generated mask folder

    def __getitem__(self, i):
        imgList = np.load('{}/{}'.format(self.dataset, self.imgnpy), allow_pickle = True).item()
        if self.imgRange:
            idx = i + self.imgRange[0]
        else:
            idx = i

        imgFolder, imgFile, imgSlice = imgList[idx]

        imgPath = r"{}/volume_pt{}/volume-{}.nii".format(self.dataset, imgFolder, imgFile)
        img = sitk.ReadImage(imgPath)
        img = sitk.GetArrayFromImage(img)
        img = img[imgSlice]
        
        if not self.iter:
            maskPath = r"{}/segmentations/segmentation-{}".format(self.dataset, imgFile)
            mask = sitk.ReadImage(maskPath)
            mask = sitk.GetArrayFromImage(mask)
            mask = mask[imgSlice]
            if self.size:
                mask = Image.fromarray(mask)
                mask = mask.resize((self.size,self.size))
                mask = np.array(mask)
        else:
            # name of saved mask is here, in format file name + slice index
            maskPath = r"{}/{}.jpg".format(self.iter, imgFile[:-4] + str(imgSlice))
            mask = Image.open(maskPath)
            if self.size:
                mask = mask.resize((self.size,self.size))
                mask = np.array(mask)
                
        if not self.iter:
            mask = mask[:,:,0]
            if self.tp == 'poly':
                # if it's poly, arg stands for sensitivity of algorithm (the maximum ratio between max distance over line length)
                # details: if AB is a line while a ACB is the curve from origin mask, and C is the furthest point away from line AB
                #          d is the distance from point C to line AB;
                #          if d is smaller than arg * length of AB, than the curve ACB will be degraded to line AB;
                #          otherwise, other point B will be chosen 
                
                # higher arg leads to worse mask (lower iou wrt origin mask) 
                assert(self.arg and self.arg < 1)
                mask = create_poly(mask, self.arg)
            elif self.tp == 'filter':
                # if it's filter, arg is the kernal size of average filter
                # larger kernal size leads to more coarse mask
                # maybe NEED CHANGE, the average filter will be applied 10 times, you may want to change it
                # in trainprocess.py create_filter function, in while loop parameter
                assert(self.arg and self.arg % 2 == 1)
                mask = create_filter(mask, self.arg)
            elif self.tp == 'box':
                mask = create_box(mask)
            elif self.tp == 'scribble':
                mask = create_scribble(mask)
            elif self.tp != 'train':
                print('unknown anno type')
                # maybe you want to raise a error here

        else:
            # do something to modify updated mask
            pass
        
        if self.size:
            img = Image.fromarray(img)
            img = img.resize((self.size,self.size))
            img = np.array(img)


        maskShape = mask.shape
        temp = [0 if it == 0 else 1 for it in mask.flatten()]
        mask = np.array(temp).reshape(maskShape)

        img = np.dstack((img))
        Img = torch.tensor(img)
        Mask = torch.tensor(mask)

        return Img.float(), Mask.long(), imgFile[:-4] + str(imgSlice)
    
    def __len__(self):
        imgList = np.load('{}/{}'.format(self.dataset, self.imgnpy), allow_pickle = True).item()
        if self.imgRange:
            return self.imgRange[1] - self.imgRange[0]
        else:
            return len(imgList)
        
# combine of three human dataset
# imgFrom 0: https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-tiktok-dancing-dataset
# imgFrom 1: https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-mads-dataset
# imgFrom 2: https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset
class personDataset (Dataset):
    def __init__(self, dataset, imgnpy, size = 256, imgRange = None, transform = None, arg = None, tp = 'train', iter = None):
        self.dataset = dataset # image folder
        self.imgnpy = imgnpy # name list of images
        self.size= size # resize image to this size
        self.imgRange = imgRange # range of used images in imgmpy
        self.transform = transform # transformer to input
        self.arg = arg # potential arg for polygon and filter generator
        self.tp = tp # 'train' for perfect mask; 'box', 'poly', 'filter', 'scribble; NEED CHANGE if require other operations
        self.iter = iter # name of previous mask folder, read from origin mask if None, else read from previous generated mask folder

    def __getitem__(self, i):
        imgList = np.load('{}'.format(self.imgnpy), allow_pickle = True).item()
        if self.imgRange:
            idx = i + self.imgRange[0]
        else:
            idx = i
        imgFrom, imgFile = imgList[idx]
        if imgFrom == 0:
            imgFolder = r"dancing/segmentation_full_body_tik_tok_2615_img/images"
            maskFolder = r"dancing/segmentation_full_body_tik_tok_2615_img/masks"
        elif imgFrom == 1:
            imgFolder = r"motion/segmentation_full_body_mads_dataset_1192_img/images"
            maskFolder = r"motion/segmentation_full_body_mads_dataset_1192_img/masks"
        else:
            assert(imgFrom == 2)
            imgFolder = r"person1/supervisely_person_clean_2667_img/images"
            maskFolder = r"person1/supervisely_person_clean_2667_img/masks"

        imgPath = r"{}/{}".format(imgFolder, imgFile)
        img = Image.open(imgPath)
        
        
        
        if not self.iter:
            maskPath = r"{}/{}".format(maskFolder, imgFile)
            mask = Image.open(maskPath)
        else:
            maskPath = r"{}/{}.jpg".format(self.iter, imgFile[:-4])
            mask = Image.open(maskPath)
        
        
        
        
        if self.size:
            img = img.resize((self.size,self.size))
            img = np.array(img)
            mask = mask.resize((self.size,self.size))
            mask = np.array(mask)
        else:
            img = np.array(img)
            mask = np.array(mask)
        
        if not self.iter:
            mask = mask[:,:,0]
            if self.tp == 'poly':
                # if it's poly, arg stands for sensitivity of algorithm (the maximum ratio between max distance over line length)
                # details: if AB is a line while a ACB is the curve from origin mask, and C is the furthest point away from line AB
                #          d is the distance from point C to line AB;
                #          if d is smaller than arg * length of AB, than the curve ACB will be degraded to line AB;
                #          otherwise, other point B will be chosen 
                
                # higher arg leads to worse mask (lower iou wrt origin mask) 
                assert(self.arg and self.arg < 1)
                mask = create_poly(mask, self.arg)
            elif self.tp == 'filter':
                # if it's filter, arg is the kernal size of average filter
                # larger kernal size leads to more coarse mask
                # maybe NEED CHANGE, the average filter will be applied 10 times, you may want to change it
                # in trainprocess.py create_filter function, in while loop parameter
                assert(self.arg and self.arg % 2 == 1)
                mask = create_filter(mask, self.arg)
            elif self.tp == 'box':
                mask = create_box(mask)
            elif self.tp == 'scribble':
                mask = create_scribble(mask)
            elif self.tp != 'train':
                print('unknown anno type')
                # maybe you want to raise a error here

        else:
            # do something to modify updated mask
            pass

        maskShape = mask.shape
        temp = [0 if it == 0 else 1 for it in mask.flatten()]
        mask = np.array(temp).reshape(maskShape)
        Img = torch.tensor(img)
        Img = Img.permute((2, 0, 1))
        Mask = torch.tensor(mask)

        return Img.float(), Mask.long(), imgFile[:-4]
    
    def __len__(self):
        imgList = np.load('{}'.format(self.imgnpy), allow_pickle = True).item()
        if self.imgRange:
            return self.imgRange[1] - self.imgRange[0]
        else:
            return len(imgList)

