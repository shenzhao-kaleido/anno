# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from tqdm import trange
from torch.optim import Adam
from torch.nn import DataParallel
from model import UNet
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from pycocotools.coco import COCO
import skimage.io as io
import random
import cv2
import SimpleITK as sitk
import math
from collections import namedtuple
import sys

Point = namedtuple('Point', ['x', 'y'])
EPSILON = math.sqrt(sys.float_info.epsilon)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"

# IoU and IoUAll are to get iou from output prediction
def IoU(y_pre, y_true):
    #y_pre = torch.max(y_pre, 3)
    y_pre = y_pre.int()
    y_true = y_true.int()
    I = torch.bitwise_and(y_pre, y_true).sum().item()
    U = torch.bitwise_or(y_pre, y_true).sum().item()
    if U == 0:
        return 0
    if I > U:
        print('iou counting error') #won't happend
    return float(I)/U

def IoUAll(y_pre, y_true):
    # size = 57600 #brat 240
    size = 65536 #256*256
    y_pre = y_pre.int()
    y_true = y_true.int()
    length = y_pre.size()[0]
    assert(length%size==0)
    pic_num = int(length / size)
    IoUList = []
    for i in range(pic_num):
        pre = y_pre[size*i:size*(i+1)]
        true = y_true[size*i:size*(i+1)]
        iou = IoU(pre, true)
        IoUList.append(iou)
    return np.mean(IoUList)


def learn_model(loaders, save_path, pre, saveFolder):
    # loaders: dataloader for train, test
    # save_path: folder to save trained model
    # pre: name for saved pt file

    model = UNet(num_classes=2, in_channels=2)
    # in_channels: 1 for liver, 2 for river and person, 4 for brain
    model.cuda()
    model = DataParallel(model)
    print('start',pre)
    
    loader_train, loader_valid  = loaders
    optimizer = Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_valid_acc = -0.01
    train_acc_list = []
    valid_acc_list = []

    ct = 0
    stop = 0 # for early stop, if model doesn't improve after one epoch, stop += 1, if stop is 20, then stop training
    with trange(100) as pbar:
    # with trange(100) as pbar:
        for epoch in pbar:
            train_acc, loss = run_epoch(loader_train, model, loss_fn, optimizer)
            valid_acc,_ = run_epoch(loader_valid, model, loss_fn, None)
            train_acc_list.append(train_acc)
            valid_acc_list.append(valid_acc)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(model.state_dict(), os.path.join(save_path, pre+".pt"))
                stop = 0
            pbar.set_description("acc: {:.3f}, v_acc: {:.3f}, best_v_acc: {:.3f}, loss: {:.3f}".format(train_acc, valid_acc, best_valid_acc, loss))
            if ct % 10 == 0: # save every 10 epochs
                np.save(os.path.join(save_path, pre+'_train_temp.npy'), train_acc_list)
                np.save(os.path.join(save_path, pre+'_test_temp.npy'), valid_acc_list)
                torch.save(model.state_dict(), os.path.join(save_path, pre+"_temp.pt"))

                # useless part, to record and get process without get into docker
                # f = open(r'/workspace/process.txt', 'w')
                # f.write(str(ct))
                # f.close()
            ct += 1
            stop += 1
            if stop == 20:
                print('early stop')
                np.save(os.path.join(save_path, pre+'_train_stop.npy'), train_acc_list)
                np.save(os.path.join(save_path, pre+'_valid_stop.npy'), valid_acc_list)
                break
            
    np.save(os.path.join(save_path, pre+'_train.npy'), train_acc_list)
    np.save(os.path.join(save_path, pre+'_valid.npy'), valid_acc_list)
    print('saving current prediction')
    model.load_state_dict(torch.load('{}/{}.pt'.format(save_path, pre)))
    test_epoch(loader_train, model, saveFolder)

def test_epoch(loader, model, saveFolder):
    # to save predictions after one iteration
    model.eval()
    for data in loader:
        inputs, names = data[0].cuda(), data[2]
        outputs = model(inputs)
        _, predicts = torch.max(outputs.permute(0, 2, 3, 1), axis = 3)
        predict = predicts.cpu().detach().numpy()
        assert(len(predict) == len(names))
        for arr, name in zip(predict, names):
            img = Image.fromarray(arr.astype(np.unit8))
            img.save('{}/{}.jpg'.format(saveFolder, name))


# run one epoch
# train if optimizer input, else test model
# if final, save all results
def run_epoch(loader, model, loss_fn, optimizer):
    IoUSum = 0
    LSum = 0
    if optimizer is None:
        model.eval()
    else:
        model.train()

    for data in loader:
        inputs, labels = data[0].cuda(), data[1].cuda()
        outputs = model(inputs)
        if optimizer is not None:
            optimizer.zero_grad()
            loss = 0.0
            output = outputs.permute(2, 3, 0, 1).contiguous().view(-1, 2)
            label = labels.permute(1, 2, 0).contiguous().view(-1)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            LSum += loss.item()

        _, predicts = torch.max(outputs.permute(0, 2, 3, 1), axis = 3)
        predict = predicts.view(-1)
        IoUSum += IoUAll(predict, labels.view(-1))
    # drop last = True for test loader, so no remainings
    acc = IoUSum/len(loader)
    return acc, LSum



# main function
# callback: trainProcess (setup data loader)- learn_model (manage training para)- run_epoch (run one epoch) - test-epoch (save predictions)
def trainProcess(data, dataset, imgnpy, trainR, validR, name, validnpy = None):
    # data: name of dataSet
    # dataset: folder to place images and labels
    # imgnpy: name list for all images
    # trainR: the range of images used in train from imgnpy
    # validR: the range of images used in test from imgnpy, shouldn't overlap with trainR without validnpy
    # name: saved model's name
    # validnpy: the saperated name list for test images, if doesn't exist then will be same as imgnpy
    torch.backends.cudnn.benchmark = True
    # NEED CHANGE: save folder
    save_path = '/workspace/result_f'
    modelpre = name
    # train_transform = transforms.Compose([RandomFlip(), RandomCrop(20)])
    # NEED CHANGE
    # if need scribble, change next line to
    # train = data(dataset, imgnpy, imgRange = trainR, tp = 'scribble')
    # similiarly, you could have
    # train = data(dataset, imgnpy, imgRange = trainR, tp = 'filter', arg = 5) # 5 stands for kernal size 5
    # train = data(dataset, imgnpy, imgRange = trainR, tp = 'poly', arg = 0.05) 
    # train = data(dataset, imgnpy, imgRange = trainR, tp = 'box')
    train = data(dataset, imgnpy, imgRange = trainR) # , transform = train_transform)
    if not validnpy:
        print('same train and valid')
        validnpy = imgnpy
    valid = data(dataset, validnpy, imgRange = validR)
    print('len', len(train), len(valid))
    train_loader = DataLoader(dataset = train, batch_size = 16, num_workers = 8, shuffle = True, drop_last = True)
    valid_loader = DataLoader(dataset = valid, batch_size = 8, num_workers = 2, shuffle = False, drop_last = True)
    saveFolder = 'river_test_1' # folder to save predictions, NEED CHANGE
    learn_model((train_loader, valid_loader), save_path, modelpre, saveFolder)
    iterNum = 10
    for iter in range(iterNum):
        train = data(dataset, imgnpy, imgRange = trainR, iter = saveFolder)
        if not validnpy:
            validnpy = imgnpy
        train_loader = DataLoader(dataset = train, batch_size = 16, num_workers = 8, shuffle = True, drop_last = True)
        learn_model((train_loader, valid_loader), save_path, modelpre, saveFolder)

# polygon generator, rt needs to be tested separately
def create_poly(seg, rt):
    masks = np.zeros((seg.shape))
    mask = seg #* 255.0
    x = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    ret, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    ct, h = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for s in ct:
        approx = cv2.approxPolyDP(s, rt*cv2.arcLength(s, True), True)
        temp = np.zeros((seg.shape))
        tempSum = cv2 .fillPoly(temp,[approx],1)
        # ignore too small part
        if np.sum(tempSum) < 50:
            continue
        cv2 .fillPoly(masks,[approx],1)
    re = np.array(masks)
    return re

# box generator
def create_box(mask):
    masks = np.zeros(mask.shape)
    mask = mask 
    x = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    ret, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    ct, h = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for s in ct:
        x,y,w,h = cv2.boundingRect(s)
        # ignore too small part
        # should be changed if images size is not 256*256
        if w * h < 100 or w <5 or h<5:
            continue
        cv2.rectangle(masks, (x, y), (x + w, y + h), 1, -1)
    np.array(masks)
    return masks

# filter generator
def create_filter(mask, ksize):
    # ksize: kernal size
    # default to run 10 times, can be changed
    mask = Image.fromarray(mask).resize((256,256))
    x = cv2.cvtColor(np.asarray(mask.convert('RGB')).astype(np.uint8), cv2.COLOR_RGB2BGR)
    mask = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    temp = mask.copy()
    for i in range(10):
        temp = cv2.blur(temp, (ksize,ksize))
    masks = cv2.threshold(temp, 0.5*np.max(temp), 1, cv2.THRESH_BINARY)[1]
    masks = np.array(masks)
    return masks

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

# transformer for crop and flip
class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample[0], sample[1]
        h, w = image.size[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, new_h)
        left = np.random.randint(0, new_w)
        image = np.array(image)
        mask = np.array(mask)
        image = image[top:,left:]
        mask = mask[top:,left:]
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        image = image.resize((h, w))
        mask = mask.resize((h,w))
        return image, mask

class RandomFlip(object):
    # def __init__(self, direction):
    #     self.dir = direction
    def __call__(self, sample):
        image, mask = sample[0], sample[1]
        trans = np.random.randint(0,2,2)
        if trans[0]:
            image = ImageOps.flip(image)
            mask = ImageOps.flip(mask)
        if trans[1]:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        return image, mask

# create triangles from polygon, helper function to create scribble
def earclip(polygon):
    ear_vertex = []
    triangles = []

    polygon = [Point(*point) for point in polygon]

    if _is_clockwise(polygon):
        polygon.reverse()

    point_count = len(polygon)
    for i in range(point_count):
        prev_index = i - 1
        prev_point = polygon[prev_index]
        point = polygon[i]
        next_index = (i + 1) % point_count
        next_point = polygon[next_index]

        if _is_ear(prev_point, point, next_point, polygon):
            ear_vertex.append(point)

    while ear_vertex and point_count >= 3:
        ear = ear_vertex.pop(0)
        i = polygon.index(ear)
        prev_index = i - 1
        prev_point = polygon[prev_index]
        next_index = (i + 1) % point_count
        next_point = polygon[next_index]

        polygon.remove(ear)
        point_count -= 1
        triangles.append(((prev_point.x, prev_point.y), (ear.x, ear.y), (next_point.x, next_point.y)))
        if point_count > 3:
            prev_prev_point = polygon[prev_index - 1]
            next_next_index = (i + 1) % point_count
            next_next_point = polygon[next_next_index]

            groups = [
                (prev_prev_point, prev_point, next_point, polygon),
                (prev_point, next_point, next_next_point, polygon),
            ]
            for group in groups:
                p = group[1]
                if _is_ear(*group):
                    if p not in ear_vertex:
                        ear_vertex.append(p)
                elif p in ear_vertex:
                    ear_vertex.remove(p)
    return triangles


def _is_clockwise(polygon):
    s = 0
    polygon_count = len(polygon)
    for i in range(polygon_count):
        point = polygon[i]
        point2 = polygon[(i + 1) % polygon_count]
        s += (point2.x - point.x) * (point2.y + point.y)
    return s > 0


def _is_convex(prev, point, next):
    return _triangle_sum(prev.x, prev.y, point.x, point.y, next.x, next.y) < 0


def _is_ear(p1, p2, p3, polygon):
    ear = _contains_no_points(p1, p2, p3, polygon) and \
        _is_convex(p1, p2, p3) and \
        _triangle_area(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y) > 0
    return ear


def _contains_no_points(p1, p2, p3, polygon):
    for pn in polygon:
        if pn in (p1, p2, p3):
            continue
        elif _is_point_inside(pn, p1, p2, p3):
            return False
    return True


def _is_point_inside(p, a, b, c):
    area = _triangle_area(a.x, a.y, b.x, b.y, c.x, c.y)
    area1 = _triangle_area(p.x, p.y, b.x, b.y, c.x, c.y)
    area2 = _triangle_area(p.x, p.y, a.x, a.y, c.x, c.y)
    area3 = _triangle_area(p.x, p.y, a.x, a.y, b.x, b.y)
    areadiff = abs(area - sum([area1, area2, area3])) < EPSILON
    return areadiff


def _triangle_area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def _triangle_sum(x1, y1, x2, y2, x3, y3):
    return x1 * (y3 - y2) + x2 * (y1 - y3) + x3 * (y2 - y1)

# get shared point for two triangles
def find_same_element(x, y):
    num = 0
    item = []
    for i in x:
        for j in y:
            if list(i) == list(j):
                item.append(i)
                num += 1
    return num, item

# find position of triangle wrt to whole polygon
# how many sides are shared with other triangle
def get_adjacent_points(obj, poly):
    adj_point = []
    adj_num = 0
    for triangle in poly:
        num, item = find_same_element(triangle, obj)
        if num == 2:
            adj_num += 1
            adj_point.append(list(item))
    return np.array(adj_point)

# get mid points of line
def middle_pts(pts):
    assert(len(pts)==2)
    x = 0.5*(pts[0]+pts[1])
    x = x.astype(int)
    return x

# get the third points of triangle, given two points
def other_pts(pts, triangle):
    assert(len(pts)==2)
    for p1 in triangle:
        flag = 0
        for p2 in pts:
            if list(p1) == list(p2):
                flag = 1
        if flag == 0:
            return p1

# end helper function

def create_scribble(mask): 
    '''
    1. get approximate polygon from mask
    2. trancate polygon into several triangles
    3. connect key points: convex of polygon, mid point of shared sides for two triangles
    '''
    im_list = []
    x = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2BGR)
    # dilate and then erode, trying to get a convex polygon without hole
    # holes lead to extra scribbles
    x = cv2.dilate(x,np.ones((5,5)), iterations = 4)
    x = cv2.erode(x,np.ones((5,5)), iterations = 4)
    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    _, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    ct, _ = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for s in ct:
        bg = np.zeros((mask.shape))
        # ratio 0.07 might need to be tuned
        # higher ratio for more segments/complex lines
        poly = cv2.approxPolyDP(s, 0.07*cv2.arcLength(s, True), True)
        poly = poly.reshape((poly.shape[0], poly.shape[2]))
        # if it's a point, return a thicker point
        if len(poly) == 1:
            x, y = poly[0]
            bg[x-5:x+5,y-5:y+5] == 1
            im_list.append(cv2.dilate(bg,np.ones((3,3)), iterations = 4))
            continue
        # if it's a line, return a thicker line
        elif len(poly) == 2:
            cv2.polylines(bg, [poly], 0, 1)
            im_list.append(cv2.dilate(bg,np.ones((3,3)), iterations = 4))
            continue

        poly_tri = np.array(earclip(poly)) # poly_tri will be a set of triangles
        if len(poly_tri.shape) != 3: # in case something goes wrong, didn't happen yet
            continue
        assert(poly_tri.shape[1] == 3) # three points for triangle
        assert(poly_tri.shape[2] == 2) # x-y for points
        # if only one triangle, choose the longest median
        if len(poly_tri) == 1:
            bg = np.zeros((mask.shape))
            adj_point = poly_tri[0]
            # print('adj', adj_point)
            middle_12 = middle_pts(adj_point[:2])
            # print(middle_12)
            # print(adj_point[2])
            len_12 = np.linalg.norm(middle_12 - adj_point[2])
            middle_23 = middle_pts(adj_point[1:])
            len_23 = np.linalg.norm(middle_23 - adj_point[0])
            middle_13 = middle_pts([adj_point[0], adj_point[2]])
            len_13 = np.linalg.norm(middle_13 - adj_point[1])
            lenall = [len_12, len_13, len_23]
            if len_12 == np.max(lenall):
                apex = np.vstack((middle_12, adj_point[2])).astype(int)
            elif len_23 == np.max(lenall):
                apex = np.vstack((middle_23, adj_point[0])).astype(int)
            else:
                apex = np.vstack((middle_13, adj_point[1])).astype(int)
            cv2.polylines(bg, [apex], 0, 1)
            im_list.append(cv2.dilate(bg,np.ones((3,3)), iterations = 4))
        else:
            for i in poly_tri: # go through each triangles
                bg = np.zeros((mask.shape))
                adj_point= get_adjacent_points(i,poly_tri)
                if len(adj_point)>0:
                    # if only one shared side, connect mid point of this side and the other point, which is an apex of polygon
                    if len(adj_point) == 1:
                        adj_point = adj_point[0]
                        apex = np.vstack((middle_pts(adj_point), other_pts(adj_point, i))).astype(int)
                        cv2.polylines(bg, [apex], 0, 1)
                        im_list.append(cv2.dilate(bg,np.ones((3,3)), iterations = 4))
                    # if one sides are shared, connect their mid points
                    elif len(adj_point) == 2:
                        apex = np.vstack((middle_pts(adj_point[0]), middle_pts(adj_point[1]))).astype(int)
                        cv2.polylines(bg, [apex], 0, 1)
                        im_list.append(cv2.dilate(bg,np.ones((3,3)), iterations = 4))
                    # otherwise, this triangle is in the middle of polygon (three shared sides)
                    # connect all mid pts
                    else:
                        assert(len(adj_point) == 3)
                        m1 = middle_pts(adj_point[0])
                        m2 = middle_pts(adj_point[1])
                        m3 = middle_pts(adj_point[2])
                        m12 = middle_pts([m1,m2])
                        apex = np.vstack((m1,m2))
                        cv2.polylines(bg, [apex], 0, 1)
                        im_list.append(cv2.dilate(bg,np.ones((3,3)), iterations = 4))
                        apex = np.vstack((m12,m3))
                        cv2.polylines(bg, [apex], 0, 1)
                        im_list.append(cv2.dilate(bg,np.ones((3,3)), iterations = 4))
    # get each part together
    if len(im_list)>0:
        t = im_list[0].astype(np.int64).flatten()
        for i in range(1,len(im_list)):
            t = np.bitwise_or(t, im_list[i].astype(np.int64).flatten())
        t = t.reshape((mask.shape)) 
    else:
        t = mask   
    return t