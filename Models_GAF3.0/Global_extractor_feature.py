###############################
#用desnet161提取global的特征向量，保存成npy，供后续预训练crossvit
###############################


import warnings

warnings.filterwarnings('ignore')

# from __future__ import print_function, division
from PIL import Image


from torchvision import transforms, datasets
import numpy as np
import os
import numpy as np
import torch
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import matplotlib.pyplot as plt

from PIL import Image

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import pickle


from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# %% md

# Path Definitions

# %%

dataset_path = '../Dataset/emotiw/'

processed_dataset_path = '../Dataset/GlobalFeatures/'

# %% md

# desnet161 Model Definition for Extracting Face Features

def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.num_batches_tracked = True
    else:
        for name, module11 in module._modules.items():
            module11= recursion_change_bn(module11)

def recursion_change_bn2(module):
    if isinstance(module, torchvision.models.densenet._DenseLayer):
        module.memory_efficient = True
    else:
        for name, module22 in module._modules.items():
            module22 = recursion_change_bn2(module22)

def recursion_change_bn3(module):
    if isinstance(module, torch.nn.modules.pooling.AvgPool2d):
        module.divisor_override = None
    else:
        for name, module33 in module._modules.items():
            module33 = recursion_change_bn3(module33)
##对导入的模型参数进行处理，来迎合版本变化
global_model = torch.load('../TrainedModels/TrainDataset/DenseNet161_EmotiW', map_location=lambda storage, loc: storage).module.features
##对导入的模型参数进行处理，来迎合版本变化
for name, module in global_model._modules.items():
    recursion_change_bn(module)

for name, module in global_model._modules.items():
    recursion_change_bn2(module)

for name, module in global_model._modules.items():
    recursion_change_bn3(module)


# %% md

# Load Train and Val Dataset

# %%

image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_path, x)) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%

training_dataset = image_datasets['train']
validation_dataset = image_datasets['val']

# %%

neg_train = sorted(os.listdir(dataset_path + 'train/Negative/'))
neu_train = sorted(os.listdir(dataset_path + 'train/Neutral/'))
pos_train = sorted(os.listdir(dataset_path + 'train/Positive/'))

neg_val = sorted(os.listdir(dataset_path + 'val/Negative/'))
neu_val = sorted(os.listdir(dataset_path + 'val/Neutral/'))
pos_val = sorted(os.listdir(dataset_path + 'val/Positive/'))

# %%

neg_train_filelist = [x.split('.')[0] for x in neg_train]
neu_train_filelist = [x.split('.')[0] for x in neu_train]
pos_train_filelist = [x.split('.')[0] for x in pos_train]

neg_val_filelist = [x.split('.')[0] for x in neg_val]
neu_val_filelist = [x.split('.')[0] for x in neu_val]
pos_val_filelist = [x.split('.')[0] for x in pos_val]


print(neg_train_filelist[:10])
print(neu_train_filelist[:10])
print(pos_train_filelist[:10])

print(neg_val_filelist[:10])
print(neu_val_filelist[:10])
print(pos_val_filelist[:10])

# %%

train_filelist = neg_train_filelist + neu_train_filelist + pos_train_filelist
val_filelist = neg_val_filelist + neu_val_filelist + pos_val_filelist

# %%

print(len(training_dataset))
print(len(validation_dataset))
train_global_data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_global_data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# %% md

# Extract global Features

# %%

for i in range(len(training_dataset)):
    image, label = training_dataset[i]

    image = image.convert('RGB')  ##用opencv或者是PIL包下面的图形处理函数，把输入的图片从灰度图转为RGB空间的彩色图。这种方法可以适合数据集中既包含有RGB图片又含有灰度图的情况    后加的

    if train_global_data_transform:
        image = train_global_data_transform(image)

    if image.shape[0] == 1:
        image_1 = np.zeros((3, 224, 224), dtype=float)
        image_1[0] = image
        image_1[1] = image
        image_1[2] = image
        image = image_1
        image = torch.FloatTensor(image.tolist())

    image = image.view(1, 3, 224,224) # 1 3 224 224

    print(train_filelist[i])

    if label == 0:
        # if os.path.isfile(processed_dataset_path + 'train/Negative/' + train_filelist[i] + '.npz'):
        #     print(train_filelist[i] + ' Already present')
        #     continue
        features = global_model.forward(image)
        out = F.relu(features, inplace=False)  # out: 32 2208 7 7
        global_features_initial = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0),-1)  # global_features_initial: 32 2208
        global_features_initial = Variable(global_features_initial)
        global_features_initial = global_features_initial.view(-1, 2208)  # global_features_initial: 32 2208

        connected_layer = nn.Linear(2208, 256)
        Dropout = nn.Dropout(p = 0.5)

        global_features = Dropout(connected_layer(global_features_initial))  # global_features:32 256
        global_features = global_features.view(-1, 1, 256)                   # 32 1 256

        global_features = global_features.data.numpy()

        np.savez(processed_dataset_path + 'train/Negative/' + train_filelist[i], a=global_features)

    elif label == 1:
        if os.path.isfile(processed_dataset_path + 'train/Neutral/' + train_filelist[i] + '.npz'):
            print(train_filelist[i] + ' Already present')
            continue
        features = global_model.forward(image)
        out = F.relu(features, inplace=False)  # out: 32 2208 7 7
        global_features_initial = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)  # global_features_initial: 32 2208
        global_features_initial = Variable(global_features_initial)
        global_features_initial = global_features_initial.view(-1, 2208)  # global_features_initial: 32 2208

        connected_layer = nn.Linear(2208, 256)
        Dropout = nn.Dropout(p=0.5)

        global_features = Dropout(connected_layer(global_features_initial))  # global_features:32 256
        global_features = global_features.view(-1, 1, 256)

        global_features = global_features.data.numpy()

        np.savez(processed_dataset_path + 'train/Neutral/' + train_filelist[i], a=global_features)

    else:
        if os.path.isfile(processed_dataset_path + 'train/Positive/' + train_filelist[i] + '.npz'):
            print(train_filelist[i] + ' Already present')
            continue
        features = global_model.forward(image)
        out = F.relu(features, inplace=False)  # out: 32 2208 7 7
        global_features_initial = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)  # global_features_initial: 32 2208
        global_features_initial = Variable(global_features_initial)
        global_features_initial = global_features_initial.view(-1, 2208)  # global_features_initial: 32 2208

        connected_layer = nn.Linear(2208, 256)
        Dropout = nn.Dropout(p=0.5)

        global_features = Dropout(connected_layer(global_features_initial))  # global_features:32 256
        global_features = global_features.view(-1, 1, 256)

        global_features = global_features.data.numpy()
        np.savez(processed_dataset_path + 'train/Positive/' + train_filelist[i], a=global_features)

# for i in range(len(validation_dataset)):
#     image, label = validation_dataset[i]
#
#     image = image.convert('RGB')  ##用opencv或者是PIL包下面的图形处理函数，把输入的图片从灰度图转为RGB空间的彩色图。这种方法可以适合数据集中既包含有RGB图片又含有灰度图的情况    后加的
#
#     if val_global_data_transform:
#         image = val_global_data_transform(image)
#
#     if image.shape[0] == 1:
#         image_1 = np.zeros((3, 224, 224), dtype=float)
#         image_1[0] = image
#         image_1[1] = image
#         image_1[2] = image
#         image = image_1
#         image = torch.FloatTensor(image.tolist())
#
#     image = image.view(1, 3, 224,224)
#
#     print(val_filelist[i])
#
#     if label == 0:
#         if os.path.isfile(processed_dataset_path + 'val/Negative/' + val_filelist[i] + '.npz'):
#             print(val_filelist[i] + ' Already present')
#             continue
#         features = global_model.forward(image)
#         out = F.relu(features, inplace=False)  # out: 32 2208 7 7
#         global_features_initial = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0),-1)  # global_features_initial: 32 2208
#         global_features_initial = Variable(global_features_initial)
#         global_features_initial = global_features_initial.view(-1, 2208)  # global_features_initial: 32 2208
#
#         connected_layer = nn.Linear(2208, 256)
#         Dropout = nn.Dropout(p = 0.5)
#
#         global_features = Dropout(connected_layer(global_features_initial))  # global_features:32 256
#         global_features = global_features.view(-1, 1, 256)                   # 32 1 256
#
#         global_features = global_features.data.numpy()
#
#         np.savez(processed_dataset_path + 'val/Negative/' + val_filelist[i], a=global_features)
#
#     elif label == 1:
#         if os.path.isfile(processed_dataset_path + 'val/Neutral/' + val_filelist[i] + '.npz'):
#             print(val_filelist[i] + ' Already present')
#             continue
#         features = global_model.forward(image)
#         out = F.relu(features, inplace=False)  # out: 32 2208 7 7
#         global_features_initial = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)  # global_features_initial: 32 2208
#         global_features_initial = Variable(global_features_initial)
#         global_features_initial = global_features_initial.view(-1, 2208)  # global_features_initial: 32 2208
#
#         connected_layer = nn.Linear(2208, 256)
#         Dropout = nn.Dropout(p=0.5)
#
#         global_features = Dropout(connected_layer(global_features_initial))  # global_features:32 256
#         global_features = global_features.view(-1, 1, 256)
#
#         global_features = global_features.data.numpy()
#
#         np.savez(processed_dataset_path + 'val/Neutral/' + val_filelist[i], a=global_features)
#
#     else:
#         if os.path.isfile(processed_dataset_path + 'val/Positive/' + val_filelist[i] + '.npz'):
#             print(val_filelist[i] + ' Already present')
#             continue
#         features = global_model.forward(image)
#         out = F.relu(features, inplace=False)  # out: 32 2208 7 7
#         global_features_initial = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)  # global_features_initial: 32 2208
#         global_features_initial = Variable(global_features_initial)
#         global_features_initial = global_features_initial.view(-1, 2208)  # global_features_initial: 32 2208
#
#         connected_layer = nn.Linear(2208, 256)
#         Dropout = nn.Dropout(p=0.5)
#
#         global_features = Dropout(connected_layer(global_features_initial))  # global_features:32 256
#         global_features = global_features.view(-1, 1, 256)
#
#         global_features = global_features.data.numpy()
#         np.savez(processed_dataset_path + 'val/Positive/' + val_filelist[i], a=global_features)


