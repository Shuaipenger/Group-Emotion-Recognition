###############################
#用desnet161提取global的特征向量，保存成npz，供后续预训练crossvit
###############################


import warnings

from torchvision.models import DenseNet

warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
from torch import nn, einsum

# Path Definitions

dataset_path = '/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/'

processed_dataset_path = '/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/GlobalFeatures/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
###############################################################################
# desnet161 Model Definition for Extracting Face Features

global_model = models.densenet161(pretrained=False)
num_ftrs = global_model.classifier.in_features
global_model.classifier = nn.Linear(num_ftrs, 2)

state_dict = torch.load('/home/prmi/GSP/Group-Level-Emotion-Recognition-master/TrainedModels/TrainDataset_my_models_wzq/model_1_1_densenet_emotiw_lr001.pt', map_location=lambda storage, loc: storage)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

global_model.load_state_dict(new_state_dict)
global_model = global_model.to(device)
global_model = nn.DataParallel(global_model)

# %% md

# Load Train and Val Dataset

# %%

image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_path, x)) for x in ['train', 'val','test']}

class_names = image_datasets['train'].classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%

training_dataset = image_datasets['train']
validation_dataset = image_datasets['val']
test_dataset = image_datasets['test']

# %%

neg_train = sorted(os.listdir(dataset_path + 'train/Negative/'))
pos_train = sorted(os.listdir(dataset_path + 'train/Positive/'))

neg_val = sorted(os.listdir(dataset_path + 'val/Negative/'))
pos_val = sorted(os.listdir(dataset_path + 'val/Positive/'))

neg_test = sorted(os.listdir(dataset_path + 'test/Negative/'))
pos_test = sorted(os.listdir(dataset_path + 'test/Positive/'))

# %%

neg_train_filelist = [x.split('.')[0] for x in neg_train]
pos_train_filelist = [x.split('.')[0] for x in pos_train]

neg_val_filelist = [x.split('.')[0] for x in neg_val]
pos_val_filelist = [x.split('.')[0] for x in pos_val]

neg_test_filelist = [x.split('.')[0] for x in neg_test]
pos_test_filelist = [x.split('.')[0] for x in pos_test]


print(neg_train_filelist[:10])
print(pos_train_filelist[:10])

print(neg_val_filelist[:10])
print(pos_val_filelist[:10])

print(neg_test_filelist[:10])
print(pos_test_filelist[:10])

# %%

train_filelist = neg_train_filelist  + pos_train_filelist
val_filelist = neg_val_filelist  + pos_val_filelist
test_filelist = neg_test_filelist  + pos_test_filelist

# %%

print(len(training_dataset))
print(len(validation_dataset))
print(len(test_dataset))
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

test_global_data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# %% md

# Extract global Features

# %%

# for i in range(len(training_dataset)):
#     image, label = training_dataset[i]
#
#     image = image.convert('RGB')  ##用opencv或者是PIL包下面的图形处理函数，把输入的图片从灰度图转为RGB空间的彩色图。这种方法可以适合数据集中既包含有RGB图片又含有灰度图的情况    后加的
#
#     if train_global_data_transform:
#         image = train_global_data_transform(image)
#
#     if image.shape[0] == 1:
#         image_1 = np.zeros((3, 224, 224), dtype=float)
#         image_1[0] = image
#         image_1[1] = image
#         image_1[2] = image
#         image = image_1
#         image = torch.FloatTensor(image.tolist())
#
#     image = image.view(1, 3, 224,224)    #image: tensor 1 3 224 224
#     image = image.to(device)
#
#     print(train_filelist[i])
#
#     if label == 0:
#         if os.path.isfile(processed_dataset_path + 'train/Negative/' + train_filelist[i] + '.npz'):
#             print(train_filelist[i] + ' Already present')
#             continue
#         features = global_model.module.features.forward(image)
#         out = F.relu(features, inplace=False)  # out: 32 2208 7 7
#         global_features_initial = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0),-1)  # global_features_initial: 32 2208
#         global_features_initial = Variable(global_features_initial)
#         global_features_initial = global_features_initial.view(-1, 2208)  # global_features_initial: 32 2208
#
#         connected_layer = nn.Linear(2208, 256)
#         Dropout = nn.Dropout(p = 0.5)
#         connected_layer = connected_layer.to(device)
#         Dropout = Dropout.to(device)
#
#         global_features = Dropout(connected_layer(global_features_initial))  # global_features:32 256
#         global_features = global_features.view(-1, 1, 256)                   # 32 1 256
#
#         global_features = global_features.cpu().data.numpy()
#
#         np.savez(processed_dataset_path + 'train/Negative/' + train_filelist[i], a=global_features)
#
#     else:
#         if os.path.isfile(processed_dataset_path + 'train/Positive/' + train_filelist[i] + '.npz'):
#             print(train_filelist[i] + ' Already present')
#             continue
#         features = global_model.module.features.forward(image)
#         out = F.relu(features, inplace=False)  # out: 32 2208 7 7
#         global_features_initial = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)  # global_features_initial: 32 2208
#         global_features_initial = Variable(global_features_initial)
#         global_features_initial = global_features_initial.view(-1, 2208)  # global_features_initial: 32 2208
#
#         connected_layer = nn.Linear(2208, 256)
#         Dropout = nn.Dropout(p=0.5)
#         connected_layer = connected_layer.to(device)
#         Dropout = Dropout.to(device)
#
#         global_features = Dropout(connected_layer(global_features_initial))  # global_features:32 256
#         global_features = global_features.view(-1, 1, 256)
#
#         global_features = global_features.cpu().data.numpy()
#         np.savez(processed_dataset_path + 'train/Positive/' + train_filelist[i], a=global_features)
#
# for i in range(len(validation_dataset)):
#     image, label = validation_dataset[i]
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
#     image = image.to(device)
#
#     print(val_filelist[i])
#
#     if label == 0:
#         if os.path.isfile(processed_dataset_path + 'val/Negative/' + val_filelist[i] + '.npz'):
#             print(val_filelist[i] + ' Already present')
#             continue
#         features = global_model.module.features.forward(image)
#         out = F.relu(features, inplace=False)  # out: 32 2208 7 7
#         global_features_initial = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0),-1)  # global_features_initial: 32 2208
#         global_features_initial = Variable(global_features_initial)
#         global_features_initial = global_features_initial.view(-1, 2208)  # global_features_initial: 32 2208
#
#         connected_layer = nn.Linear(2208, 256)
#         Dropout = nn.Dropout(p = 0.5)
#         connected_layer = connected_layer.to(device)
#         Dropout = Dropout.to(device)
#
#         global_features = Dropout(connected_layer(global_features_initial))  # global_features:32 256
#         global_features = global_features.view(-1, 1, 256)                   # 32 1 256
#
#         global_features = global_features.cpu().data.numpy()
#
#         np.savez(processed_dataset_path + 'val/Negative/' + val_filelist[i], a=global_features)
#
#     else:
#         if os.path.isfile(processed_dataset_path + 'val/Positive/' + val_filelist[i] + '.npz'):
#             print(val_filelist[i] + ' Already present')
#             continue
#         features = global_model.module.features.forward(image)
#         out = F.relu(features, inplace=False)  # out: 32 2208 7 7
#         global_features_initial = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)  # global_features_initial: 32 2208
#         global_features_initial = Variable(global_features_initial)
#         global_features_initial = global_features_initial.view(-1, 2208)  # global_features_initial: 32 2208
#
#         connected_layer = nn.Linear(2208, 256)
#         Dropout = nn.Dropout(p=0.5)
#         connected_layer = connected_layer.to(device)
#         Dropout = Dropout.to(device)
#
#         global_features = Dropout(connected_layer(global_features_initial))  # global_features:32 256
#         global_features = global_features.view(-1, 1, 256)
#
#         global_features = global_features.cpu().data.numpy()
#         np.savez(processed_dataset_path + 'val/Positive/' + val_filelist[i], a=global_features)

for i in range(len(test_dataset)):
    image, label = test_dataset[i]
    image = image.convert('RGB')  ##用opencv或者是PIL包下面的图形处理函数，把输入的图片从灰度图转为RGB空间的彩色图。这种方法可以适合数据集中既包含有RGB图片又含有灰度图的情况    后加的

    if test_global_data_transform:
        image = test_global_data_transform(image)

    if image.shape[0] == 1:
        image_1 = np.zeros((3, 224, 224), dtype=float)
        image_1[0] = image
        image_1[1] = image
        image_1[2] = image
        image = image_1
        image = torch.FloatTensor(image.tolist())

    image = image.view(1, 3, 224,224)
    image = image.to(device)

    print(test_filelist[i])

    if label == 0:
        if os.path.isfile(processed_dataset_path + 'test/Negative/' + test_filelist[i] + '.npz'):
            print(test_filelist[i] + ' Already present')
            continue
        features = global_model.module.features.forward(image)
        out = F.relu(features, inplace=False)  # out: 32 2208 7 7
        global_features_initial = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0),-1)  # global_features_initial: 32 2208
        global_features_initial = Variable(global_features_initial)
        global_features_initial = global_features_initial.view(-1, 2208)  # global_features_initial: 32 2208

        connected_layer = nn.Linear(2208, 256)
        Dropout = nn.Dropout(p = 0.5)
        connected_layer = connected_layer.to(device)
        Dropout = Dropout.to(device)

        global_features = Dropout(connected_layer(global_features_initial))  # global_features:32 256
        global_features = global_features.view(-1, 1, 256)                   # 32 1 256

        global_features = global_features.cpu().data.numpy()

        np.savez(processed_dataset_path + 'test/Negative/' + test_filelist[i], a=global_features)

    else:
        if os.path.isfile(processed_dataset_path + 'test/Positive/' + test_filelist[i] + '.npz'):
            print(test_filelist[i] + ' Already present')
            continue
        features = global_model.module.features.forward(image)
        out = F.relu(features, inplace=False)  # out: 32 2208 7 7
        global_features_initial = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)  # global_features_initial: 32 2208
        global_features_initial = Variable(global_features_initial)
        global_features_initial = global_features_initial.view(-1, 2208)  # global_features_initial: 32 2208

        connected_layer = nn.Linear(2208, 256)
        Dropout = nn.Dropout(p=0.5)
        connected_layer = connected_layer.to(device)
        Dropout = Dropout.to(device)

        global_features = Dropout(connected_layer(global_features_initial))  # global_features:32 256
        global_features = global_features.view(-1, 1, 256)

        global_features = global_features.cpu().data.numpy()
        np.savez(processed_dataset_path + 'test/Positive/' + test_filelist[i], a=global_features)




