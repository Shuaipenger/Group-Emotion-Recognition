#----------------------------------------------------------------------------
# IMPORTING MODULES
#----------------------------------------------------------------------------

from __future__ import print_function, division

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from PIL import Image

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import pickle

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math
from scipy.special import binom


#---------------------------------------------------------------------------
# IMPORTANT PARAMETERS
#---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else 'cpu'

data_dir = '../Dataset/AlignedCroppedImages/'
root_dir = "../Dataset/"
epochs = 1
batch_size = 32
maxFaces = 15
numClasses = 3
aligned_path = '../TrainedModels/TrainDataset/AlignedModelTrainerSoftmax_AlignedModel_EmotiW_lr01_Softmax-shiyan'

#---------------------------------------------------------------------------
# SPHEREFACE MODEL FOR ALIGNED MODELS
#---------------------------------------------------------------------------

class LSoftmaxLinear(nn.Module):

    def __init__(self, input_dim, output_dim, margin):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.margin = margin

        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))

        self.divisor = math.pi / self.margin
        self.coeffs = binom(margin, range(0, margin + 1, 2))
        self.cos_exps = range(self.margin, -1, -2)
        self.sin_sq_exps = range(len(self.cos_exps))
        self.signs = [1]
        for i in range(1, len(self.sin_sq_exps)):
            self.signs.append(self.signs[-1] * -1)

    def reset_parameters(self):
        nn.init.kaiming_normal(self.weight.data.t())

    def find_k(self, cos):
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, target=None):
        if self.training:
            assert target is not None
            logit = input.matmul(self.weight)
            batch_size = logit.size(0)
            logit_target = logit[range(batch_size), target]
            weight_target_norm = self.weight[:, target].norm(p=2, dim=0)
            input_norm = input.norm(p=2, dim=1)

            # norm_target_prod: (batch_size,)
            norm_target_prod = weight_target_norm * input_norm

            # cos_target: (batch_size,)
            cos_target = logit_target / (norm_target_prod + 1e-10)
            sin_sq_target = 1 - cos_target**2

            num_ns = self.margin//2 + 1
            # coeffs, cos_powers, sin_sq_powers, signs: (num_ns,)
            coeffs = Variable(input.data.new(self.coeffs))
            cos_exps = Variable(input.data.new(self.cos_exps))
            sin_sq_exps = Variable(input.data.new(self.sin_sq_exps))
            signs = Variable(input.data.new(self.signs))

            cos_terms = cos_target.unsqueeze(1) ** cos_exps.unsqueeze(0)
            sin_sq_terms = (sin_sq_target.unsqueeze(1)
                            ** sin_sq_exps.unsqueeze(0))

            cosm_terms = (signs.unsqueeze(0) * coeffs.unsqueeze(0)
                          * cos_terms * sin_sq_terms)
            cosm = cosm_terms.sum(1)
            k = self.find_k(cos_target)

            ls_target = norm_target_prod * (((-1)**k * cosm) - 2*k)
            logit[range(batch_size), target] = ls_target
            return logit
        else:
            assert target is None
            return input.matmul(self.weight)

class sphere20a(nn.Module):
    def __init__(self,classnum=10574,feature=False):
        super(sphere20a, self).__init__()
        self.classnum = classnum
        self.feature = feature
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6,512)
        self.fc6 = LSoftmaxLinear(512,self.classnum, 4)


    def forward(self, x, y):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0),-1)
        x = self.fc5(x)
        if self.feature: return x

        x = self.fc6(x)
        return x

#---------------------------------------------------------------------------
# DATASET AND LOADERS
#---------------------------------------------------------------------------

neg_train = sorted(os.listdir('../Dataset/emotiw/train/'+'Negative/'))
neu_train = sorted(os.listdir('../Dataset/emotiw/train/'+'Neutral/'))
pos_train = sorted(os.listdir('../Dataset/emotiw/train/'+'Positive/'))

train_filelist = neg_train + neu_train + pos_train

val_filelist = []
test_filelist = []

with open('../Dataset/val_list', 'rb') as fp:
    val_filelist = pickle.load(fp)

with open('../Dataset/test_list', 'rb') as fp:
    test_filelist = pickle.load(fp)

for i in train_filelist:
    if i[0] != 'p' and i[0] != 'n':
        train_filelist.remove(i)
        
for i in val_filelist:
    if i[0] != 'p' and i[0] != 'n':
        val_filelist.remove(i)

dataset_sizes = [len(train_filelist), len(val_filelist), len(test_filelist)]
print(dataset_sizes)

train_faces_data_transform = transforms.Compose([
        transforms.Resize((96,112)),
        transforms.ToTensor()
    ])

val_faces_data_transform = transforms.Compose([
        transforms.Resize((96,112)),
        transforms.ToTensor()
    ])

test_faces_data_transform = transforms.Compose([
        transforms.Resize((96,112)),
        transforms.ToTensor()
    ])

class EmotiWDataset(Dataset):
    
    def __init__(self, filelist, root_dir, loadTrain=True, transformGlobal=transforms.ToTensor(), transformFaces = transforms.ToTensor()):
        """
        Args:
            filelist: List of names of image/feature files.
            root_dir: Dataset directory
            transform (callable, optional): Optional transformer to be applied
                                            on an image sample.
        """
        
        self.filelist = filelist
        self.root_dir = root_dir
        self.transformGlobal = transformGlobal
        self.transformFaces = transformFaces
        self.loadTrain = loadTrain
            
    def __len__(self):
        return (len(self.filelist))
    
    def __getitem__(self, idx):
        train = ''
        if self.loadTrain:
            train = 'train'
        else:
            train = 'val'
        filename = self.filelist[idx].split('.')[0]
        labeldict = {'neg':'Negative',
                     'neu':'Neutral',
                     'pos':'Positive',
                     'Negative': 0,
                     'Neutral': 1,
                     'Positive':2}

        labelname = labeldict[filename.split('_')[0]]
        features = np.load(self.root_dir+'FaceFeatures/'+train+'/'+labelname+'/'+filename+'.npz')['a']
        numberFaces = features.shape[0]            
        maxNumber = min(numberFaces, maxFaces)
        
        features1 = np.zeros((maxFaces, 3, 96, 112), dtype = 'float32')
        
        for i in range(maxNumber):
            face = Image.open(self.root_dir + 'AlignedCroppedImages/'+train+'/'+ labelname + '/' + filename+ '_' + str(i) + '.jpg')
                
            if self.transformFaces:
                face = self.transformFaces(face)
                
            features1[i] = face.numpy()
            
        features1 = torch.from_numpy(features1)
        
        sample = {'features': features1, 'label':labeldict[labelname], 'numberFaces': numberFaces}
        
        return sample

train_dataset = EmotiWDataset(train_filelist, root_dir, loadTrain = True, transformFaces = train_faces_data_transform)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=0)

val_dataset = EmotiWDataset(val_filelist, root_dir, loadTrain=False, transformFaces = val_faces_data_transform)

val_dataloader = DataLoader(val_dataset, shuffle =True, batch_size = batch_size, num_workers = 0)

test_dataset = EmotiWDataset(test_filelist, root_dir, loadTrain=False, transformFaces = test_faces_data_transform)

test_dataloader = DataLoader(test_dataset, shuffle =True, batch_size = batch_size, num_workers = 0)

#---------------------------------------------------------------------------
# MODEL DEFINITION
#---------------------------------------------------------------------------

class FaceAttention(nn.Module):
    def __init__(self, align_model):
        super(FaceAttention, self).__init__()

        self.align_model = align_model

    def forward(self, face_features_initial, numberFaces, labels):

        # maxNumber = np.minimum(numberFaces, maxFaces).float()
        numberFaces = numberFaces.data.cpu().numpy()  # 修改
        maxNumber = np.minimum(numberFaces, maxFaces)  # 修改

        face_features = torch.zeros((face_features_initial.shape[0], maxFaces, 3), dtype=torch.float)

        for j in range(face_features_initial.shape[0]):
            face = face_features_initial[j]
            tensor = torch.zeros((2,), dtype=torch.long)
            faceLabels = tensor.new_full((maxFaces,), labels[j], dtype=torch.long)
            faceLabels = faceLabels.to(device)
            face_features[j, :, :] = self.align_model.forward(face, faceLabels)

        face_features = face_features.to(device)
        face_features_sum = torch.zeros((face_features_initial.shape[0], 3), dtype=torch.float)

        face_features_sum = face_features_sum.to(device)

        face_features_avg = torch.zeros((face_features_initial.shape[0], 3), dtype=torch.float)

        face_features_avg = face_features_avg.to(device)

        for i in range(face_features_initial.shape[0]):
            for j in range(int(maxNumber[i])):
                face_features_sum[i] = face_features_sum[i] + face_features[i][j]

            if int(maxNumber[i]) != 0:
                y = float(maxNumber[i])
                face_features_avg[i] = face_features_sum[i] / y

        return face_features_avg

align_model = torch.load(aligned_path, map_location=lambda storage, loc: storage).module

model_ft = align_model
model_ft = model_ft.to(device)
model_ft = nn.DataParallel(model_ft)

output_train_model0 = []
output_valid_model0 = []
output_test_model0 = []

label_train = []
label_valid = []
label_test = []


#---------------------------------------------------------------------------
# TRAINING
#---------------------------------------------------------------------------

def train_model(model, criterion, optimizer=None, scheduler=None, num_epochs = 1):
    
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in range(0, 3):
            if phase == 0:
                dataloaders = train_dataloader
                # scheduler.step()
                model.eval()
            elif phase == 1:
                dataloaders = val_dataloader
                model.eval()
            elif phase == 2:
                dataloaders = test_dataloader
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for i_batch, sample_batched in enumerate(dataloaders):
                labels = sample_batched['label']
                face_features = sample_batched['features']
                numberFaces = sample_batched['numberFaces']
                labels = labels.to(device)
                face_features = face_features.to(device)
                numberFaces = numberFaces.to(device)

                with torch.set_grad_enabled(phase == 0):
                    outputs = model(face_features, numberFaces, labels)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 0:
                        output_train_model0.extend(outputs.data.cpu().numpy())
                        label_train.extend(labels.cpu().numpy())
                    elif phase == 1:
                        output_valid_model0.extend(outputs.data.cpu().numpy())
                        label_valid.extend(labels.cpu().numpy())
                    else:
                        output_test_model0.extend(outputs.data.cpu().numpy())
                        label_test.extend(labels.cpu().numpy())

                running_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            np.savez('model0_output_data',
                     output_train_model0=output_train_model0,
                     output_valid_model0=output_valid_model0,
                     output_test_model0=output_test_model0,
                     label_train=label_train,
                     label_valid=label_valid,
                     label_test=label_test)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 1 and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {: .0f}m {:0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    return model

criterion = nn.CrossEntropyLoss()

model_ft = train_model(model_ft, criterion, None, None, num_epochs=epochs)

np.savez('model0_output_data',
         output_train_model0=output_train_model0,
         output_valid_model0=output_valid_model0,
         output_test_model0=output_test_model0,
         label_train=label_train,
         label_valid=label_valid,
         label_test=label_test)
