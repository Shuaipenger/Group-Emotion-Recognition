#----------------------------------------------------------------------------
# IMPORTING MODULES
#----------------------------------------------------------------------------

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image

from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import time
import os
import copy
import pickle

#---------------------------------------------------------------------------
# IMPORTANT PARAMETERS
#---------------------------------------------------------------------------

root_dir = "../Dataset/"
# data_dir = '../Dataset/wzq_dataset/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 1
batch_size = 8

#---------------------------------------------------------------------------
# DATASET AND LOADERS
#---------------------------------------------------------------------------

neg_train = sorted(os.listdir('../Dataset/wzq_dataset7:2:1/train/' + 'Negative/'))
pos_train = sorted(os.listdir('../Dataset/wzq_dataset7:2:1/train/' + 'Positive/'))
train_filelist = neg_train  + pos_train
neg_val = sorted(os.listdir('../Dataset/wzq_dataset7:2:1/val/' + 'Negative/'))
pos_val = sorted(os.listdir('../Dataset/wzq_dataset7:2:1/val/' + 'Positive/'))
val_filelist = neg_val  + pos_val
neg_test = sorted(os.listdir('../Dataset/wzq_dataset7:2:1/test/' + 'Negative/'))
pos_test = sorted(os.listdir('../Dataset/wzq_dataset7:2:1/test/' + 'Positive/'))
test_filelist = neg_test  + pos_test


neg_trainF = sorted(os.listdir('../Dataset/wzq_dataset7:2:1/CroppedFaces2/train/' + 'Negative/'))
pos_trainF = sorted(os.listdir('../Dataset/wzq_dataset7:2:1/CroppedFaces2/train/' + 'Positive/'))
train_filelistF = neg_trainF  + pos_trainF
neg_valF = sorted(os.listdir('../Dataset/wzq_dataset7:2:1/CroppedFaces2/val/' + 'Negative/'))
pos_valF = sorted(os.listdir('../Dataset/wzq_dataset7:2:1/CroppedFaces2/val/' + 'Positive/'))
val_filelistF = neg_valF  + pos_valF
neg_testF = sorted(os.listdir('../Dataset/wzq_dataset7:2:1/CroppedFaces2/test/' + 'Negative/'))
pos_testF = sorted(os.listdir('../Dataset/wzq_dataset7:2:1/CroppedFaces2/test/' + 'Positive/'))
test_filelistF = neg_testF  + pos_testF

def npz2jpg(list):
    n = len(list)
    for i in range(0,n):
        list[i]= list[i].split('.')[0] + '.jpg'
    return list

npz2jpg(train_filelistF)
npz2jpg(val_filelistF)
npz2jpg(test_filelistF)

##删除list中相同的元素
train_filelist = [ i for i in train_filelist if i not in train_filelistF]
val_filelist = [i for i in val_filelist if i not in val_filelistF]
test_filelist = [i for i in test_filelist if i not in test_filelistF]

dataset_sizes = [len(train_filelist), len(val_filelist),len(test_filelist)]
print(dataset_sizes)

train_global_data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_global_data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
test_global_data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class wzq_Dataset(Dataset):
    
    def __init__(self, filelist, root_dir, loadTrain=True, transformGlobal=transforms.ToTensor(), transformFaces = transforms.ToTensor(), loadTest = False):
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
        self.loadTest = loadTest
            
    def __len__(self):
        return (len(self.filelist)) 
 
    def __getitem__(self, idx):
        train = ''
        if self.loadTrain:
            train = 'train'
        elif  self.loadTest:
            train = 'test'
        else:
            train = 'val'
        
        filename = self.filelist[idx].split('.')[0]
        labeldict = {'neg': 'Negative',
                     'Vneg': 'Negative',
                     'Tneg': 'Negative',
                     'pos': 'Positive',
                     'Vpos': 'Positive',
                     'Tpos': 'Positive',
                     'Negative': 0,
                     'Positive': 1}

        labelname = labeldict[filename.split('_')[0]]

        #IMAGE 

        image = Image.open(self.root_dir+'wzq_dataset7:2:1/'+train+'/'+labelname+'/'+filename+'.jpg')
        image = image.convert('RGB')

        if self.transformGlobal:
            image = self.transformGlobal(image)
        if image.shape[0] == 1:
            image_1 = np.zeros((3, 224, 224), dtype = float)
            image_1[0] = image
            image_1[1] = image
            image_1[2] = image
            image = image_1
            image = torch.FloatTensor(image.tolist()) 
        
        #SAMPLE
        sample = {'image': image, 'label':labeldict[labelname]}
        return sample


train_dataset = wzq_Dataset(train_filelist, root_dir, loadTrain = True, transformGlobal=train_global_data_transform,loadTest = False)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=0)

val_dataset = wzq_Dataset(val_filelist, root_dir, loadTrain=False, transformGlobal= val_global_data_transform,loadTest = False)
val_dataloader = DataLoader(val_dataset, shuffle =True, batch_size = batch_size, num_workers = 0)

test_dataset = wzq_Dataset(test_filelist, root_dir, loadTrain=False, transformGlobal= test_global_data_transform,loadTest = True)
test_dataloader = DataLoader(test_dataset, shuffle =True, batch_size = batch_size, num_workers = 0)
print('Dataset Loaded')

#---------------------------------------------------------------------------
# MODEL DEFINITION
#---------------------------------------------------------------------------


model_ft = torch.load('../TrainedModels/TrainDataset_my_models_wzq/model_1_1_densenet_New_data_NoFace.pt', map_location=lambda storage, loc: storage)
print('Densenet161 model_No_Face Loaded! ')

model_ft = model_ft.to(device)
model_ft = nn.DataParallel(model_ft)

output_train_model = []
output_valid_model = []
output_test_model = []

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
        
        for phase in range(0,3):
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
                inputs = sample_batched['image']
                labels = sample_batched['label']

                inputs = inputs.to(device)
                labels = labels.to(device)

                
                with torch.set_grad_enabled(phase == 0):
                    output1 = model(inputs)
                    _, preds = torch.max(output1, 1)
                    loss = criterion(output1, labels)

                    if phase == 0:
                        output_train_model.extend(output1.data.cpu().numpy())
                        label_train.extend(labels.cpu().numpy())
                    elif phase == 1:
                        output_valid_model.extend(output1.data.cpu().numpy())
                        label_valid.extend(labels.cpu().numpy())
                    else:
                        output_test_model.extend(output1.data.cpu().numpy())
                        label_test.extend(labels.cpu().numpy())


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            np.savez('model_output_data_Densenet_No_Face',
                     output_train_model=output_train_model,
                     output_valid_model=output_valid_model,
                     output_test_model=output_test_model,
                     label_train=label_train,
                     label_valid=label_valid,
                     label_test=label_test)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        print()
    time_elapsed = time.time() - since
    print('Training complete in {: .0f}m {:0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    return model

criterion = nn.CrossEntropyLoss()

model_ft = train_model(model_ft, criterion, None, None, num_epochs=epochs)

np.savez('model_output_data_Densenet_No_Face',
         output_train_model=output_train_model,
         output_valid_model=output_valid_model,
         output_test_model=output_test_model,
         label_train=label_train,
         label_valid=label_valid,
         label_test=label_test)