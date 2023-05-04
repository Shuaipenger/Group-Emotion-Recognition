from __future__ import print_function, division
import warnings

warnings.filterwarnings("ignore")
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from PIL import Image
import numpy as np
from torchvision import datasets, models, transforms
import os
import copy
import pickle
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum

# ---------------------------------------------------------------------------
# IMPORTANT PARAMETERS
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else 'cpu'
root_dir = "../Dataset/"
epochs = 1
batch_size = 4
random_seed = 8

# ---------------------------------------------------------------------------
# DATASET AND LOADERS
# ---------------------------------------------------------------------------

neg_train = sorted(os.listdir('../Dataset/wzq_dataset7:2:1/train_skeleton/'+'Negative/'))
pos_train = sorted(os.listdir('../Dataset/wzq_dataset7:2:1/train_skeleton/' + 'Positive/'))
train_filelist = neg_train  + pos_train

neg_val = sorted(os.listdir('../Dataset/wzq_dataset7:2:1/val_skeleton/'+'Negative/'))
pos_val = sorted(os.listdir('../Dataset/wzq_dataset7:2:1/val_skeleton/'+'Positive/'))

val_filelist = neg_val  + pos_val

neg_test = sorted(os.listdir('../Dataset/wzq_dataset7:2:1/test_skeleton/'+'Negative/'))
pos_test = sorted(os.listdir('../Dataset/wzq_dataset7:2:1/test_skeleton/'+'Positive/'))

test_filelist = neg_test  + pos_test
dataset_sizes = [len(train_filelist), len(val_filelist), len(test_filelist)]
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

    def __init__(self, filelist, root_dir, loadTrain=True, transformGlobal=transforms.ToTensor(),
                 transformFaces=transforms.ToTensor(), loadTest = False):
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
        if self.loadTrain:
            return (len(train_filelist))
        elif  self.loadTest:
            return (len(test_filelist))
        else:
            return (len(val_filelist))

    def __getitem__(self, idx):
        train = ''
        if self.loadTrain:
            train = 'train_skeleton'
        elif  self.loadTest:
            train = 'test_skeleton'
        else:
            train = 'val_skeleton'

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

        # IMAGE

        image = Image.open(self.root_dir + 'wzq_dataset7:2:1/' + train + '/' + labelname + '/' + filename + '.png')
        image = image.convert('RGB')

        if self.transformGlobal:
            image = self.transformGlobal(image)
        if image.shape[0] == 1:
            image_1 = np.zeros((3, 224, 224), dtype=float)
            image_1[0] = image
            image_1[1] = image
            image_1[2] = image
            image = image_1
            image = torch.FloatTensor(image.tolist())

            # SAMPLE
        sample = {'image': image, 'label': labeldict[labelname]}
        return sample


train_dataset = wzq_Dataset(train_filelist, root_dir, loadTrain=True, transformGlobal=train_global_data_transform,loadTest = False)
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, num_workers=0)

val_dataset = wzq_Dataset(val_filelist, root_dir, loadTrain=False, transformGlobal=val_global_data_transform,loadTest = False)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=0)

test_dataset = wzq_Dataset(test_filelist, root_dir, loadTrain=False, transformGlobal=test_global_data_transform,loadTest = True)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=0)
print('Dataset Loaded')

model_ft = torch.load('../TrainedModels/TrainDataset_my_models_wzq/model_skeleton_EfficientNet_New_data.pt', map_location=lambda storage, loc: storage)

print('Skeleton Efficient Loaded! (Model 4)')

model_ft = model_ft.to(device)
model_ft = nn.DataParallel(model_ft)

output_train_model1 = []
output_valid_model1 = []
output_test_model1 = []

label_train = []
label_valid = []
label_test = []


def train_model(model, criterion, optimizer=None, scheduler=None, num_epochs=1):
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
                inputs = sample_batched['image']
                labels = sample_batched['label']

                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 0):
                    output1 = model(inputs)

                    if phase == 0:
                        output_train_model1.extend(output1.data.cpu().numpy())
                        label_train.extend(labels.cpu().numpy())
                    elif phase == 1:
                        output_valid_model1.extend(output1.data.cpu().numpy())
                        label_valid.extend(labels.cpu().numpy())
                    else:
                        output_test_model1.extend(output1.data.cpu().numpy())
                        label_test.extend(labels.cpu().numpy())

                    _, preds = torch.max(output1, 1)
                    loss = criterion(output1, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            np.savez('model4_Skeleton_Efficient_output_data',
                     output_train_model1=output_train_model1,
                     output_valid_model1=output_valid_model1,
                     output_test_model1=output_test_model1,
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

model = train_model(model_ft, criterion, None, None, num_epochs=epochs)

np.savez('model4_Skeleton_Efficient_output_data',
         output_train_model1=output_train_model1,
         output_valid_model1=output_valid_model1,
         output_test_model1=output_test_model1,
         label_train=label_train,
         label_valid=label_valid,
         label_test=label_test)