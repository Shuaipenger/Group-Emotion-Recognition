# %% md

# Import Modules

# %%

import warnings

warnings.filterwarnings('ignore')

# %%

from src import detect_faces, show_bboxes
from PIL import Image

import torch
from torchvision import transforms, datasets
import numpy as np
import os

# %% md

# Path Definitions

# %%

dataset_path = '../Dataset/wzq_dataset/'

processed_dataset_path = '../Dataset/wzq_dataset/FaceCoordinates/'

# %% md

# Load Train and Val Dataset

# %%

image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_path, x))
                  for x in ['train', 'val']}

class_names = image_datasets['train'].classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%

print(class_names)

# %%

training_dataset = image_datasets['train']
validation_dataset = image_datasets['val']

# %%

neg_train = sorted(os.listdir(dataset_path + 'train/Negative/'))
pos_train = sorted(os.listdir(dataset_path + 'train/Positive/'))

neg_val = sorted(os.listdir(dataset_path + 'val/Negative/'))
pos_val = sorted(os.listdir(dataset_path + 'val/Positive/'))

# %%

neg_train_filelist = [x.split('.')[0] for x in neg_train]
pos_train_filelist = [x.split('.')[0] for x in pos_train]

neg_val_filelist = [x.split('.')[0] for x in neg_val]
pos_val_filelist = [x.split('.')[0] for x in pos_val]

# %%

print(neg_train_filelist[:10])
print(pos_train_filelist[:10])

print(neg_val_filelist[:10])
print(pos_val_filelist[:10])

# %%

train_filelist = neg_train_filelist  + pos_train_filelist
val_filelist = neg_val_filelist  + pos_val_filelist

# %%

print(len(training_dataset))
print(len(validation_dataset))

# %% md

# Extract Faces from Image using MTCNN

# %%

for i in range(len(training_dataset)):
    image, label = training_dataset[i]
    print(train_filelist[i])
    try:
        if label == 0:
            if os.path.isfile(processed_dataset_path + 'train/Negative/' + train_filelist[i] + '.npz'):
                print(train_filelist[i] + ' Already present')
                continue
            bounding_boxes, landmarks = detect_faces(image)
            bounding_boxes = np.asarray(bounding_boxes)
            if bounding_boxes.size == 0:
                print('MTCNN model handling empty face condition at ' + train_filelist[i])
            np.savez(processed_dataset_path + 'train/Negative/' + train_filelist[i], a=bounding_boxes, b=landmarks)

        else:

            if os.path.isfile(processed_dataset_path + 'train/Positive/' + train_filelist[i] + '.npz'):
                print(train_filelist[i] + ' Already present')
                continue
            bounding_boxes, landmarks = detect_faces(image)
            bounding_boxes = np.asarray(bounding_boxes)
            if bounding_boxes.size == 0:
                print('MTCNN model handling empty face condition at ' + train_filelist[i])
            np.savez(processed_dataset_path + 'train/Positive/' + train_filelist[i], a=bounding_boxes, b=landmarks)

    except ValueError:
        print('No faces detected for ' + train_filelist[i] + ". Also MTCNN failed.")
        if label == 0:
            np.savez(processed_dataset_path + 'train/Negative/' + train_filelist[i], a=np.zeros(1), b=np.zeros(1))
        elif label == 1:
            np.savez(processed_dataset_path + 'train/Positive/' + train_filelist[i], a=np.zeros(1), b=np.zeros(1))
        continue

# %%

for i in range(len(validation_dataset)):
    image, label = validation_dataset[i]
    print(val_filelist[i])
    try:
        if label == 0:
            if os.path.isfile(processed_dataset_path + 'val/Negative/' + val_filelist[i] + '.npz'):
                print(val_filelist[i] + ' Already present')
                continue
            bounding_boxes, landmarks = detect_faces(image)
            bounding_boxes = np.asarray(bounding_boxes)
            if bounding_boxes.size == 0:
                print('MTCNN model handling empty face condition at ' + val_filelist[i])
            np.savez(processed_dataset_path + 'val/Negative/' + val_filelist[i], a=bounding_boxes, b=landmarks)

        else:
            if os.path.isfile(processed_dataset_path + 'val/Positive/' + val_filelist[i] + '.npz'):
                print(val_filelist[i] + ' Already present')
                continue
            bounding_boxes, landmarks = detect_faces(image)
            bounding_boxes = np.asarray(bounding_boxes)
            if bounding_boxes.size == 0:
                print('MTCNN model handling empty face condition at ' + val_filelist[i])
            np.savez(processed_dataset_path + 'val/Positive/' + val_filelist[i], a=bounding_boxes, b=landmarks)

    except ValueError:
        print('No faces detected for ' + val_filelist[i] + ". Also MTCNN failed.")
        if label == 0:
            np.savez(processed_dataset_path + 'val/Negative/' + val_filelist[i], a=np.zeros(1), b=np.zeros(1))
        else:
            np.savez(processed_dataset_path + 'val/Positive/' + val_filelist[i], a=np.zeros(1), b=np.zeros(1))
        continue