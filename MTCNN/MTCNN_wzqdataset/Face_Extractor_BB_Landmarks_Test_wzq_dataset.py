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

dataset_path = '/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/'

processed_dataset_path = '/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceCoordinates/'

# %% md

# Load Train and Val Dataset

# %%

image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_path, x))
                  for x in ['test']}

class_names = image_datasets['test'].classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(class_names)

test_dataset = image_datasets['test']

neg_test = sorted(os.listdir(dataset_path + 'test/Negative/'))
pos_test = sorted(os.listdir(dataset_path + 'test/Positive/'))

neg_test_filelist = [x.split('.')[0] for x in neg_test]
pos_test_filelist = [x.split('.')[0] for x in pos_test]

print(neg_test_filelist[:10])
print(pos_test_filelist[:10])

test_filelist = neg_test_filelist  + pos_test_filelist

print(len(test_dataset))

# Extract Faces from Image using MTCNN

for i in range(len(test_dataset)):
    image, label = test_dataset[i]
    print(test_filelist[i])
    try:
        if label == 0:
            if os.path.isfile(processed_dataset_path + 'test/Negative/' + test_filelist[i] + '.npz'):
                print(test_filelist[i] + ' Already present')
                continue
            bounding_boxes, landmarks = detect_faces(image)
            bounding_boxes = np.asarray(bounding_boxes)
            if bounding_boxes.size == 0:
                print('MTCNN model handling empty face condition at ' + test_filelist[i])
            np.savez(processed_dataset_path + 'test/Negative/' + test_filelist[i], a=bounding_boxes, b=landmarks)

        else:

            if os.path.isfile(processed_dataset_path + 'test/Positive/' + test_filelist[i] + '.npz'):
                print(test_filelist[i] + ' Already present')
                continue
            bounding_boxes, landmarks = detect_faces(image)
            bounding_boxes = np.asarray(bounding_boxes)
            if bounding_boxes.size == 0:
                print('MTCNN model handling empty face condition at ' + test_filelist[i])
            np.savez(processed_dataset_path + 'test/Positive/' + test_filelist[i], a=bounding_boxes, b=landmarks)

    except ValueError:
        print('No faces detected for ' + test_filelist[i] + ". Also MTCNN failed.")
        if label == 0:
            np.savez(processed_dataset_path + 'test/Negative/' + test_filelist[i], a=np.zeros(1), b=np.zeros(1))
        elif label == 1:
            np.savez(processed_dataset_path + 'test/Positive/' + test_filelist[i], a=np.zeros(1), b=np.zeros(1))
        continue

# %%

