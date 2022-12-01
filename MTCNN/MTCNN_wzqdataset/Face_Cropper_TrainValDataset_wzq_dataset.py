# %% md

# Import Modules

# %%

import warnings

warnings.filterwarnings('ignore')

from src import detect_faces, show_bboxes
from PIL import Image

import torch
from torchvision import transforms, datasets
import numpy as np
import os

# %% md

# Path Definition

# %%

dataset_path = '/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/'
face_coordinates_directory = '/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceCoordinates/'
processed_dataset_path = '/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/CroppedFaces2/'

# %% md

# Load Train and Val Dataset

# %%

image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_path, x))
                  for x in ['train', 'val','test']}

class_names = image_datasets['train'].classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%

print(class_names)

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

# %%

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

# %% md

# Crop Faces

# %%
#
# for i in range(len(training_dataset)):
#     try:
#         image, label = training_dataset[i]
#         face_list = []
#         landmarks_new_coordinates = []
#         if label == 0:
#             if os.path.isfile(processed_dataset_path + 'train/Negative/' + train_filelist[i] + '.npz'):
#                 print(train_filelist[i] + ' Already present')
#                 continue
#
#             bbox_lm = np.load(face_coordinates_directory + 'train/Negative/' + train_filelist[i] + '.npz')
#             bounding_boxes = bbox_lm['a']
#             if bounding_boxes.size == 0 or (bounding_boxes[0] == 0).all():
#                 print("No bounding boxes for " + train_filelist[i] + ". Adding empty file for the same")
#                 # np.savez(processed_dataset_path + 'train/Negative/' + train_filelist[i], a=np.zeros(1), b=np.zeros(1))
#                 # os.remove(processed_dataset_path + 'train/Negative/' + train_filelist[i])
#
#                 continue
#             landmarks = bbox_lm['b']
#
#             for j in range(len(bounding_boxes)):
#                 bbox_coordinates = bounding_boxes[j]
#                 landmark = landmarks[j]
#                 img_face = image.crop(
#                     (bbox_coordinates[0], bbox_coordinates[1], bbox_coordinates[2], bbox_coordinates[3]))
#
#                 x = bbox_coordinates[0]
#                 y = bbox_coordinates[1]
#
#                 for k in range(5):
#                     landmark[k] -= x
#                     landmark[k + 5] -= y
#                 img_face = np.array(img_face)
#                 landmark = np.array(landmark)
#
#                 if len(face_list) != 0:
#                     if img_face.shape[0] == face_list[-1].shape[0]:
#                         img_face = image.crop((bbox_coordinates[0] - 1, bbox_coordinates[1] - 1, bbox_coordinates[2],
#                                                bbox_coordinates[3]))
#                         img_face = np.array(img_face)
#                         landmark += 1
#
#                 face_list.append(img_face)
#                 landmarks_new_coordinates.append(landmark)
#             face_list = np.asarray(face_list)
#             landmarks_new_coordinates = np.asarray(landmarks_new_coordinates)
#             np.savez(processed_dataset_path + 'train/Negative/' + train_filelist[i], a=face_list,
#                      b=landmarks_new_coordinates)
#
#         elif label == 1:
#
#             if os.path.isfile(processed_dataset_path + 'train/Positive/' + train_filelist[i] + '.npz'):
#                 print(train_filelist[i] + ' Already present')
#                 continue
#
#             bbox_lm = np.load(face_coordinates_directory + 'train/Positive/' + train_filelist[i] + '.npz')
#             bounding_boxes = bbox_lm['a']
#             if bounding_boxes.size == 0 or (bounding_boxes[0] == 0).all():
#                 print("No bounding boxes for " + train_filelist[i] + ". Adding empty file for the same")
#                 # np.savez(processed_dataset_path + 'train/Positive/' + train_filelist[i], a=np.zeros(1), b=np.zeros(1))
#
#                 # os.remove(processed_dataset_path + 'train/Positive/' + train_filelist[i])
#
#                 continue
#             landmarks = bbox_lm['b']
#
#             for j in range(len(bounding_boxes)):
#                 bbox_coordinates = bounding_boxes[j]
#                 landmark = landmarks[j]
#                 img_face = image.crop(
#                     (bbox_coordinates[0], bbox_coordinates[1], bbox_coordinates[2], bbox_coordinates[3]))
#
#                 x = bbox_coordinates[0]
#                 y = bbox_coordinates[1]
#
#                 for k in range(5):
#                     landmark[k] -= x
#                     landmark[k + 5] -= y
#                 img_face = np.array(img_face)
#                 landmark = np.array(landmark)
#
#                 if len(face_list) != 0:
#                     if img_face.shape[0] == face_list[-1].shape[0]:
#                         img_face = image.crop((bbox_coordinates[0] - 1, bbox_coordinates[1] - 1, bbox_coordinates[2],
#                                                bbox_coordinates[3]))
#                         img_face = np.array(img_face)
#                         landmark += 1
#
#                 face_list.append(img_face)
#                 landmarks_new_coordinates.append(landmark)
#             face_list = np.asarray(face_list)
#             landmarks_new_coordinates = np.asarray(landmarks_new_coordinates)
#             np.savez(processed_dataset_path + 'train/Positive/' + train_filelist[i], a=face_list,
#                      b=landmarks_new_coordinates)
#
#         if i % 100 == 0:
#             print(i)
#     except:
#         print("Error/interrput at validation dataset file " + val_filelist[i])
#         print(bounding_boxes)
#         print(landmarks)
#         print(bounding_boxes.shape)
#         print(landmarks.shape)
#         break
#
# for i in range(len(validation_dataset)):
#     try:
#         image, label = validation_dataset[i]
#         face_list = []
#         landmarks_new_coordinates = []
#         if label == 0:
#             if os.path.isfile(processed_dataset_path + 'val/Negative/' + val_filelist[i] + '.npz'):
#                 print(val_filelist[i] + ' Already present')
#                 continue
#
#             bbox_lm = np.load(face_coordinates_directory + 'val/Negative/' + val_filelist[i] + '.npz')
#             bounding_boxes = bbox_lm['a']
#             if bounding_boxes.size == 0 or (bounding_boxes[0] == 0).all():
#                 print("No bounding boxes for " + val_filelist[i] + ". Adding empty file for the same")
#                 # np.savez(processed_dataset_path + 'val/Negative/' + val_filelist[i], a=np.zeros(1), b=np.zeros(1))
#
#                 # os.remove(processed_dataset_path + 'val/Negative/' + val_filelist[i])
#
#                 continue
#             landmarks = bbox_lm['b']
#
#             for j in range(len(bounding_boxes)):
#                 bbox_coordinates = bounding_boxes[j]
#                 landmark = landmarks[j]
#                 img_face = image.crop(
#                     (bbox_coordinates[0], bbox_coordinates[1], bbox_coordinates[2], bbox_coordinates[3]))
#
#                 x = bbox_coordinates[0]
#                 y = bbox_coordinates[1]
#
#                 for k in range(5):
#                     landmark[k] -= x
#                     landmark[k + 5] -= y
#                 img_face = np.array(img_face)
#                 landmark = np.array(landmark)
#
#                 if len(face_list) != 0:
#                     if img_face.shape[0] == face_list[-1].shape[0]:
#                         img_face = image.crop((bbox_coordinates[0] - 1, bbox_coordinates[1] - 1, bbox_coordinates[2],
#                                                bbox_coordinates[3]))
#                         img_face = np.array(img_face)
#                         landmark += 1
#
#                 face_list.append(img_face)
#                 landmarks_new_coordinates.append(landmark)
#             face_list = np.asarray(face_list)
#             landmarks_new_coordinates = np.asarray(landmarks_new_coordinates)
#             np.savez(processed_dataset_path + 'val/Negative/' + val_filelist[i], a=face_list,
#                      b=landmarks_new_coordinates)
#
#         elif label == 1:
#
#             if os.path.isfile(processed_dataset_path + 'val/Positive/' + val_filelist[i] + '.npz'):
#                 print(val_filelist[i] + ' Already present')
#                 continue
#
#             bbox_lm = np.load(face_coordinates_directory + 'val/Positive/' + val_filelist[i] + '.npz')
#             bounding_boxes = bbox_lm['a']
#             if bounding_boxes.size == 0 or (bounding_boxes[0] == 0).all():
#                 print("No bounding boxes for " + val_filelist[i] + ". Adding empty file for the same")
#                 # np.savez(processed_dataset_path + 'val/Positive/' + val_filelist[i], a=np.zeros(1), b=np.zeros(1))
#                 # os.remove(processed_dataset_path + 'val/Positive/' + val_filelist[i])
#                 continue
#             landmarks = bbox_lm['b']
#
#             for j in range(len(bounding_boxes)):
#                 bbox_coordinates = bounding_boxes[j]
#                 landmark = landmarks[j]
#                 img_face = image.crop(
#                     (bbox_coordinates[0], bbox_coordinates[1], bbox_coordinates[2], bbox_coordinates[3]))
#
#                 x = bbox_coordinates[0]
#                 y = bbox_coordinates[1]
#
#                 for k in range(5):
#                     landmark[k] -= x
#                     landmark[k + 5] -= y
#                 img_face = np.array(img_face)
#                 landmark = np.array(landmark)
#
#                 if len(face_list) != 0:
#                     if img_face.shape[0] == face_list[-1].shape[0]:
#                         img_face = image.crop((bbox_coordinates[0] - 1, bbox_coordinates[1] - 1, bbox_coordinates[2],
#                                                bbox_coordinates[3]))
#                         img_face = np.array(img_face)
#                         landmark += 1
#
#                 face_list.append(img_face)
#                 landmarks_new_coordinates.append(landmark)
#             face_list = np.asarray(face_list)
#             landmarks_new_coordinates = np.asarray(landmarks_new_coordinates)
#             np.savez(processed_dataset_path + 'val/Positive/' + val_filelist[i], a=face_list,
#                      b=landmarks_new_coordinates)
#
#         if i % 100 == 0:
#             print(i)
#     except:
#         print("Error/interrput at validation dataset file " + val_filelist[i])
#         print(bounding_boxes)
#         print(landmarks)
#         print(bounding_boxes.shape)
#         print(landmarks.shape)
#         break

for i in range(len(test_dataset)):
    try:
        image, label = test_dataset[i]
        face_list = []
        landmarks_new_coordinates = []
        if label == 0:
            if os.path.isfile(processed_dataset_path + 'test/Negative/' + test_filelist[i] + '.npz'):
                print(test_filelist[i] + ' Already present')
                continue

            bbox_lm = np.load(face_coordinates_directory + 'test/Negative/' + test_filelist[i] + '.npz')
            bounding_boxes = bbox_lm['a']
            if bounding_boxes.size == 0 or (bounding_boxes[0] == 0).all():
                print("No bounding boxes for " + test_filelist[i] + ". Adding empty file for the same")
                # np.savez(processed_dataset_path + 'val/Negative/' + val_filelist[i], a=np.zeros(1), b=np.zeros(1))

                # os.remove(processed_dataset_path + 'val/Negative/' + val_filelist[i])

                continue
            landmarks = bbox_lm['b']

            for j in range(len(bounding_boxes)):
                bbox_coordinates = bounding_boxes[j]
                landmark = landmarks[j]
                img_face = image.crop(
                    (bbox_coordinates[0], bbox_coordinates[1], bbox_coordinates[2], bbox_coordinates[3]))

                x = bbox_coordinates[0]
                y = bbox_coordinates[1]

                for k in range(5):
                    landmark[k] -= x
                    landmark[k + 5] -= y
                img_face = np.array(img_face)
                landmark = np.array(landmark)

                if len(face_list) != 0:
                    if img_face.shape[0] == face_list[-1].shape[0]:
                        img_face = image.crop((bbox_coordinates[0] - 1, bbox_coordinates[1] - 1, bbox_coordinates[2],
                                               bbox_coordinates[3]))
                        img_face = np.array(img_face)
                        landmark += 1

                face_list.append(img_face)
                landmarks_new_coordinates.append(landmark)
            face_list = np.asarray(face_list)
            landmarks_new_coordinates = np.asarray(landmarks_new_coordinates)
            np.savez(processed_dataset_path + 'test/Negative/' + test_filelist[i], a=face_list, b=landmarks_new_coordinates)

        elif label == 1:

            if os.path.isfile(processed_dataset_path + 'test/Positive/' + test_filelist[i] + '.npz'):
                print(test_filelist[i] + ' Already present')
                continue

            bbox_lm = np.load(face_coordinates_directory + 'test/Positive/' + test_filelist[i] + '.npz')
            bounding_boxes = bbox_lm['a']
            if bounding_boxes.size == 0 or (bounding_boxes[0] == 0).all():
                print("No bounding boxes for " + test_filelist[i] + ". Adding empty file for the same")
                # np.savez(processed_dataset_path + 'val/Positive/' + val_filelist[i], a=np.zeros(1), b=np.zeros(1))
                # os.remove(processed_dataset_path + 'val/Positive/' + val_filelist[i])
                continue
            landmarks = bbox_lm['b']

            for j in range(len(bounding_boxes)):
                bbox_coordinates = bounding_boxes[j]
                landmark = landmarks[j]
                img_face = image.crop(
                    (bbox_coordinates[0], bbox_coordinates[1], bbox_coordinates[2], bbox_coordinates[3]))

                x = bbox_coordinates[0]
                y = bbox_coordinates[1]

                for k in range(5):
                    landmark[k] -= x
                    landmark[k + 5] -= y
                img_face = np.array(img_face)
                landmark = np.array(landmark)

                if len(face_list) != 0:
                    if img_face.shape[0] == face_list[-1].shape[0]:
                        img_face = image.crop((bbox_coordinates[0] - 1, bbox_coordinates[1] - 1, bbox_coordinates[2],
                                               bbox_coordinates[3]))
                        img_face = np.array(img_face)
                        landmark += 1

                face_list.append(img_face)
                landmarks_new_coordinates.append(landmark)
            face_list = np.asarray(face_list)
            landmarks_new_coordinates = np.asarray(landmarks_new_coordinates)
            np.savez(processed_dataset_path + 'test/Positive/' + test_filelist[i], a=face_list,
                     b=landmarks_new_coordinates)

        if i % 100 == 0:
            print(i)
    except:
        print("Error/interrput at validation dataset file " + test_filelist[i])
        print(bounding_boxes)
        print(landmarks)
        print(bounding_boxes.shape)
        print(landmarks.shape)
        break