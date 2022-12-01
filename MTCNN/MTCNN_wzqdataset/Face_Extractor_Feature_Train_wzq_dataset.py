import warnings
warnings.filterwarnings('ignore')

from PIL import Image

import torch
from torchvision import transforms, datasets
import numpy as np
import os
import numpy as np
import torch
from torch.autograd import Variable
from src.get_nets import PNet, RNet, ONet
from src.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from src.first_stage import run_first_stage
import torch.nn as nn



# Path Definitions


dataset_path = '../Dataset/wzq_dataset/'

processed_dataset_path = '../Dataset/wzq_dataset/FaceFeatures2/'


# MTCNN Model Definition for Extracting Face Features


pnet = PNet()
rnet = RNet()
onet = ONet()
onet.eval()


# %%

class OnetFeatures(nn.Module):
    def __init__(self, original_model):
        super(OnetFeatures, self).__init__()
        self.features = nn.Sequential(*list(onet.children())[:-3])

    def forward(self, x):
        x = self.features(x)
        return x


def get_face_features(image, min_face_size=20.0,
                      thresholds=[0.6, 0.7, 0.8],
                      nms_thresholds=[0.7, 0.7, 0.7]):
    """
    Arguments:
        image: an instance of PIL.Image.
        min_face_size: a float number.
        thresholds: a list of length 3.
        nms_thresholds: a list of length 3.

    Returns:
        two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
        bounding boxes and facial landmarks.
    """

    # LOAD MODELS
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    onet.eval()

    # BUILD AN IMAGE PYRAMID
    width, height = image.size
    min_length = min(height, width)

    min_detection_size = 12
    factor = 0.707  # sqrt(0.5)

    # scales for scaling the image
    scales = []

    # scales the image so that
    # minimum size that we can detect equals to
    # minimum face size that we want to detect
    m = min_detection_size / min_face_size
    min_length *= m

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m * factor ** factor_count)
        min_length *= factor
        factor_count += 1

    # STAGE 1

    # it will be returned
    bounding_boxes = []

    # run P-Net on different scales
    for s in scales:
        boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0])
        bounding_boxes.append(boxes)

    # collect boxes (and offsets, and scores) from different scales
    bounding_boxes = [i for i in bounding_boxes if i is not None]
    bounding_boxes = np.vstack(bounding_boxes)

    keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
    bounding_boxes = bounding_boxes[keep]

    # use offsets predicted by pnet to transform bounding boxes
    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
    # shape [n_boxes, 5]

    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 2

    img_boxes = get_image_boxes(bounding_boxes, image, size=24)
    img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
    output = rnet(img_boxes)
    offsets = output[0].data.numpy()  # shape [n_boxes, 4]
    probs = output[1].data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]

    keep = nms(bounding_boxes, nms_thresholds[1])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 3

    img_boxes = get_image_boxes(bounding_boxes, image, size=48)
    if len(img_boxes) == 0:
        return [], []
    img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
    output = onet(img_boxes)

    faceFeatureModel = OnetFeatures(onet)

    featureOutputs = faceFeatureModel(img_boxes)

    landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
    offsets = output[1].data.numpy()  # shape [n_boxes, 4]
    probs = output[2].data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[2])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]
    landmarks = landmarks[keep]

    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    keep = nms(bounding_boxes, nms_thresholds[2], mode='min')

    featureOutputs = featureOutputs[keep]

    return featureOutputs

# Load Train and Val Dataset

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



neg_train_filelist = [x.split('.')[0] for x in neg_train]
pos_train_filelist = [x.split('.')[0] for x in pos_train]

neg_val_filelist = [x.split('.')[0] for x in neg_val]
pos_val_filelist = [x.split('.')[0] for x in pos_val]


train_filelist = neg_train_filelist + pos_train_filelist
val_filelist = neg_val_filelist + pos_val_filelist

print(len(training_dataset))
print(len(validation_dataset))

# Extract Face Features

#
# for i in range(len(training_dataset)):
#     image, label = training_dataset[i]
#     print(train_filelist[i])
#     try:
#         if label == 0:
#             if os.path.isfile(processed_dataset_path + 'train/Negative/' + train_filelist[i] + '.npz'):
#                 print(train_filelist[i] + ' Already present')
#                 continue
#             features = get_face_features(image)
#             if (type(features)) == tuple:
#                 with open('hello.text', 'a') as f:
#                     f.write(train_filelist[i])
#                 continue
#             features = features.data.numpy()
#
#             if features.size == 0:
#                 print('MTCNN model handling empty face condition at ' + train_filelist[i])
#             np.savez(processed_dataset_path + 'train/Negative/' + train_filelist[i], a=features)
#
#         elif label == 1:
#             if os.path.isfile(processed_dataset_path + 'train/Positive/' + train_filelist[i] + '.npz'):
#                 print(train_filelist[i] + ' Already present')
#                 continue
#             features = get_face_features(image)
#             if (type(features)) == tuple:
#                 with open('hello.text', 'a') as f:
#                     f.write(train_filelist[i])
#                 continue
#             features = features.data.numpy()
#             if features.size == 0:
#                 print('MTCNN model handling empty face condition at ' + train_filelist[i])
#             np.savez(processed_dataset_path + 'train/Positive/' + train_filelist[i], a=features)
#
#     except ValueError:
#         print('No faces detected for ' + train_filelist[i] + ". Also MTCNN failed.")
#         # if label == 0:
#         #     np.savez(processed_dataset_path + 'train/Negative/' + train_filelist[i], a=np.zeros(1))
#         # elif label == 1:
#         #     np.savez(processed_dataset_path + 'train/Positive/' + train_filelist[i], a=np.zeros(1))
#         continue

for i in range(len(validation_dataset)):
    image, label = validation_dataset[i]
    print(val_filelist[i])
    try:
        if label == 0:
            if os.path.isfile(processed_dataset_path + 'val/Negative/' + val_filelist[i] + '.npz'):
                print(val_filelist[i] + ' Already present')
                continue
            features = get_face_features(image)
            if (type(features)) == tuple:
                with open('hello.text', 'a') as f:
                    f.write(val_filelist[i])
                continue
            features = features.data.numpy()

            if features.size == 0:
                print('MTCNN model handling empty face condition at ' + val_filelist[i])
            np.savez(processed_dataset_path + 'val/Negative/' + val_filelist[i], a=features)

        elif label == 1:
            if os.path.isfile(processed_dataset_path + 'val/Positive/' + val_filelist[i] + '.npz'):
                print(val_filelist[i] + ' Already present')
                continue
            features = get_face_features(image)
            if (type(features)) == tuple:
                with open('hello.text', 'a') as f:
                    f.write(val_filelist[i])
                continue
            features = features.data.numpy()
            if features.size == 0:
                print('MTCNN model handling empty face condition at ' + val_filelist[i])
            np.savez(processed_dataset_path + 'val/Positive/' + val_filelist[i], a=features)

    except ValueError:
        print('No faces detected for ' + val_filelist[i] + ". Also MTCNN failed.")
        # if label == 0:
        #     np.savez(processed_dataset_path + 'val/Negative/' + val_filelist[i], a=np.zeros(1))
        # elif label == 1:
        #     np.savez(processed_dataset_path + 'val/Positive/' + val_filelist[i], a=np.zeros(1))
        continue