import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from torchvision import transforms, datasets
import os
import torch

dataset_path = '/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/'

processed_dataset_path = '/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/'

image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_path, x))
                  for x in ['train', 'val', 'test']}

class_names = image_datasets['train'].classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print(class_names)

training_dataset = image_datasets['train']
validation_dataset = image_datasets['val']
test_dataset = image_datasets['test']

neg_train = sorted(os.listdir(dataset_path + 'train/Negative/'))
pos_train = sorted(os.listdir(dataset_path + 'train/Positive/'))

neg_val = sorted(os.listdir(dataset_path + 'val/Negative/'))
pos_val = sorted(os.listdir(dataset_path + 'val/Positive/'))

neg_test = sorted(os.listdir(dataset_path + 'test/Negative/'))
pos_test = sorted(os.listdir(dataset_path + 'test/Positive/'))


neg_train_filelist = [x.split('.')[0] for x in neg_train]
pos_train_filelist = [x.split('.')[0] for x in pos_train]

neg_val_filelist = [x.split('.')[0] for x in neg_val]
pos_val_filelist = [x.split('.')[0] for x in pos_val]

neg_test_filelist = [x.split('.')[0] for x in neg_test]
pos_test_filelist = [x.split('.')[0] for x in pos_test]


train_filelist = neg_train_filelist + pos_train_filelist
val_filelist = neg_val_filelist + pos_val_filelist
test_filelist = neg_test_filelist + pos_test_filelist

# for i in range(len(training_dataset)):
#     image, label = training_dataset[i]
#     print(train_filelist[i])
#     try:
#         if label == 0:
#             img = Image.open(dataset_path + 'train/Negative/' + train_filelist[i] + '.jpg')
#             img = img.convert('RGB')
#
#             print(img.size)
#             cropped1 = img.crop((424,222, 536, 318))  # (left, upper, right, lower) 左上，右下(x1,y1,x2,y2)
#             cropped2 = img.crop((1384, 222, 1496, 318))
#             cropped3 = img.crop((424, 762, 536, 858))
#             cropped4 = img.crop((1384, 762, 1496, 858))
#
#             cropped1.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/train/Negative/" + str(train_filelist[i]) + "_0" + ".jpg")
#             cropped2.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/train/Negative/" + str(train_filelist[i]) + "_1" + ".jpg")
#             cropped3.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/train/Negative/" + str(train_filelist[i]) + "_2" + ".jpg")
#             cropped4.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/train/Negative/" + str(train_filelist[i]) + "_3" + ".jpg")
#
#         elif label == 1:
#             img = Image.open(dataset_path + 'train/Positive/' + train_filelist[i] + '.jpg')
#             img = img.convert('RGB')
#             print(img.size)
#             cropped1 = img.crop((424,222, 536, 318))  # (left, upper, right, lower) 左上，右下(x1,y1,x2,y2)
#             cropped2 = img.crop((1384, 222, 1496, 318))
#             cropped3 = img.crop((424, 762, 536, 858))
#             cropped4 = img.crop((1384, 762, 1496, 858))
#
#             cropped1.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/train/Positive/" + str(train_filelist[i]) + "_0" + ".jpg")
#             cropped2.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/train/Positive/" + str(train_filelist[i]) + "_1" + ".jpg")
#             cropped3.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/train/Positive/" + str(train_filelist[i]) + "_2" + ".jpg")
#             cropped4.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/train/Positive/" + str(train_filelist[i]) + "_3" + ".jpg")
#
#     except ValueError:
#         print('No area detected for ' + train_filelist[i] )
#         continue

# for i in range(len(validation_dataset)):
#     image, label = validation_dataset[i]
#     print(val_filelist[i])
#     try:
#         if label == 0:
#             img = Image.open(dataset_path + 'val/Negative/' + val_filelist[i] + '.jpg')
#             img = img.convert('RGB')
#             print(img.size)
#             cropped1 = img.crop((424,222, 536, 318))  # (left, upper, right, lower) 左上，右下(x1,y1,x2,y2)
#             cropped2 = img.crop((1384, 222, 1496, 318))
#             cropped3 = img.crop((424, 762, 536, 858))
#             cropped4 = img.crop((1384, 762, 1496, 858))
#
#             cropped1.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/val/Negative/" + str(val_filelist[i]) + "_0" + ".jpg")
#             cropped2.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/val/Negative/" + str(val_filelist[i]) + "_1" + ".jpg")
#             cropped3.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/val/Negative/" + str(val_filelist[i]) + "_2" + ".jpg")
#             cropped4.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/val/Negative/" + str(val_filelist[i]) + "_3" + ".jpg")
#
#         elif label == 1:
#             img = Image.open(dataset_path + 'val/Positive/' + val_filelist[i] + '.jpg')
#             img = img.convert('RGB')
#             print(img.size)
#             cropped1 = img.crop((424,222, 536, 318))  # (left, upper, right, lower) 左上，右下(x1,y1,x2,y2)
#             cropped2 = img.crop((1384, 222, 1496, 318))
#             cropped3 = img.crop((424, 762, 536, 858))
#             cropped4 = img.crop((1384, 762, 1496, 858))
#
#             cropped1.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/val/Positive/" + str(val_filelist[i]) + "_0" + ".jpg")
#             cropped2.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/val/Positive/" + str(val_filelist[i]) + "_1" + ".jpg")
#             cropped3.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/val/Positive/" + str(val_filelist[i]) + "_2" + ".jpg")
#             cropped4.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/val/Positive/" + str(val_filelist[i]) + "_3" + ".jpg")
#
#     except ValueError:
#         print('No area detected for ' + val_filelist[i] )
#         continue

for i in range(len(test_dataset)):
    image, label = test_dataset[i]
    print(test_filelist[i])
    try:
        if label == 0:
            img = Image.open(dataset_path + 'test/Negative/' + test_filelist[i] + '.jpg')
            img = img.convert('RGB')
            print(img.size)
            cropped1 = img.crop((424,222, 536, 318))  # (left, upper, right, lower) 左上，右下(x1,y1,x2,y2)
            cropped2 = img.crop((1384, 222, 1496, 318))
            cropped3 = img.crop((424, 762, 536, 858))
            cropped4 = img.crop((1384, 762, 1496, 858))

            cropped1.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/test/Negative/" + str(test_filelist[i]) + "_0" + ".jpg")
            cropped2.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/test/Negative/" + str(test_filelist[i]) + "_1" + ".jpg")
            cropped3.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/test/Negative/" + str(test_filelist[i]) + "_2" + ".jpg")
            cropped4.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/test/Negative/" + str(test_filelist[i]) + "_3" + ".jpg")

        elif label == 1:
            img = Image.open(dataset_path + 'test/Positive/' + test_filelist[i] + '.jpg')
            img = img.convert('RGB')
            print(img.size)
            cropped1 = img.crop((424,222, 536, 318))  # (left, upper, right, lower) 左上，右下(x1,y1,x2,y2)
            cropped2 = img.crop((1384, 222, 1496, 318))
            cropped3 = img.crop((424, 762, 536, 858))
            cropped4 = img.crop((1384, 762, 1496, 858))

            cropped1.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/test/Positive/" + str(test_filelist[i]) + "_0" + ".jpg")
            cropped2.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/test/Positive/" + str(test_filelist[i]) + "_1" + ".jpg")
            cropped3.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/test/Positive/" + str(test_filelist[i]) + "_2" + ".jpg")
            cropped4.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/FaceFeatures_Crop_sampling/test/Positive/" + str(test_filelist[i]) + "_3" + ".jpg")

    except ValueError:
        print('No area detected for ' + test_filelist[i] )
        continue
