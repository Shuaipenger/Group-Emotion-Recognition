# %% md

# Import Modules

# %%

# % matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from PIL import Image

# %% md

# Load Train and Val Dataset

# %%

neg_train = sorted(os.listdir('./Dataset/wzq_dataset/CroppedFaces/train/Negative/'))
pos_train = sorted(os.listdir('./Dataset/wzq_dataset/CroppedFaces/train/Positive/'))

print(len(neg_train))
print(len(pos_train))

# %%

neg_val = sorted(os.listdir('./Dataset/wzq_dataset/CroppedFaces/val/Negative/'))
pos_val = sorted(os.listdir('./Dataset/wzq_dataset/CroppedFaces/val/Positive/'))

print(len(neg_val))
print(len(pos_val))

# %% md

# Align Faces of Training Dataset

# %%

# for j in range(len(neg_train)-1):
#
#     print(j)
#     np.load.__defaults__ = (None, True, True, 'ASCII')
#     data = np.load("./Dataset/wzq_dataset/CroppedFaces/train/Negative/neg_" + str(j) + ".npz")
#     np.load.__defaults__ = (None, False, True, 'ASCII')
#
#     faces = data['a']
#     landmarks = data['b']
#     desiredFaceWidth = 112
#     desiredFaceHeight = 96
#     desiredLeftEye = [0.30, 0.30]
#     desiredRightEyeX = 1.0 - desiredLeftEye[0]
#
#     if landmarks.all() == [0]:
#         continue
#
#     for i in range(len(faces)):
#         leftEyeCenter = [landmarks[i][0], landmarks[i][5]]
#         rightEyeCenter = [landmarks[i][1], landmarks[i][6]]
#
#         dY = rightEyeCenter[1] - leftEyeCenter[1]
#         dX = rightEyeCenter[0] - leftEyeCenter[0]
#         angle = np.degrees(np.arctan2(dY, dX))
#
#         eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) / 2, (leftEyeCenter[1] + rightEyeCenter[1]) / 2)
#
#         dist = np.sqrt((dX ** 2) + (dY ** 2))
#         desiredDist = (desiredRightEyeX - desiredLeftEye[0])
#         desiredDist *= desiredFaceWidth
#         scale = desiredDist / dist
#
#         M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
#
#         tX = desiredFaceWidth * 0.5
#         tY = desiredFaceHeight * desiredLeftEye[1]
#         M[0, 2] += (tX - eyesCenter[0])
#         M[1, 2] += (tY - eyesCenter[1])
#
#         (w, h) = (desiredFaceWidth, desiredFaceHeight)
#         output = cv2.warpAffine(faces[i], M, (w, h), flags=cv2.INTER_CUBIC)
#
#         im = Image.fromarray(output)
#         # im.save(r".\Dataset\wzq_dataset\AlignedCroppedImages\train\Negative\neg_" + str(j) + "_" + str(i) + ".jpg")
#         # im.show()
#         im.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/AlignedCroppedImages/train/Negative/neg_" + str(j) + "_" + str(i) + ".jpg")
#     j = j + 1




# %%

for j in range(len(pos_train)-1):

    print(j)

    np.load.__defaults__ = (None, True, True, 'ASCII')
    data = np.load("./Dataset/wzq_dataset/CroppedFaces/train/Positive/pos_" + str(j) + ".npz")
    np.load.__defaults__ = (None, False, True, 'ASCII')

    faces = data['a']
    landmarks = data['b']
    desiredFaceWidth = 112
    desiredFaceHeight = 96
    desiredLeftEye = [0.30, 0.30]
    desiredRightEyeX = 1.0 - desiredLeftEye[0]

    if landmarks.all() == [0]:
        continue

    for i in range(len(faces)):
        leftEyeCenter = [landmarks[i][0], landmarks[i][5]]
        rightEyeCenter = [landmarks[i][1], landmarks[i][6]]

        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX))

        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) / 2, (leftEyeCenter[1] + rightEyeCenter[1]) / 2)

        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - desiredLeftEye[0])
        desiredDist *= desiredFaceWidth
        scale = desiredDist / dist

        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        tX = desiredFaceWidth * 0.5
        tY = desiredFaceHeight * desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        (w, h) = (desiredFaceWidth, desiredFaceHeight)
        output = cv2.warpAffine(faces[i], M, (w, h), flags=cv2.INTER_CUBIC)

        im = Image.fromarray(output)
        im.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/AlignedCroppedImages/train/Positive/pos_" + str(j) + "_" + str(i) + ".jpg")

    j = j + 1
# %% md
#
# Align Faces of Val Dataset
#
# %%

# for j in range(len(neg_val)-1):
#
#     print(j)
#     np.load.__defaults__ = (None, True, True, 'ASCII')
#     data = np.load("./Dataset/wzq_dataset/CroppedFaces/val/Negative/neg_" + str(j) + ".npz")
#     np.load.__defaults__ = (None, False, True, 'ASCII')
#
#     faces = data['a']
#     landmarks = data['b']
#     desiredFaceWidth = 112
#     desiredFaceHeight = 96
#     desiredLeftEye = [0.30, 0.30]
#     desiredRightEyeX = 1.0 - desiredLeftEye[0]
#
#     if landmarks.all() == [0]:
#         continue
#
#     for i in range(len(faces)):
#         leftEyeCenter = [landmarks[i][0], landmarks[i][5]]
#         rightEyeCenter = [landmarks[i][1], landmarks[i][6]]
#
#         dY = rightEyeCenter[1] - leftEyeCenter[1]
#         dX = rightEyeCenter[0] - leftEyeCenter[0]
#         angle = np.degrees(np.arctan2(dY, dX))
#
#         eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) / 2, (leftEyeCenter[1] + rightEyeCenter[1]) / 2)
#
#         dist = np.sqrt((dX ** 2) + (dY ** 2))
#         desiredDist = (desiredRightEyeX - desiredLeftEye[0])
#         desiredDist *= desiredFaceWidth
#         scale = desiredDist / dist
#
#         M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
#
#         tX = desiredFaceWidth * 0.5
#         tY = desiredFaceHeight * desiredLeftEye[1]
#         M[0, 2] += (tX - eyesCenter[0])
#         M[1, 2] += (tY - eyesCenter[1])
#
#         (w, h) = (desiredFaceWidth, desiredFaceHeight)
#         output = cv2.warpAffine(faces[i], M, (w, h), flags=cv2.INTER_CUBIC)
#
#         im = Image.fromarray(output)
#         im.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/AlignedCroppedImages/val/Negative/neg_" + str(j) + "_" + str(i) + ".jpg")

#
#
# for j in range(len(pos_val)-1):
#
#     print(j)
#     np.load.__defaults__ = (None, True, True, 'ASCII')
#     data = np.load("./Dataset/wzq_dataset/CroppedFaces/val/Positive/pos_" + str(j) + ".npz")
#     np.load.__defaults__ = (None, False, True, 'ASCII')
#
#     faces = data['a']
#     landmarks = data['b']
#     desiredFaceWidth = 112
#     desiredFaceHeight = 96
#     desiredLeftEye = [0.30, 0.30]
#     desiredRightEyeX = 1.0 - desiredLeftEye[0]
#
#     if landmarks.all() == [0]:
#         continue
#
#     for i in range(len(faces)):
#         leftEyeCenter = [landmarks[i][0], landmarks[i][5]]
#         rightEyeCenter = [landmarks[i][1], landmarks[i][6]]
#
#         dY = rightEyeCenter[1] - leftEyeCenter[1]
#         dX = rightEyeCenter[0] - leftEyeCenter[0]
#         angle = np.degrees(np.arctan2(dY, dX))
#
#         eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) / 2, (leftEyeCenter[1] + rightEyeCenter[1]) / 2)
#
#         dist = np.sqrt((dX ** 2) + (dY ** 2))
#         desiredDist = (desiredRightEyeX - desiredLeftEye[0])
#         desiredDist *= desiredFaceWidth
#         scale = desiredDist / dist
#
#         M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
#
#         tX = desiredFaceWidth * 0.5
#         tY = desiredFaceHeight * desiredLeftEye[1]
#         M[0, 2] += (tX - eyesCenter[0])
#         M[1, 2] += (tY - eyesCenter[1])
#
#         (w, h) = (desiredFaceWidth, desiredFaceHeight)
#         output = cv2.warpAffine(faces[i], M, (w, h), flags=cv2.INTER_CUBIC)
#
#         im = Image.fromarray(output)
#         im.save("/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/AlignedCroppedImages/val/Positive/pos_" + str(j) + "_" + str(i) + ".jpg")
