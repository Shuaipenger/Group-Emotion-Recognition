import os, random, shutil


def copyFile(fileDir, tarDir):

    list = sorted(os.listdir(fileDir))
    filenumber = len(list)

    for i in range(filenumber):
        name = list[i].split('.')[0] +'.jpg'
        path = "/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/train/Positive/"
        path2 = "/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/val/Positive/"
        path3 = "/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset/test/Positive/"
        if name[0] == 'p':
            shutil.copy(path +name , tarDir + name)
        elif name[0] == 'V':
            shutil.copy(path2 + name[1:], tarDir + name)
        elif name[0] == 'T':
            shutil.copy(path3 + name[1:], tarDir + name)

    return

if __name__ == '__main__':
    fileDir = r"/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset_test/FaceCoordinates/Positive/"  # 源图片文件夹路径
    tarDir = "/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset_test/test/Positive2/"# 移动到新的文件夹路径
    copyFile(fileDir, tarDir)


