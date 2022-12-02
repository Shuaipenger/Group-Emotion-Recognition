import os, random, shutil


def moveFile(fileDir, tarDir, FaceCoordinates, toFaceCoordinates, FaceFeatures, toFaceFeatures,CroppedFaces2,
             toCroppedFaces2,AlignedCroppedImages,toAlignedCroppedImages,GlobalFeatures,toGlobalFeatures ,FaceFeatures_Crop_sampling,toFaceFeatures_Crop_sampling):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.33  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(sample)

    for name in sample:
        shutil.move(fileDir + name, tarDir + name)
        name = name.split('.')[0]

        with open('test_N.txt', 'a') as f: ##记录名字
            f.write(name)

        if  os.path.isfile(FaceCoordinates  + name + '.npz'):
            shutil.move(FaceCoordinates + name + '.npz', toFaceCoordinates + name + '.npz')

        if  os.path.isfile(FaceFeatures  + name + '.npz'):
            shutil.move(FaceFeatures + name + '.npz', toFaceFeatures + name + '.npz')

        if os.path.isfile(CroppedFaces2 + name + '.npz'):
            shutil.move(CroppedFaces2 + name + '.npz', toCroppedFaces2 + name + '.npz')

        if os.path.isfile(GlobalFeatures + name + '.npz'):
            shutil.move(GlobalFeatures + name + '.npz', toGlobalFeatures + name + '.npz')

        if os.path.isfile(FaceFeatures_Crop_sampling + name + '_0.jpg'):
            shutil.move(FaceFeatures_Crop_sampling + name + '_0.jpg', toFaceFeatures_Crop_sampling + name + '_0.jpg')
            shutil.move(FaceFeatures_Crop_sampling + name + '_1.jpg', toFaceFeatures_Crop_sampling + name + '_1.jpg')
            shutil.move(FaceFeatures_Crop_sampling + name + '_2.jpg', toFaceFeatures_Crop_sampling + name + '_2.jpg')
            shutil.move(FaceFeatures_Crop_sampling + name + '_3.jpg', toFaceFeatures_Crop_sampling + name + '_3.jpg')

        for i in range(16):
            if os.path.isfile(AlignedCroppedImages + name + '_%d.jpg'%(i)):
                shutil.move(AlignedCroppedImages + name + '_%d.jpg'%(i), toAlignedCroppedImages + name + '_%d.jpg'%(i))
            else :
                break
    return

if __name__ == '__main__':
    fileDir = "/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset7:2:1/valtest/Negative/"  # 源图片文件夹路径
    tarDir = "/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset_test/test/Negative/"# 移动到新的文件夹路径

    FaceCoordinates = "/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset7:2:1/FaceCoordinates/Negative/"
    toFaceCoordinates ="/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset_test/FaceCoordinates/Negative/"

    FaceFeatures = "/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset7:2:1/FaceFeatures/Negative/"
    toFaceFeatures ="/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset_test/FaceFeatures/Negative/"

    CroppedFaces2 ="/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset7:2:1/CroppedFaces2/Negative/"
    toCroppedFaces2 ="/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset_test/CroppedFaces2/Negative/"

    AlignedCroppedImages = "/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset7:2:1/AlignedCroppedImages/Negative/"
    toAlignedCroppedImages ="/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset_test/AlignedCroppedImages/Negative/"

    GlobalFeatures =   "/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset7:2:1/GlobalFeatures/Negative/"
    toGlobalFeatures ="/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset_test/GlobalFeatures/Negative/"

    FaceFeatures_Crop_sampling = "/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset7:2:1/FaceFeatures_Crop_sampling/Negative/"
    toFaceFeatures_Crop_sampling ="/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset_test/FaceFeatures_Crop_sampling/Negative/"

    moveFile(fileDir, tarDir, FaceCoordinates, toFaceCoordinates, FaceFeatures, toFaceFeatures,CroppedFaces2,
             toCroppedFaces2,AlignedCroppedImages,toAlignedCroppedImages,GlobalFeatures,toGlobalFeatures ,FaceFeatures_Crop_sampling,toFaceFeatures_Crop_sampling)


