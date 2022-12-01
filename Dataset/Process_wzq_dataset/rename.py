import os
# path="/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset2/Negative/" # 文件夹路径
path="/home/prmi/GSP/Group-Level-Emotion-Recognition-master/Dataset/wzq_dataset_test/FaceFeatures/Positive/" # 文件夹路径
os.chdir(path) #更改当前路径
filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
# print(filelist) #文件夹中所有文件名
for i in range(len(filelist)):
	old_name=filelist[i] # 旧名字
	new_name = old_name[8:] # 新名字
	# new_name = 'Vneg_%d.npz'%(i) # 新名字
	# if old_name[0] == 'o'
	# 	new_name = 'p'+ old_name # 新名字

	os.rename(old_name,new_name) #重命名
