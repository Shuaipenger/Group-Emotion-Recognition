## 安装环境

存储库需要以下依赖项：

+ Python v3.8.12
+ PyTorch v1.10.0
+ TorchVision v0.11.1
+ NumPy
+ SciPy
+ Scikit-Learn
+ Matplotlib
+ PIL
+ Pickle

## Group_Emotion_Recognition文件介绍

**Models_GAF3.0**：在GAF2.0数据集上训练的代码文件。

**Models_GroupandScene**：在我们的数据集上训练的代码文件。

**MTCNN**：提取人脸图片的相关文件。

**My_ModelOutputs**：在GAF2.0数据库上训练后保存的模型结果文件、准确率复现代码和提取模型结果的代码文件。

**My_modelOutputs_wzq_dataset**：在我们数据库上训练后的模型结果文件、准确率复现代码和提取模型结果的代码文件。

**TrainedModels**：保存的模型文件。

**Visualization**：混淆矩阵可视化。

## Dataset文件介绍

包含emotic、emotiw和我们自己收集的数据集，以及中间处理文件。

**AlignedCroppedImages**: 剪切对齐后的GAF2.0数据集中人脸图片。

**CroppedFaces**：剪切处理的GAF2.0数据集中人脸图片

**FaceCoordinates**：处理GAF2.0数据集中人脸位置坐标文件。

**FaceFeatures**：处理GAF2.0数据集得到的提取的人脸特征向量。

**GlobalFeatures**：使用Densenet161网络提取GAF2.0数据集图片得到的全局特征向量。

**Process_wzq_dataset**：制作数据集时，从视频中处理提取图片的程序。

**emotic**：emotic数据集文件。

**emotiw**：GAF2.0数据集文件。

**emotiw_skeletons**：使用openpose处理GAF2.0数据集得到的人体骨骼姿态图片。

**wzq_dataset**：我们收集的数据集文件和相关处理后的图片文件。
  *  AlignedCroppedImages：剪切并对齐处理的我们的数据集得到的人脸图片。
  *  CroppedFaces2：剪切处理的我们的数据集得到的人脸图片。
  *  FaceFeatures2：处理我们数据集提取的人脸特征向量。
  *  GlobalFeatures：使用Densenet161网络提取的全局特征向量。
  *  FaceFeatures_Crop_sampling：剪切得到的图片块
  *  train:训练集
  *  val:验证集
  *  test:测试集
  *  train_skeleton:训练集提取的人体姿态图
  *  val_skeleton:验证集提取的人体姿态图
  *  test_skeleton:测试集提取的人体姿态图
  
**test_list**：这包含来自GAF2.0数据集的图像列表，用作EVAL数据集。

**val_list**：这包含来自GAF2.0数据集的图像列表，用作VAL数据集。

### 处理数据集过程：
图片->MTCNN->制作FaceCoordinates、FaceFeatures、CroppedFaces、AlignedCroppedimages文件

**FaceCoordinates**：用`MTCNN/Face_Extractor_BB_Landmarks_Test.ipynb`和`MTCNN/Face_Extractor_ BB_ Landmarks_ Train.ipynb`处理数据集得到

**FaceFeatures**：用`Face_Extractor_Feature_Test.py`和`Face_Extractor_Feature_Train.ipynb`处理数据集得到。使用 `MTCNN`提取图像中人脸的特征向量。这包含从 MTCNN 的最后一层提取的相应图像中人脸的 256 维面部特征列表。

**CroppedFaces**：由`Face_Cropper_TestDataset.ipynb`和 `Face_Cropper_TrainValDataset.ipynb`处理数据集和`FaceCoordinates`文件得来。

**AlignedCroppedimages**：用`AlignedFaces_Extractor_Train.ipynb`和`AlignedFaces_Extractor_Test.ipynb`处理`CroppedFaces`得到。


## 结果复现
**复现GAF2.0数据集上的结果**

1.使用代码文件训练并保存为模型文件

2.使用模型文件提取npz文件（使用`My_ModelX_OutputSaver_XXX.py`代码提取npz）      

3.①复现准确率（`Acc_Reappearance.py`）②用npz文件做Ensemble调比例（`GirdSearch.py`）

|   | 代码 | 模型文件  | 保存的npz结果文件  |  
|---|------|---|---|
|  Model0 | Model0_AlignedModelTrainerSoftmax_AlignedModel_EmotiW_lr01_Softmax.py | AlignedModelTrainerSoftmax_AlignedModel_EmotiW_lr01_Softmax-shiyan  |  model0_output_data |   |   |
|  Model1 | Model1_Densenet_Emotiw_PretrainEmotiC_lr001.py     |model_1_2_densenet_emotiw_pretrainemotic_lr001.pt | model1_output_data  |   
|  Model2 | Model2_resnet18_emotiW.py                          |model_2_2_resnet18_EmotiW   | model2_output_data  |   
|  Model3 | Model3_skeletons.py | model_3_1_DenseNet161_skeletons_model1  | model3_output_data  |   
|  Model4 | Model4_skeletons_EfficientNet.py  | EfficientNet_skeletons  |model4_1_output_data   |  
|  Model5 | Model5_FaceAttention_AlignedModel_pre_cross_vit.py  |FaceAttention_AlignedModel_FullTrain_lr001_dropout_BN_SoftmaxLr01   |model5_output_data   |  

**复现在Group and Scene数据集上的结果**

1.使用代码文件训练并保存为模型文件

2.使用模型文件提取npz文件（使用`My_ModelX_OutputSaverXXX.py`代码提取npz）      

3.①复现准确率（`Acc_Reappearance_wzq.py`）②用npz文件做Ensemble调比例（`GirdSearch_wzq.py`）

| |   | 代码 | 模型文件  | 保存的npz结果文件  |  
|---|---|------|---|---|
| Scene_densenet | Model1 | Densenet_Emotiw_lr001_wzq.py    |model_1_2_densenet_New_data.pt | model1_Scene_Densenet_output_data  |   
| Scene_resnet | Model2 | resnet18_emotiW_wzq.py         |model_2_1_resnet18_EmotiW_wzq  | model2_Scene_resnet_output_data |   
| Skeleton_densenet | Model3 | 7_skeletons_wzq.py  | model_skeleton_densenet_New_data2.pt | model3_Skeleton_densenet_output_data |   
| Skeleton_efficientnet | Model4 | 7_skeletons_EfficientNet_wzq.py | model_skeleton_EfficientNet_New_data.pt | model4_Skeleton_Efficient_output_data  |  
| Hybrid network model | Model5 | 1_FaceAttention_AlignedModel_pre_cross_vit_wzq_onlyone_crossvit.py |model_5_2_All_New_data  |model5_2_output_data_onecrossvit  |  
| Face Detected |Hybird Network Model| 1_FaceAttention_AlignedModel_pre_cross_vit_wzq_two_crossvit_Faces.py | model_6_2A_All_New_data  |  model6_2A_output_data.npz |  
| Face Detected | Scene-Densenet161| Densenet_Emotiw_lr001_wzq_Face.py   |model_1_2_densenet_New_data_Face.pt | model_output_data_Densenet_Face2.npz |   
| No Face Detected |  Hybird Network Model |1_FaceAttention_AlignedModel_pre_cross_vit_wzq_two_crossvit_NoFaces.py |model_6_2B_All_New_data  | model6_2B_output_data.npz | 
| No Face Detected |  Scene-Densenet161 | Densenet_Emotiw_lr001_wzq_NoFace.py | model_1_2_densenet_New_data_NoFace.pt  | model_output_data_Densenet_No_Face2.npz  |  



# 注：

1. MTCNN 模型参考自[this](https://github.com/TropComplique/mtcnn-pytorch).

2. SphereFace 模型（用于对齐模型）的实现参考自 [this](https://github.com/clcarwin/sphereface_pytorch).

3. Openpose的代码实现自 [this](https://github.com/CMU-Perceptual-Computing-Lab/openpose).

4. 网络部分代码参考自[this](https://github.com/vlgiitr/Group-Level-Emotion-Recognition).
