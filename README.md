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

## Group_Emotion_Recognition_HMM文件介绍

**Models_GAF3.0**：在GAF3.0数据集上训练的代码文件。

**Models_wzq_dataset**：在我们的数据集上训练的代码文件。

**MTCNN**：提取人脸图片的相关文件。

**My_ModelOutputs**：在GAF3.0数据库上训练后保存的模型结果文件、准确率复现代码和提取模型结果的代码文件。

**My_modelOutputs_wzq_dataset**：在我们数据库上训练后的模型结果文件、准确率复现代码和提取模型结果的代码文件。

**TrainedModels**：保存的模型文件。

**Visualization**：混淆矩阵可视化。

## Dataset文件介绍

包含emotic、emotiw和我们自己收集的数据集，以及中间处理文件。

**AlignedCroppedImages**: 剪切对齐后的GAF3.0数据集中人脸图片。

**CroppedFaces**：剪切处理的GAF3.0数据集中人脸图片

**FaceCoordinates**：处理GAF3.0数据集中人脸位置坐标文件。

**FaceFeatures**：处理GAF3.0数据集得到的提取的人脸特征向量。

**GlobalFeatures**：使用Densenet161网络提取GAF3.0数据集图片得到的全局特征向量。

**Process_wzq_dataset**：制作数据集时，从视频中处理提取图片的程序。

**emotic**：emotic数据集文件。

**emotiw**：GAF3.0数据集文件。

**emotiw_skeletons**：使用openpose处理GAF3.0数据集得到的人体骨骼姿态图片。

**wzq_dataset**：我们收集的数据集文件和相关处理后的图片文件。
  *  AlignedCroppedImages：剪切并对齐处理的我们的数据集得到的人脸图片。
  *  CroppedFaces：剪切处理的我们的数据集得到的人脸图片。
  *  FaceFeatures：处理我们数据集提取的人脸特征向量。
  *  GlobalFeatures：使用Densenet161网络提取的全局特征向量。
  *  FaceFeatures_Crop_sampling：剪切得到的图片块
**test_list**：这包含来自GAF3.0数据集的图像列表，用作EVAL数据集。

**val_list**：这包含来自GAF3.0数据集的图像列表，用作VAL数据集。

### 处理数据集过程：
图片->MTCNN->制作FaceCoordinates、FaceFeatures、CroppedFaces、AlignedCroppedimages文件

**FaceCoordinates**：用`MTCNN/Face_Extractor_BB_Landmarks_Test.ipynb`和`MTCNN/Face_Extractor_ BB_ Landmarks_ Train.ipynb`处理数据集得到

**FaceFeatures**：用`Face_Extractor_Feature_Test.py`和`Face_Extractor_Feature_Train.ipynb`处理数据集得到。使用 `MTCNN`提取图像中人脸的特征向量。这包含从 MTCNN 的最后一层提取的相应图像中人脸的 256 维面部特征列表。

**CroppedFaces**：由`Face_Cropper_TestDataset.ipynb`和 `Face_Cropper_TrainValDataset.ipynb`处理数据集和`FaceCoordinates`文件得来。

**AlignedCroppedimages**：用`AlignedFaces_Extractor_Train.ipynb`和`AlignedFaces_Extractor_Test.ipynb`处理`CroppedFaces`得到。


## 结果复现
**复现GAF3.0数据集上的结果**

1.使用代码文件训练并保存为模型文件

2.使用模型文件提取npz文件（使用`My_Model_OutputSaver_TrainDataset.py`代码提取npz）      

3.①用npz文件做混淆矩阵（`Confusion_matrix.py`）②复现准确率（`Acc_Reappearance.py`）③用npz文件做Ensemble调比例（`GirdSearch.py`）

|   | 代码 | 模型文件  | 保存的npz结果文件  |  
|---|------|---|---|
|  Model0 | Model0_AlignedModelTrainerSoftmax_AlignedModel_EmotiW_lr01_Softmax.py | AlignedModelTrainerSoftmax_AlignedModel_EmotiW_lr01_Softmax-shiyan  |  model0_output_data |   |   |
|  Model1 | Model1_Densenet_Emotiw_PretrainEmotiC_lr001.py     |model_1_2_densenet_emotiw_pretrainemotic_lr001.pt | model1_output_data  |   
|  Model2 | Model2_resnet18_emotiW.py                          |model_2_2_resnet18_EmotiW   | model2_output_data  |   
|  Model3 | Model3_skeletons.py | model_3_1_DenseNet161_skeletons_model1  | model3_output_data  |   
|  Model4 | Model4_skeletons_EfficientNet.py  | EfficientNet_skeletons  |model4_1_output_data   |  
|  Model5 | Model5_FaceAttention_AlignedModel_pre_cross_vit.py  |FaceAttention_AlignedModel_FullTrain_lr001_dropout_BN_SoftmaxLr01   |model5_output_data   |  

**复现我们的数据集上的结果**

1.使用代码文件训练并保存为模型文件

2.使用模型文件提取npz文件（使用`My_ModelOutputs_wzq_dataset.py`代码提取npz）      

3.①用npz文件做混淆矩阵（`Confusion_matrix_wzq_dataset.py`）②复现准确率（`Acc_Reappearance_wzq_dataset.py`）③用npz文件做Ensemble调比例（`GirdSearch_wzq_dataset.py`）

| |   | 代码 | 模型文件  | 保存的npz结果文件  |  
|---|---|------|---|---|
| Face Detected |Hybird Network Model| 1_FaceAttention_AlignedModel_pre_cross_vit_wzq_two_crossvit_Faces.py | model_6_1A_All_New_data  |  model6_1_output_data.npz |  
| Face Detected | Scene-Densenet161| Densenet_Emotiw_lr001_wzq_Face.py   |model_1_1_densenet_New_data_Face.pt | model_output_data_Densenet_Face.npz |   
| No Face Detected |  Hybird Network Model |1_FaceAttention_AlignedModel_pre_cross_vit_wzq_two_crossvit_NoFaces.py |model_6_1B_All_New_data2  | model6_2_output_data.npz | 
| No Face Detected |  Scene-Densenet161 | Densenet_Emotiw_lr001_wzq_NoFace.py | model_1_1_densenet_New_data_NoFace.pt  | model_output_data_Densenet_No_Face.npz  |  
| All |  Hybird Network Model_A | 1_FaceAttention_AlignedModel_pre_cross_vit_wzq_onlyone_crossvit.py | model_5_1_All_New_data |model5_output_data_onecrossvit.npz |  



# 注：

1. MTCNN 模型的实现改编自[this](https://github.com/TropComplique/mtcnn-pytorch).

2. SphereFace 模型（用于对齐模型）的实现改编自 [this](https://github.com/clcarwin/sphereface_pytorch).

3. Openpose的代码实现改编自 [this](https://github.com/CMU-Perceptual-Computing-Lab/openpose).

4. 整体网络框架部分参考改编自[this](https://github.com/vlgiitr/Group-Level-Emotion-Recognition).
