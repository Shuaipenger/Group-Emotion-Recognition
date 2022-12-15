# Group_Emotion_Recognition_HMM
Models_GAF3.0：在GAF3.0数据集上训练的代码文件。

Models_wzq_dataset：在我们的数据集上训练的代码文件。

MTCNN：提取人脸图片的相关文件。

My_ModelOutputs：在GAF3.0数据库上训练后保存的模型结果文件、准确率复现代码和提取模型结果的代码文件。

My_modelOutputs_wzq_dataset：在我们数据库上训练后的模型结果文件、准确率复现代码和提取模型结果的代码文件。

TrainedModels：保存的模型文件。

Visualization：混淆矩阵可视化。

# 安装环境

The following dependencies are required by the repository:

+ Python v3.8.12
+ PyTorch v1.10.0
+ TorchVision v0.11.1
+ NumPy
+ SciPy
+ Scikit-Learn
+ Matplotlib
+ PIL
+ Pickle



# 结果复现
1.使用代码文件训练并保存为模型文件

2.使用模型文件提取npz文件（使用My_Model_OutputSaver_TrainDataset.py代码提取npz）      

3.①用npz文件做混淆矩阵（Confusion_matrix.py）②复现准确率（Acc_Reappearance.py）③用npz文件做Ensemble调比例（GirdSearch.py）

|   | 代码 | 模型文件  | 保存的npz结果文件  |  
|---|------|---|---|
|  Model0 | Model0_AlignedModelTrainerSoftmax_AlignedModel_EmotiW_lr01_Softmax.py | AlignedModelTrainerSoftmax_AlignedModel_EmotiW_lr01_Softmax-shiyan  |  model0_output_data |   |   |
|  Model1 | Model1_Densenet_Emotiw_PretrainEmotiC_lr001.py     |model_1_2_densenet_emotiw_pretrainemotic_lr001.pt | model1_output_data  |   
|  Model2 | Model2_resnet18_emotiW.py                          |model_2_2_resnet18_EmotiW   | model2_output_data  |   
|  Model3 | Model3_skeletons.py | model_3_1_DenseNet161_skeletons_model1  | model3_output_data  |   
|  Model4 | Model4_skeletons_EfficientNet.py  | EfficientNet_skeletons  |model4_1_output_data   |  
|  Model5 | Model5_FaceAttention_AlignedModel_pre_cross_vit.py  |FaceAttention_AlignedModel_FullTrain_lr001_dropout_BN_SoftmaxLr01   |model5_output_data   |  
