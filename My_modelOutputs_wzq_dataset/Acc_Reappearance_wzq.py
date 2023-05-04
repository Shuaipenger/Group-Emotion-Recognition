import numpy as np
import matplotlib.pyplot as pl
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


path_output_data1 = '../My_modelOutputs_wzq_dataset/model1_0_Scene_Densenet_output_data.npz'
path_output_data2 = '../My_modelOutputs_wzq_dataset/model2_Scene_resnet_output_data.npz'
path_output_data3 = '../My_modelOutputs_wzq_dataset/model3_Skeleton_densenet_output_data.npz'
path_output_data4 = '../My_modelOutputs_wzq_dataset/model4_Skeleton_Efficient_output_data.npz'
path_output_data5 = '../My_modelOutputs_wzq_dataset/model5_2_output_data_onecrossvit.npz'

data1 = np.load(path_output_data1)
data2 = np.load(path_output_data2)
data3 = np.load(path_output_data3)
data4 = np.load(path_output_data4)
data5 = np.load(path_output_data5)

model1_train = data1['output_train_model1']
model2_train = data2['output_train_model1']
model3_train = data3['output_train_model1']
model4_train = data4['output_train_model1']
model5_train = data5['output_train_model5']

model1_val = data1['output_valid_model1']
model2_val = data2['output_valid_model1']
model3_val = data3['output_valid_model1']
model4_val = data4['output_valid_model1']
model5_val = data5['output_valid_model5']

model1_eval = data1['output_test_model1']
model2_eval = data2['output_test_model1']
model3_eval = data3['output_test_model1']
model4_eval = data4['output_test_model1']
model5_eval = data5['output_test_model5']

#  Label is  the same sequence, labels are generic
train_label = data1['label_train']
valid_label = data1['label_valid']
eval_label = data1['label_test']

print('Dataset Loaded')
single = 0
Ensemble = 1

if single:
    pred_train = np.argmax(model5_train, axis=1)
    pred_val = np.argmax(model5_val, axis=1)
    pred_eval = np.argmax(model5_eval, axis=1)

    correct_train = np.sum(pred_train == train_label)
    correct_val = np.sum(pred_val == valid_label)
    correct_eval = np.sum(pred_eval == eval_label)

    train_accuracy = correct_train / model5_train.shape[0]
    val_accuracy = correct_val / model5_val.shape[0]
    eval_accuracy = correct_eval / model5_eval.shape[0]

    print('Train Acc: {:.4f}'.format(train_accuracy))
    print('Val Acc: {:.4f}'.format(val_accuracy))
    print('Test Acc: {:.4f}'.format(eval_accuracy))

elif Ensemble:
    m1 = 0.3
    m2 = 0.25
    m3 = 0
    m4 = 0.45
    m5 = 0

    output_train = m1 * model1_train + m2 * model2_train + m3 * model3_train + m4 * model4_train + m5 * model5_train
    output_val = m1 * model1_val + m2 * model2_val + m3 * model3_val + m4 * model4_val + m5 * model5_val
    output_eval = m1 * model1_eval + m2 * model2_eval + m3 * model3_eval + m4 * model4_eval + m5 * model5_eval

    pred_train = np.argmax(output_train, axis=1)
    pred_val = np.argmax(output_val, axis=1)
    pred_eval = np.argmax(output_eval, axis=1)

    correct_train = np.sum(pred_train == train_label)
    correct_val = np.sum(pred_val == valid_label)
    correct_eval = np.sum(pred_eval == eval_label)

    train_accuracy = correct_train / output_train.shape[0]
    val_accuracy = correct_val / output_val.shape[0]
    eval_accuracy = correct_eval / output_eval.shape[0]

    print('Train Acc: {:.4f}'.format(train_accuracy))
    print('Val Acc: {:.4f}'.format(val_accuracy))
    print('Test Acc: {:.4f}'.format(eval_accuracy))





