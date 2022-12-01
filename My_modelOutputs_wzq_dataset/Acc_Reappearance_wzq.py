import numpy as np
import matplotlib.pyplot as pl
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

path_output_data1 = '../My_modelOutputs_wzq_dataset/model5_output_data_onecrossvit.npz'


data1 = np.load(path_output_data1)


model1_train = data1['output_train_model5']

model1_val = data1['output_valid_model5']

model1_test = data1['output_test_model5']

#  Label is  the same sequence, labels are generic
train_label = data1['label_train']
valid_label = data1['label_valid']
test_label = data1['label_test']

print('Dataset Loaded')
single = 1
Ensemble = 0

if single:
    pred_train = np.argmax(model1_train, axis=1)
    pred_val = np.argmax(model1_val, axis=1)
    pred_test = np.argmax(model1_test, axis=1)

    correct_train = np.sum(pred_train == train_label)
    correct_val = np.sum(pred_val == valid_label)
    correct_test = np.sum(pred_test == test_label)

    train_accuracy = correct_train / model1_train.shape[0]
    val_accuracy = correct_val / model1_val.shape[0]
    test_accuracy = correct_test / model1_test.shape[0]

    print('Train Acc: {:.4f}'.format(train_accuracy))
    print('Val Acc: {:.4f}'.format(val_accuracy))
    print('Test Acc: {:.4f}'.format(test_accuracy))

elif Ensemble:
    m1 = 0.3
    m2 = 0
    m3 = 0
    m4 = 0.2
    m5 = 0.5

    output_train = m1 * model1_train + m2 * model2_train + m3 * model3_train + m4 * model4_train + m5 * model5_train
    output_val = m1 * model1_val + m2 * model2_val + m3 * model3_val + m4 * model4_val + m5 * model5_val

    pred_train = np.argmax(output_train, axis=1)
    pred_val = np.argmax(output_val, axis=1)

    correct_train = np.sum(pred_train == train_label)
    correct_val = np.sum(pred_val == valid_label)

    train_accuracy = correct_train / output_train.shape[0]
    val_accuracy = correct_val / output_val.shape[0]

    print('Train Acc: {:.4f}'.format(train_accuracy))
    print('Val Acc: {:.4f}'.format(val_accuracy))



