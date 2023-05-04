import numpy as np
import matplotlib.pyplot as pl
from sklearn import metrics


path_output_data1 = '../My_modelOutputs_wzq_dataset/model1_Scene_Densenet_output_data.npz'
path_output_data2 = '../My_modelOutputs_wzq_dataset/model2_Scene_resnet_output_data.npz'
path_output_data3 = '../My_modelOutputs_wzq_dataset/model3_Skeleton_densenet_output_data.npz'
path_output_data4 = '../My_modelOutputs_wzq_dataset/model4_Skeleton_Efficient_output_data.npz'
path_output_data5 = '../My_modelOutputs_wzq_dataset/model5_output_data_onecrossvit.npz'

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
train_label = data5['label_train']
valid_label = data5['label_valid']
eval_label = data5['label_test']


print('Dataset Loaded')

single = 0
Ensemble = 1

if single :
    pred_val = np.argmax(model5_val, axis=1)
    pred_eval = np.argmax(model2_eval, axis=1)

    y_pred = pred_val
    y_true = valid_label

elif Ensemble :
    m1 = 0.3
    m2 = 0.25
    m3 = 0
    m4 = 0.45
    m5 = 0

    output_val = m1 * model1_val + m2 * model2_val + m3 * model3_val + m4 * model4_val + m5 * model5_val
    output_eval = m1 * model1_eval + m2 * model2_eval + m3 * model3_eval + m4 * model4_eval + m5 * model5_eval

    pred_val = np.argmax(output_val, axis=1)
    pred_eval = np.argmax(output_eval, axis=1)

    y_pred = pred_eval
    y_true = eval_label


def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.8, axis_labels=None):
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": 24,
    }
    pl.rcParams.update(config)
    # 利用sklearn中的函数生成混淆矩阵并归一化
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    pl.figure(figsize=(7, 5.8))
    pl.imshow(cm, interpolation='nearest', cmap=pl.get_cmap('Blues'))
    pl.colorbar()  # 绘制图例
    pl.clim(0, 1)

    # 图像标题
    if title is not None:
        pl.title(title)
    # 绘制坐标

    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    pl.xticks(num_local, axis_labels, rotation=0, fontsize=24)  # 将标签印在x轴坐标上， 并倾斜45度
    pl.yticks(num_local, axis_labels, rotation=90, fontsize=24)  # 将标签印在y轴坐标上
    pl.ylabel('True label', fontsize=24)
    pl.xlabel('Predicted label', fontsize=24)

    # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                pl.text(j, i, format(float(cm[i][j] * 100), '.2f') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
    # 显示
    pl.show()

plot_matrix(y_true, y_pred, [0, 1], title='',axis_labels=['Negative', 'Positive'])
