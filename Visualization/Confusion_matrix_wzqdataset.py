import numpy as np
import matplotlib.pyplot as pl
from sklearn import metrics

path_output_data1 = '../My_modelOutputs_wzq_dataset/model6_1_output_data.npz'
path_output_data2 = '../My_modelOutputs_wzq_dataset/model_output_data_Densenet_No_Face.npz'

data1 = np.load(path_output_data1)
data2 = np.load(path_output_data2)

model1_val = data1['output_valid_model6_1']
model2_val = data2['output_valid_model']

model1_test = data1['output_test_model6_1']
model2_test = data2['output_test_model']

valid_label1 = data1['label_valid']
test_label1 = data1['label_test']

valid_label2 = data2['label_valid']
test_label2 = data2['label_test']

print('Dataset Loaded')

single = 0
Ensemble = 1

if single :
    pred_val1 = np.argmax(model1_val, axis=1)
    pred_test1 = np.argmax(model1_test, axis=1)

#
    y_pred = pred_test1
    y_true = test_label1

elif Ensemble :
    model_val =  np.concatenate((model1_val , model2_val))
    valid_label = np.concatenate((valid_label1 , valid_label2))
    pred_val = np.argmax(model_val, axis=1)

    model_test = np.concatenate((model1_test , model2_test))
    test_label = np.concatenate((test_label1 , test_label2))
    pred_eval = np.argmax(model_test, axis=1)

    y_pred = pred_eval
    y_true = test_label


def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.8, axis_labels=None):
# 利用sklearn中的函数生成混淆矩阵并归一化
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

# 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
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
    pl.xticks(num_local, axis_labels, rotation=0)  # 将标签印在x轴坐标上， 并倾斜45度
    pl.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    pl.ylabel('True label')
    pl.xlabel('Predicted label')

# 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                pl.text(j, i, format(float(cm[i][j] * 100 ), '.2f') + '%',
                        ha="center", va="center", fontsize ="16",
                        color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
# 显示
    pl.show()

plot_matrix(y_true, y_pred, [0, 1], title='',axis_labels=['Negative', 'Positive'])
