import numpy as np
import matplotlib.pyplot as pl
from sklearn import metrics


path_output_data0 = '../My_ModelOutputs/model0_output_data.npz'
path_output_data1 = '../My_ModelOutputs/model1_output_data.npz'
path_output_data2 = '../My_ModelOutputs/model2_output_data.npz'
path_output_data3 = '../My_ModelOutputs/model3_output_data.npz'
path_output_data4 = '../My_ModelOutputs/model4_1_output_data.npz'
path_output_data5 = '../My_ModelOutputs/model5_output_data.npz'

data0 = np.load(path_output_data0)
data1 = np.load(path_output_data1)
data2 = np.load(path_output_data2)
data3 = np.load(path_output_data3)
data4 = np.load(path_output_data4)
data5 = np.load(path_output_data5)

model0_val = data0['output_valid_model0']
model1_val = data1['output_valid_model1']
model2_val = data2['output_valid_model2']
model3_val = data3['output_valid_model3']
model4_val = data4['output_valid_model4']
model5_val = data5['output_valid_model5']

model0_eval = data0['output_test_model0']
model1_eval = data1['output_test_model1']
model2_eval = data2['output_test_model2']
model3_eval = data3['output_test_model3']
model4_eval = data4['output_test_model4']
model5_eval = data5['output_test_model5']

valid_label = data1['label_valid']
eval_label = data1['label_test']

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
    m2 = 0
    m3 = 0
    m4 = 0.2
    m5 = 0.5

    output_val = m1 * model1_val + m2 * model2_val + m3 * model3_val + m4 * model4_val + m5 * model5_val
    output_eval = m1 * model1_eval + m2 * model2_eval + m3 * model3_eval + m4 * model4_eval + m5 * model5_eval

    pred_val = np.argmax(output_val, axis=1)
    pred_eval = np.argmax(output_eval, axis=1)

    y_pred = pred_val
    y_true = valid_label


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

plot_matrix(y_true, y_pred, [0, 1, 2], title='',axis_labels=['Negative', 'Neutral', 'Positive'])
