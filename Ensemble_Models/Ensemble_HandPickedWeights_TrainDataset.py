import numpy as np

label_to_name = {0: 'Negative',
                 1: 'Neutral',
                 2: 'Positive'}

path_output_data = '../ModelOutputs/fourteen_models_output_data.npz'

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.

    Rows are scores for each class. 
    Columns are predictions (samples).
    """
    scoreMatExp = np.exp(np.asarray(x))

    result = np.zeros((scoreMatExp.shape[0], 3))

    result[:, 0] = scoreMatExp[:, 0] / scoreMatExp.sum(1)
    result[:, 1] = scoreMatExp[:, 1] / scoreMatExp.sum(1)
    result[:, 2] = scoreMatExp[:, 2] / scoreMatExp.sum(1)

    return result

data = np.load(path_output_data)
model1_train = data['output_train_model1']
model2_train = data['output_train_model2']
model3_train = data['output_train_model3']
model4_train = data['output_train_model4']
model5_train = data['output_train_model5']
model6_train = data['output_train_model6']
model7_train = data['output_train_model7']
model8_train = data['output_train_model8']
model9_train = data['output_train_model9']
model10_train = data['output_train_model10']
model11_train = data['output_train_model11']
model12_train = data['output_train_model12']
model13_train = data['output_train_model13']
model14_train = data['output_train_model14']
model1_val = data['output_valid_model1']
model2_val = data['output_valid_model2']
model3_val = data['output_valid_model3']
model4_val = data['output_valid_model4']
model5_val = data['output_valid_model5']
model6_val = data['output_valid_model6']
model7_val = data['output_valid_model7']
model8_val = data['output_valid_model8']
model9_val = data['output_valid_model9']
model10_val = data['output_valid_model10']
model11_val = data['output_valid_model11']
model12_val = data['output_valid_model12']
model13_val = data['output_valid_model13']
model14_val = data['output_valid_model14']
model1_eval = data['output_test_model1']
model2_eval = data['output_test_model2']
model3_eval = data['output_test_model3']
model4_eval = data['output_test_model4']
model5_eval = data['output_test_model5']
model6_eval = data['output_test_model6']
model7_eval = data['output_test_model7']
model8_eval = data['output_test_model8']
model9_eval = data['output_test_model9']
model10_eval = data['output_test_model10']
model11_eval = data['output_test_model11']
model12_eval = data['output_test_model12']
model13_eval = data['output_test_model13']
model14_eval = data['output_test_model14']
train_label = data['label_train']
valid_label = data['label_valid']
eval_label = data['label_test']
print('Dataset Loaded')

#
# m1 = -1
# m2 = 5
# m3 = 10
# m4 = 10
# m5 = 1
# m6 = 5
# m7 = 2.5
# m8 = 5

m1 = 1
m2 = 1
m3 = 1
m4 = 1
m5 = 1
m6 = 1
m7 = 1
m8 = 1

output_train = m1 * model1_train + m2 * model2_train + m3 * model3_train + m4 * model4_train + \
               m5 * model5_train + m6 * model6_train + m7 * model7_train + m8 * model8_train

output_val = m1 * model1_val + m2 * model2_val + m3 * model3_val + m4 * model4_val + m5 * model5_val + \
             m6 * model6_val + m7 * model7_val + m8 * model8_val

output_eval = m1 * model1_eval + m2 * model2_eval + m3 * model3_eval + m4 * model4_eval + \
              m5 * model5_eval + m6 * model6_eval + m7 * model7_eval + m8 * model8_eval


pred_train = np.argmax(output_train, axis=1)
pred_val = np.argmax(output_val, axis=1)
pred_eval = np.argmax(output_eval, axis=1)

correct_train = np.sum(pred_train == train_label)
correct_val = np.sum(pred_val == valid_label)
correct_eval = np.sum(pred_eval == eval_label)

train_accuracy = correct_train / output_train.shape[0]
val_accuracy = correct_val / output_val.shape[0]
eval_accuracy = correct_eval / output_eval.shape[0]

print(train_accuracy)
print(val_accuracy)
print(eval_accuracy)
