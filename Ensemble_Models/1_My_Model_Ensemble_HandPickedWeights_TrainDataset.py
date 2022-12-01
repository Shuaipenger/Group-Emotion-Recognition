import numpy as np

label_to_name = {0: 'Negative',
                 1: 'Neutral',
                 2: 'Positive'}

path_output_data = '../Ensemble_Models/fourteen_models_output_data.npz'
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

model1_val = data['output_valid_model1']
model2_val = data['output_valid_model2']
model3_val = data['output_valid_model3']
model4_val = data['output_valid_model4']
model5_val = data['output_valid_model5']

model1_eval = data['output_test_model1']
model2_eval = data['output_test_model2']
model3_eval = data['output_test_model3']
model4_eval = data['output_test_model4']
model5_eval = data['output_test_model5']

train_label = data['label_train']
valid_label = data['label_valid']
eval_label = data['label_test']
print('Dataset Loaded')

# m1 = 0.2
# m2 = 0.1
# m3 = 0.1
# m4 = 0.1
# m5 = 0.5

m1 = 0.12
m2 = 0.08
m3 = 0.12
m4 = 0.08
m5 = 0.6

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

print(train_accuracy)
print(val_accuracy)
print(eval_accuracy)
