from accident_classification_model.accident_classification_function import *
import time
import warnings
warnings.filterwarnings("ignore")


epochs = 30
all_time = []
all_train_time = []
all_predict_time = []
all_accuracy = []
model_time = []


for epoch in np.arange(epochs):
    print('epochs=' + str(epoch) + ' start...')
    start_time = time.time()
    temp_model_time = []

    knn_label, knn_time = knn_predict_model()
    temp_model_time.append(knn_time[0])
    temp_model_time.append(knn_time[1])

    svm_label, svm_time = svm_predict_model()
    temp_model_time.append(svm_time[0])
    temp_model_time.append(svm_time[1])

    adaboost_label, adaboost_time = adaboost_predict_model()
    temp_model_time.append(adaboost_time[0])
    temp_model_time.append(adaboost_time[1])

    gbdt_lr_label, gbdt_lr_time = gbdt_lr_predict_model()
    temp_model_time.append(gbdt_lr_time[0])
    temp_model_time.append(gbdt_lr_time[1])

    xgboost_label, xgboost_time = xgboost_predict_model()
    temp_model_time.append(xgboost_time[0])
    temp_model_time.append(xgboost_time[1])

    model_time.append(temp_model_time)

    train_time = knn_time[0] + svm_time[0] + adaboost_time[0] + gbdt_lr_time[0] + xgboost_time[0]
    predict_time = knn_time[1] + svm_time[1] + adaboost_time[1] + gbdt_lr_time[1] + xgboost_time[1]

    y_label = np.vstack((knn_label, svm_label, adaboost_label, gbdt_lr_label, xgboost_label))
    y_label = np.transpose(y_label)

    label_result = find_best_classification(y_label)

    label_true = np.full(len(x_data[0][:2000, ]), 0)
    for i_value in np.arange(1, 5):
        label_true = np.hstack((label_true, np.full(len(x_data[i_value][:2000, ]), i_value)))

    accuracy = classification_accuracy(label_result, label_true)
    end_time = time.time()

    calculate_time = end_time - start_time
    all_time.append(calculate_time)
    all_train_time.append(train_time)
    all_predict_time.append(predict_time)
    all_accuracy.append(accuracy)

    print('epochs = ' + str(epoch) + ' has been done...')
    print('accuracy = ' + str(accuracy[6]))
    print('time = ' + str(calculate_time) + '\n')

father_path = os.path.abspath('..\\..\\..') + 'Calculations\\'
result_directory = father_path + 'fault_classification\\'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

result_document = result_directory + 'accident classification result.txt'
if os.path.exists(result_document):
    os.remove(result_document)

with open(result_document, 'w+') as f:
    f.write('time\t')
    for i_value in np.arange(epochs):
        if i_value != epochs - 1:
            f.write(str(all_time[i_value]) + ',')
        else:
            f.write(str(all_time[i_value]) + '\n')

    f.write('train_time\t')
    for i_value in np.arange(epochs):
        if i_value != epochs - 1:
            f.write(str(all_train_time[i_value]) + ',')
        else:
            f.write(str(all_train_time[i_value]) + '\n')

    f.write('predict_time\t')
    for i_value in np.arange(epochs):
        if i_value != epochs - 1:
            f.write(str(all_predict_time[i_value]) + ',')
        else:
            f.write(str(all_predict_time[i_value]) + '\n')

    for i_value in np.arange(epochs):
        f.write('epoch = ' + str(i_value) + '\n')
        f.write('right_number\t')
        length = 5
        for j_value in np.arange(length):
            if j_value != length - 1:
                f.write(str(all_accuracy[i_value][0][j_value]) + ',')
            else:
                f.write(str(all_accuracy[i_value][0][j_value]) + '\n')

        f.write('wrong_number\t')
        length = 5
        for j_value in np.arange(length):
            if j_value != length - 1:
                f.write(str(all_accuracy[i_value][1][j_value]) + ',')
            else:
                f.write(str(all_accuracy[i_value][1][j_value]) + '\n')

        f.write('unknown_number\t')
        length = 5
        for j_value in np.arange(length):
            if j_value != length - 1:
                f.write(str(all_accuracy[i_value][2][j_value]) + ',')
            else:
                f.write(str(all_accuracy[i_value][2][j_value]) + '\n')

        f.write('accuracy\t')
        length = 5
        for j_value in np.arange(length):
            if j_value != length - 1:
                f.write(str(all_accuracy[i_value][3][j_value]) + ',')
            else:
                f.write(str(all_accuracy[i_value][3][j_value]) + '\n')

        f.write('wrong\t')
        length = 5
        for j_value in np.arange(length):
            if j_value != length - 1:
                f.write(str(all_accuracy[i_value][4][j_value]) + ',')
            else:
                f.write(str(all_accuracy[i_value][4][j_value]) + '\n')

        f.write('unknown\t')
        length = 5
        for j_value in np.arange(length):
            if j_value != length - 1:
                f.write(str(all_accuracy[i_value][5][j_value]) + ',')
            else:
                f.write(str(all_accuracy[i_value][5][j_value]) + '\n')

        f.write('all_accuracy\t' + str(all_accuracy[i_value][6]) + '\n')
        f.write('all_wrong\t' + str(all_accuracy[i_value][7]) + '\n')
        f.write('all_unknown\t' + str(all_accuracy[i_value][8]) + '\n')

    f.write('knn_train_time\t' + 'knn_predict_time\t' +
            'svm_train_time\t' + 'svm_predict_time\t' +
            'adaboost_train_time\t' + 'adaboost_predict_time\t' +
            'gbdt_train_time\t' + 'gbdt_predict_time\t' +
            'xgboost_train_time\t' + 'xgboost_predict_time\n')

    column = len(model_time)
    row = len(model_time[0])

    for i_value in np.arange(column):
        for j_value in np.arange(row):
            if j_value != row - 1:
                f.write(str(model_time[i_value][j_value]) + '\t')
            else:
                f.write(str(model_time[i_value][j_value]) + '\n')

print("done...")
