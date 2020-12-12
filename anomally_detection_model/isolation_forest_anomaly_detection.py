from anomally_detection_model.isolation_forest_anomaly_detection_function import *
from anomally_detection_model.anomaly_detection_data import x_data
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import time

epochs = 20

x_normal_data = np.vstack((x_data[0], x_data[1], x_data[2], x_data[4]))
y_normal = np.full(x_normal_data.shape[0], 1)
x_anomaly_data = x_data[3]
y_anomaly = np.full(x_anomaly_data.shape[0], -1)

x_data_set = np.vstack((x_normal_data, x_anomaly_data))
y_data_set = np.hstack((y_normal, y_anomaly))

for epoch in np.arange(epochs):
    train_start_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split(x_data_set, y_data_set, test_size=0.2)
    isolation_model = IsolationForest(n_estimators=1024)
    isolation_model.fit(x_train, y_train)
    train_end_time = time.time()

    train_time = train_end_time - train_start_time

    predict_start_time = time.time()
    y_predict = isolation_model.predict(x_test)
    predict_end_time = time.time()

    predict_time = predict_end_time - predict_start_time

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i_value in np.arange(y_predict.shape[0]):
        if y_test[i_value] == 1 and y_predict[i_value] == 1:
            TP = TP + 1
        elif y_test[i_value] == 1 and y_predict[i_value] == -1:
            TN = TN + 1
        elif y_test[i_value] == -1 and y_predict[i_value] == -1:
            FP = FP + 1
        elif y_test[i_value] == -1 and y_predict[i_value] == 1:
            FN = FN + 1

    log = "epoch:{}\ttrain_time:{}\tpredict_time:{}\tTP:{}\tTN:{}\tFP:{}\tFN:{}".format(epoch, train_time, predict_time,
                                                                                        TP, TN, FP, FN)
    print(log)

    if epoch == 0:
        with open(result_document, 'w+') as f:
            f.write("epoch\ttrain_time\tpredict_time\tTP\tTN\tFP\tFN\n")
            f.write(str(epoch) + '\t' + str(train_time) + '\t' + str(predict_time) + '\t' + str(TP) + '\t' + str(TN) +
                    '\t' + str(FP) + '\t' + str(FN) + '\n')
    else:
        with open(result_document, 'a') as f:
            f.write(str(epoch) + '\t' + str(train_time) + '\t' + str(predict_time) + '\t' + str(TP) + '\t' + str(TN) +
                    '\t' + str(FP) + '\t' + str(FN) + '\n')

    if epoch == epochs - 1:
        roc_curve = roc_curve_function(y_test, y_predict)
        length = len(roc_curve)
        with open(result_document, 'a') as f:
            for i_value in np.arange(length):
                f.write(str(roc_curve[i_value][0]) + "\t" + str(roc_curve[i_value][1]) + "\n")
