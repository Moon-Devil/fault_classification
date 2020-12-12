import numpy as np
import os


father_path = os.path.abspath('..\\..\\..') + 'Calculations\\'
result_directory = father_path + 'anomaly_detection\\'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

result_document = result_directory + 'isolation_forest_result.txt'
if os.path.exists(result_document):
    os.remove(result_document)


def roc_curve_function(y_true, y_predict) -> object:
    roc_list = list(zip(y_predict, y_true))
    roc_curve = []
    x_coordinate = 0.0
    y_coordinate = 0.0
    number_positive = 0
    number_negative = 0
    length = len(roc_list)

    for i_value in np.arange(length):
        if roc_list[i_value][0] == 1 and roc_list[i_value][1] == 1:
            number_positive = number_positive + 1

        if roc_list[i_value][0] == 1 and roc_list[i_value][1] == -1:
            number_negative = number_negative + 1

    if number_negative == 0:
        roc_curve = [[0.0, 1.0], [1.0, 1.0]]
    else:
        for i_value in np.arange(length):
            if roc_list[i_value][0] == 1 and roc_list[i_value][1] == 1:
                y_coordinate = y_coordinate + 1.0 / number_positive
                roc_curve.append([x_coordinate, y_coordinate])

            if roc_list[i_value][0] == 1 and roc_list[i_value][1] == -1:
                x_coordinate = x_coordinate + 1.0 / number_negative
                roc_curve.append([x_coordinate, y_coordinate])

    return roc_curve
