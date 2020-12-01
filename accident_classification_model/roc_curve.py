from accident_classification_model.accident_classification_data import x_data
from accident_classification_model.roc_curve_knn_accident_classification_function import knn_model_function
from accident_classification_model.roc_curve_svm_accident_classification_function import svm_model_function
from accident_classification_model.roc_curve_gbdt_lr_accident_classification_function import gbdt_lr_model_function
from accident_classification_model.roc_curve_adaboost_accident_classification_function import adaboost_model_function
import os


father_path = os.path.abspath('..\\..\\..') + 'Calculations\\'
result_directory = father_path + 'fault_classification\\'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

knn_model_function(result_directory, x_data)
svm_model_function(result_directory, x_data)
gbdt_lr_model_function(result_directory, x_data)
adaboost_model_function(result_directory, x_data)

print("done...")
