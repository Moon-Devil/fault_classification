from sklearn.mixture import GaussianMixture
from accident_classification_model.accident_classification_data import x_data_set, y_data_set
from sklearn.model_selection import train_test_split
from accident_classification_model.Obsolete_version_GMM_accident_classification_function import *
from sklearn.model_selection import GridSearchCV
import os
from sklearn.externals import joblib


cv = 3

father_path = os.path.abspath('..\\..\\..') + 'Calculations\\'
result_directory = father_path + 'fault_classification\\'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

length = len(y_data_set)
y_data = np.zeros(length)
for i_value in np.arange(length):
    y_data[i_value] = int(y_data_set[i_value][3] + y_data_set[i_value][2] * 2 + y_data_set[i_value][1] * 4
                          + y_data_set[i_value][0] * 8)

x_train, x_test, y_train, y_test = train_test_split(x_data_set, y_data, test_size=0.2, random_state=0)

covariance_type = ['full', 'tied', 'diag', 'spherical']
parameter_distribution = {'covariance_type': covariance_type}
gmm = GaussianMixture(n_components=5)
grid_search = GridSearchCV(gmm, parameter_distribution, cv=cv, verbose=2, return_train_score=True)
history = grid_search.fit(x_train)

result_document = result_directory + 'GMM accident classification.txt'
if os.path.exists(result_document):
    os.remove(result_document)

GridSearch_record_data(result_document, history)

covariance_type_best = history.best_params_['covariance_type']
print("GMM最佳参数搜索已完成...")

gmm = GaussianMixture(n_components=5, covariance_type=covariance_type_best)
history = gmm.fit(x_train)

model_path = result_directory + '\\gmm_model'
if not os.path.exists(model_path):
    os.mkdir(model_path)

os.chdir(model_path)
if os.path.exists(model_path + 'gmm_model.m'):
    os.remove('gmm_model.m')
joblib.dump(gmm, 'gmm_model.m')

y_predict = gmm.predict(x_test)
with open(result_document, 'a') as f:
    f.write('y_true\t')
    for i_value in np.arange(len(y_test)):
        if i_value != len(y_test) - 1:
            f.write(str(int(y_test[i_value])) + ",")
        else:
            f.write(str(int(y_test[i_value])) + "\n")

    f.write('y_predict\t')
    for i_value in np.arange(len(y_predict)):
        if i_value != len(y_predict) - 1:
            f.write(str(int(y_predict[i_value])) + ",")
        else:
            f.write(str(int(y_predict[i_value])) + "\n")

