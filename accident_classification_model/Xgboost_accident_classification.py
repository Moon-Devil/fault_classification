from accident_classification_model.accident_classification_data import x_data_set, y_data_set
from accident_classification_model.Xgboost_accident_classification_function import *
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import os
from sklearn.externals import joblib

cv = 5

length = len(y_data_set)
y_data = np.zeros(length)
for i_value in np.arange(length):
    y_data[i_value] = int(y_data_set[i_value][3] + y_data_set[i_value][2] * 2 + y_data_set[i_value][1] * 4
                          + y_data_set[i_value][0] * 8)

x_train, x_test, y_train, y_test = train_test_split(x_data_set, y_data, test_size=0.2, random_state=0)

learning_rate = np.linspace(0.1, 1, 10)
number_estimator = np.linspace(1, 10, 10)
other_params = {
    'silent': 1, 'objective': 'multi:softmax', 'random_state': 0
}

parameter_distribution = {'learning_rate': learning_rate, 'n_estimatores': number_estimator}

xgboost = XGBClassifier(**other_params)
grid_search = GridSearchCV(xgboost, parameter_distribution, cv=cv, verbose=2, return_train_score=True)
history = grid_search.fit(x_train, y_train)

father_path = os.path.abspath('..\\..\\..') + 'Calculations\\'
result_directory = father_path + 'fault_classification\\'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

result_document = result_directory + 'Xgboost accident classification.txt'
if os.path.exists(result_document):
    os.remove(result_document)

GridSearch_record_data(result_document, history)

learning_rate = history.best_params_['learning_rate']
number_estimator = int(history.best_params_['n_estimatores'])
print("Xgboost最佳参数搜索已完成...")

xgboost = XGBClassifier(learning_rate=learning_rate, n_estimators=number_estimator, objective='multi:softmax',
                        random_state=0, silent=True)
history = xgboost.fit(x_train, y_train)

model_path = result_directory + '\\Xgboost_model'
if not os.path.exists(model_path):
    os.mkdir(model_path)

os.chdir(model_path)
if os.path.exists(model_path + 'Xgboost_model.m'):
    os.remove('Xgboost_model.m')
joblib.dump(xgboost, 'Xgboost_model.m')


y_predict = xgboost.predict(x_test)
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

print("done")
