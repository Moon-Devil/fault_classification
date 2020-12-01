import numpy as np


def GridSearch_record_data(result_document, history):
    with open(result_document, 'w+') as f:
        f.write('covariance_type\t')
        f.write(history.best_params_['covariance_type'] + '\n')
        f.write('best_params\t' + str(history.best_score_) + '\n')

        f.write('mean_fit_time\t')
        temp = history.cv_results_['mean_fit_time']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('std_fit_time\t')
        temp = history.cv_results_['std_fit_time']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('mean_score_time\t')
        temp = history.cv_results_['mean_score_time']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('std_score_time\t')
        temp = history.cv_results_['std_score_time']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('parameter_covariance_type\t')
        temp = history.cv_results_['param_covariance_type']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('mean_test_score\t')
        temp = history.cv_results_['mean_test_score']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('std_test_score\t')
        temp = history.cv_results_['std_test_score']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('mean_train_score\t')
        temp = history.cv_results_['mean_train_score']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

        f.write('std_train_score\t')
        temp = history.cv_results_['std_train_score']
        length = len(temp)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp[i_value]) + ',')
            else:
                f.write(str(temp[i_value]) + '\n')

