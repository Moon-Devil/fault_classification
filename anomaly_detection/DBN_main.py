from DBN_function import *

normal_data, anomaly_data = read_data()
normal_data = normal_data.values
anomaly_data = anomaly_data.values

rbm_1 = RBM(25, 100)
rbm_2 = RBM(100, 2)
rbm_1.fit("RBM_first_train_curve_SGTR", anomaly_data[: 1204, 125: 150], 1001, 10)
activate_h = rbm_1.predict(normal_data[: 5000, ], "train", None)
rbm_2.fit("RBM_second_train_curve_SGTR", activate_h, 1001, 10)


# activate_h = rbm_1.predict(normal_data[:5000, ], "train", None)
# rbm_2.predict(activate_h, "point", "DBM_normal_predict_point")
# rbm_1.predict(normal_data[:301, ], "line", "DBM_normal_predict_curve")
#
# activate_h = rbm_1.predict(anomaly_data[:5000, 0: 25], "train", None)
# rbm_2.predict(activate_h, "point", "DBM_prz_liquid_predict_point")
# rbm_1.predict(anomaly_data[:301, 0: 25], "line", "DBM_prz_liquid_predict_curve")
#
# activate_h = rbm_1.predict(anomaly_data[:5000, 25: 50], "train", None)
# rbm_2.predict(activate_h, "point", "DBM_prz_vapour_predict_point")
# rbm_1.predict(anomaly_data[:301, 25: 50], "line", "DBM_prz_vapour_predict_curve")
#
# activate_h = rbm_1.predict(anomaly_data[:5000, 50: 75], "train", None)
# rbm_2.predict(activate_h, "point", "DBM_rcs_cl_predict_point")
# rbm_1.predict(anomaly_data[:301, 50: 75], "line", "DBM_rcs_cl_predict_curve")
#
# activate_h = rbm_1.predict(anomaly_data[:5000, 75: 100], "train", None)
# rbm_2.predict(activate_h, "point", "DBM_rcs_hl_predict_point")
# rbm_1.predict(anomaly_data[:301, 75: 100], "line", "DBM_rcs_hl_predict_curve")
#
# activate_h = rbm_1.predict(anomaly_data[:5000, 100: 125], "train", None)
# rbm_2.predict(activate_h, "point", "DBM_sg_2nd_predict_point")
# rbm_1.predict(anomaly_data[:301, 100: 125], "line", "DBM_sg_2nd_predict_curve")
#
# activate_h = rbm_1.predict(anomaly_data[:5000, 125: 150], "train", None)
# rbm_2.predict(activate_h, "point", "DBM_sgtr_predict_point")
# rbm_1.predict(anomaly_data[:301, 125: 150], "line", "DBM_sgtr_predict_curve")

rbm_1.predict_mse(normal_data[: 1204, ], "DBM_1_SGTR_normal_predict_MSE")
rbm_1.predict_mse(anomaly_data[: 1204, 0: 25], "DBM_1_SGTR_prz_liquid_predict_MSE")
rbm_1.predict_mse(anomaly_data[: 1204, 25: 50], "DBM_1_SGTR_prz_vapour_predict_MSE")
rbm_1.predict_mse(anomaly_data[: 1204, 50: 75], "DBM_1_SGTR_rcs_cl_predict_MSE")
rbm_1.predict_mse(anomaly_data[: 1204, 75: 100], "DBM_1_SGTR_rcs_hl_predict_MSE")
rbm_1.predict_mse(anomaly_data[: 1204, 100: 125], "DBM_1_SGTR_sg_2nd_predict_MSE")
rbm_1.predict_mse(anomaly_data[: 1204, 125: 150], "DBM_1_SGTR_sgtr_predict_MSE")
