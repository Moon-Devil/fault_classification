from DBN_function_1 import *


normal_data, anomaly_data = read_data()
normal_data = normal_data.values
anomaly_data = anomaly_data.values

rbm_1 = RBM("first_RBM", 25, 100, None)
rbm_1.fit(normal_data[: 5000, ], 21, 5)

hidden_1_out = rbm_1.predict(normal_data[: 5000, ], "line", "normal_predict_point")

rbm_2 = RBM("second_RBM", 100, 2, None)
rbm_2.fit(hidden_1_out, 21, 5)

out = rbm_1.predict(normal_data[: 5000], "point", "1_normal_predict_point")
rbm_2.predict(out, "point", "2_normal_predict_point")
out = rbm_1.predict(normal_data[: 301], "line", "1_normal_predict_curve")
rbm_2.predict(out, "line", "2_normal_predict_curve")

out = rbm_1.predict(anomaly_data[: 5000, 0: 25], "point", "1_prz_liquid_predict_point")
rbm_2.predict(out, "point", "2_prz_liquid_predict_point")
out = rbm_1.predict(anomaly_data[: 301, 0: 25], "line", "1_prz_liquid_predict_curve")
rbm_2.predict(out, "line", "2_prz_liquid_predict_curve")

out = rbm_1.predict(anomaly_data[: 5000, 25: 50], "point", "1_prz_vapour_predict_point")
rbm_2.predict(out, "point", "2_prz_vapour_predict_point")
out = rbm_1.predict(anomaly_data[: 301, 25: 50], "line", "1_prz_vapour_predict_curve")
rbm_2.predict(out, "line", "2_prz_vapour_predict_curve")

out = rbm_1.predict(anomaly_data[: 5000, 50: 75], "point", "1_rcs_cl_predict_point")
rbm_2.predict(out, "point", "2_rcs_cl_predict_point")
out = rbm_1.predict(anomaly_data[: 301, 50: 75], "line", "1_rcs_cl_predict_curve")
rbm_2.predict(out, "line", "2_rcs_cl_predict_curve")

out = rbm_1.predict(anomaly_data[: 5000, 75: 100], "point", "1_rcs_hl_predict_point")
rbm_2.predict(out, "point", "2_rcs_hl_predict_point")
out = rbm_1.predict(anomaly_data[: 301, 75: 100], "line", "1_rcs_hl_predict_curve")
rbm_2.predict(out, "line", "2_rcs_hl_predict_curve")

out = rbm_1.predict(anomaly_data[: 5000, 100: 125], "point", "1_sg_2nd_predict_point")
rbm_2.predict(out, "point", "2_sg_2nd_predict_point")
out = rbm_1.predict(anomaly_data[: 301, 100: 125], "line", "1_sg_2nd_predict_curve")
rbm_2.predict(out, "line", "2_sg_2nd_predict_curve")

out = rbm_1.predict(anomaly_data[: 5000, 125: 150], "point", "1_sgtr_predict_point")
rbm_2.predict(out, "point", "2_sgtr_predict_point")
out = rbm_1.predict(anomaly_data[: 301, 125: 150], "line", "1_sgtr_predict_curve")
rbm_2.predict(out, "line", "2_sgtr_predict_curve")
