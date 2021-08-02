from AE_function import *


normal_data, anomaly_data = read_data()
normal_data = normal_data.values
anomaly_data = anomaly_data.values
auto_encoder = AutoEncoder(25)

auto_encoder.fit(normal_data[:5000, ], 201)

auto_encoder.predict(normal_data[:5000, ], "point", "AutoEncoder_normal_predict_point")
auto_encoder.predict(normal_data[:301, ], "line", "AutoEncoder_normal_predict_curve")

auto_encoder.predict(anomaly_data[:5000, 0: 25], "point", "AutoEncoder_prz_liquid_predict_point")
auto_encoder.predict(anomaly_data[:301, 0: 25], "line", "AutoEncoder_prz_liquid_predict_curve")

auto_encoder.predict(anomaly_data[:5000, 25: 50], "point", "AutoEncoder_prz_vapour_predict_point")
auto_encoder.predict(anomaly_data[:301, 25: 50], "line", "AutoEncoder_prz_vapour_predict_curve")

auto_encoder.predict(anomaly_data[:5000, 50: 75], "point", "AutoEncoder_rcs_cl_predict_point")
auto_encoder.predict(anomaly_data[:301, 50: 75], "line", "AutoEncoder_rcs_cl_predict_curve")

auto_encoder.predict(anomaly_data[:5000, 75: 100], "point", "AutoEncoder_rcs_hl_predict_point")
auto_encoder.predict(anomaly_data[:301, 75: 100], "line", "AutoEncoder_rcs_hl_predict_curve")

auto_encoder.predict(anomaly_data[:5000, 100: 125], "point", "AutoEncoder_sg_2nd_predict_point")
auto_encoder.predict(anomaly_data[:301, 100: 125], "line", "AutoEncoder_sg_2nd_predict_curve")

auto_encoder.predict(anomaly_data[:5000, 125: 150], "point", "AutoEncoder_sgtr_predict_point")
auto_encoder.predict(anomaly_data[:301, 125: 150], "line", "AutoEncoder_sgtr_predict_curve")
