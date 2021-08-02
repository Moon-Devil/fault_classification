from GAN_function import *


normal_data, anomaly_data = read_data()
normal_data = normal_data.values
anomaly_data = anomaly_data.values
gan = GAN(25)

gan.fit(normal_data[:5000, ], 201)

gan.predict(normal_data[:5000, ], "point", "GAN_normal_predict_point")
gan.predict(normal_data[:301, ], "line", "GAN_normal_predict_curve")

gan.predict(anomaly_data[:5000, 0: 25], "point", "GAN_prz_liquid_predict_point")
gan.predict(anomaly_data[:301, 0: 25], "line", "GAN_prz_liquid_predict_curve")

gan.predict(anomaly_data[:5000, 25: 50], "point", "GAN_prz_vapour_predict_point")
gan.predict(anomaly_data[:301, 25: 50], "line", "GAN_prz_vapour_predict_curve")

gan.predict(anomaly_data[:5000, 50: 75], "point", "GAN_rcs_cl_predict_point")
gan.predict(anomaly_data[:301, 50: 75], "line", "GAN_rcs_cl_predict_curve")

gan.predict(anomaly_data[:5000, 75: 100], "point", "GAN_rcs_hl_predict_point")
gan.predict(anomaly_data[:301, 75: 100], "line", "GAN_rcs_hl_predict_curve")

gan.predict(anomaly_data[:5000, 100: 125], "point", "GAN_sg_2nd_predict_point")
gan.predict(anomaly_data[:301, 100: 125], "line", "GAN_sg_2nd_predict_curve")

gan.predict(anomaly_data[:5000, 125: 150], "point", "GAN_sgtr_predict_point")
gan.predict(anomaly_data[:301, 125: 150], "line", "GAN_sgtr_predict_curve")
