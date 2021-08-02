from VAE_function import *


normal_data, anomaly_data = read_data()
normal_data = normal_data.values
anomaly_data = anomaly_data.values

vae = VAE(25)

vae.fit(anomaly_data[: 5000, 125: 150], 201)

vae.predict(normal_data[: 1204, ], "point", "VAE_SGTR_normal_predict_point")
vae.predict(normal_data[:301, ], "line", "VAE_SGTR_PRLL_normal_predict_curve")


vae.predict(anomaly_data[: 1204, 0: 25], "point", "VAE_SGTR_prz_liquid_predict_point")
vae.predict(anomaly_data[: 301, 0: 25], "line", "VAE_SGTR_prz_liquid_predict_curve")

vae.predict(anomaly_data[: 1204, 25: 50], "point", "VAE_SGTR_prz_vapour_predict_point")
vae.predict(anomaly_data[: 301, 25: 50], "line", "VAE_SGTR_prz_vapour_predict_curve")

vae.predict(anomaly_data[: 1204, 50: 75], "point", "VAE_SGTR_rcs_cl_predict_point")
vae.predict(anomaly_data[: 301, 50: 75], "line", "VAE_SGTR_rcs_cl_predict_curve")

vae.predict(anomaly_data[: 1204, 75: 100], "point", "VAE_SGTR_rcs_hl_predict_point")
vae.predict(anomaly_data[: 301, 75: 100], "line", "VAE_SGTR_rcs_hl_predict_curve")

vae.predict(anomaly_data[: 1204, 100: 125], "point", "VAE_SGTR_sg_2nd_predict_point")
vae.predict(anomaly_data[: 301, 100: 125], "line", "VAE_SGTR_sg_2nd_predict_curve")

vae.predict(anomaly_data[: 1204, 125: 150], "point", "VAE_SGTR_sgtr_predict_point")
vae.predict(anomaly_data[: 301, 125: 150], "line", "VAE_SGTR_sgtr_predict_curve")

# vae.predict_loss(normal_data[: 1203, ], "VAE_normal_predict_loss")
# vae.predict_loss(anomaly_data[: 1203, 0: 25], "VAE_prz_liquid_predict_loss")
# vae.predict_loss(anomaly_data[: 1203, 25: 50], "VAE_prz_vapour_predict_loss")
# vae.predict_loss(anomaly_data[: 1203, 50: 75], "VAE_rcs_cl_predict_loss")
# vae.predict_loss(anomaly_data[: 1203, 75: 100], "VAE_rcs_hl_predict_loss")
# vae.predict_loss(anomaly_data[: 1203, 100: 125], "VAE_sg_2nd_predict_loss")
# vae.predict_loss(anomaly_data[: 1203, 125: 150], "VAE_sgtr_predict_loss")

