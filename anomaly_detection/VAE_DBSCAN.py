from VAE_function import *
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


normal_data, anomaly_data = read_data()
normal_data = normal_data.values
anomaly_data = anomaly_data.values

vae = VAE(25)

vae.fit(anomaly_data[:5000, 125: 150], 201)

normal = vae.predict(normal_data[: 1204, ], "point", "Anomaly_detection_SGTR_normal_point")
PRLL = vae.predict(anomaly_data[:1204, 0: 25], "point", "VAE_DBSCAN_SGTR_prz_liquid_predict_point")
PRSL = vae.predict(anomaly_data[:1204, 25: 50], "point", "VAE_DBSCAN_SGTR_prz_vapour_predict_point")
RCS_CL = vae.predict(anomaly_data[:1204, 50: 75], "point", "VAE_DBSCAN_SGTR_rcs_cl_predict_point")
RCS_HL = vae.predict(anomaly_data[:1204, 75: 100], "point", "VAE_DBSCAN_SGTR_rcs_hl_predict_point")
SG2L = vae.predict(anomaly_data[:1204, 100: 125], "point", "VAE_DBSCAN_SGTR_sg_2nd_predict_point")
SGTR = vae.predict(anomaly_data[:1204, 125: 150], "point", "VAE_DBSCAN_SGTR_sgtr_predict_point")


normal = np.array(normal)
model = DBSCAN(eps=0.0003, min_samples=5)
y_predict_normal = model.fit_predict(normal)

x_data = []
normal_point = []
anomaly_point = []
clear_file("VAE_DBSCAN_SGTR_normal")
clear_file("VAE_DBSCAN_SGTR_anomaly")
length = len(y_predict_normal)
for i_index in np.arange(length):
    if y_predict_normal[i_index] == 0:
        x_data.append(normal_data[i_index, ])
        record_list = normal[i_index]
        write_to_text("VAE_DBSCAN_SGTR_normal", record_list, "a+")
    else:
        record_list = normal[i_index]
        write_to_text("VAE_DBSCAN_SGTR_anomaly", record_list, "a+")

PRLL = np.array(PRLL)
PRSL = np.array(PRSL)
RCS_CL = np.array(RCS_CL)
RCS_HL = np.array(RCS_HL)
SG2L = np.array(SG2L)
SGTR = np.array(SGTR)

plt.scatter(normal[:, 0], normal[:, 1], c='#000000')
plt.scatter(PRLL[:, 0], PRLL[:, 1])
plt.scatter(PRSL[:, 0], PRSL[:, 1])
plt.scatter(RCS_CL[:, 0], RCS_CL[:, 1])
plt.scatter(RCS_HL[:, 0], RCS_HL[:, 1])
plt.scatter(SG2L[:, 0], SG2L[:, 1])
plt.scatter(SGTR[:, 0], SGTR[:, 1])
plt.show()

print("done...")
