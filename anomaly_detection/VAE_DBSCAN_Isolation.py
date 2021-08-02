from VAE_function import *
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


def VAE_DBSCAN_Isolation_function(data, data_isolation, filename):
    clear_file(filename)
    data_length = len(data)

    for index in np.arange(data_length):
        record_list = [data[index][0], data[index][1], data_isolation[index]]
        write_to_text(filename, record_list, "a+")


def VAE_DBSCAN_Isolation_separate_function(data, data_isolation, filename1, filename2) -> int:
    clear_file(filename1)
    clear_file(filename2)
    data_length = len(data)
    count_ = 0

    for index in np.arange(data_length):
        if data_isolation[index] == 1:
            record_list = [data[index][0], data[index][1], data_isolation[index]]
            write_to_text(filename1, record_list, "a+")
        else:
            record_list = [data[index][0], data[index][1], data_isolation[index]]
            write_to_text(filename2, record_list, "a+")
            count_ += 1
    return count_


normal_data, anomaly_data = read_data()
normal_data = normal_data.values
anomaly_data = anomaly_data.values

vae = VAE(25)

vae.fit(normal_data[: 5000, ], 201)

normal = vae.predict(normal_data[: 1204, ], "point", "Anomaly_detection_SGTR_normal_point")
PRLL = vae.predict(anomaly_data[:1204, 0: 25], "point", "VAE_DBSCAN_SGTR_prz_liquid_predict_point")
PRSL = vae.predict(anomaly_data[:1204, 25: 50], "point", "VAE_DBSCAN_SGTR_prz_vapour_predict_point")
RCS_CL = vae.predict(anomaly_data[:1204, 50: 75], "point", "VAE_DBSCAN_SGTR_rcs_cl_predict_point")
RCS_HL = vae.predict(anomaly_data[:1204, 75: 100], "point", "VAE_DBSCAN_SGTR_rcs_hl_predict_point")
SG2L = vae.predict(anomaly_data[:1204, 100: 125], "point", "VAE_DBSCAN_SGTR_sg_2nd_predict_point")
SGTR = vae.predict(anomaly_data[:1204, 125: 150], "point", "VAE_DBSCAN_SGTR_sgtr_predict_point")

normal = np.array(normal)
PRLL = np.array(PRLL)
PRSL = np.array(PRSL)
RCS_CL = np.array(RCS_CL)
RCS_HL = np.array(RCS_HL)
SG2L = np.array(SG2L)
SGTR = np.array(SGTR)

model = DBSCAN(eps=0.0003, min_samples=5)
y_predict_normal = model.fit_predict(normal)

x_data = []
length = len(y_predict_normal)
for i_index in np.arange(length):
    if y_predict_normal[i_index] == 0:
        x_data.append(normal[i_index, ])

model_isolation = IsolationForest(n_estimators=5000)
history = model_isolation.fit(x_data)

normal_isolation = model_isolation.predict(normal)
PRLL_isolation = model_isolation.predict(PRLL)
PRSL_isolation = model_isolation.predict(PRSL)
RCS_CL_isolation = model_isolation.predict(RCS_CL)
RCS_HL_isolation = model_isolation.predict(RCS_HL)
SG2L_isolation = model_isolation.predict(SG2L)
SGTR_isolation = model_isolation.predict(SGTR)

plt.scatter(normal[:, 0], normal[:, 1], c=normal_isolation)
plt.show()

count = VAE_DBSCAN_Isolation_separate_function(normal, normal_isolation, "VAE_DBSCAN_Isolation_normal",
                                               "VAE_DBSCAN_Isolation_anomaly")
VAE_DBSCAN_Isolation_function(PRLL, PRLL_isolation, "VAE_DBSCAN_Isolation_PRLL")
VAE_DBSCAN_Isolation_function(PRSL, PRSL_isolation, "VAE_DBSCAN_Isolation_PRSL")
VAE_DBSCAN_Isolation_function(RCS_CL, RCS_CL_isolation, "VAE_DBSCAN_Isolation_RCS_CL")
VAE_DBSCAN_Isolation_function(RCS_HL, RCS_HL_isolation, "VAE_DBSCAN_Isolation_RCS_HL")
VAE_DBSCAN_Isolation_function(SG2L, SG2L_isolation, "VAE_DBSCAN_Isolation_SG2L")
VAE_DBSCAN_Isolation_function(SGTR, SGTR_isolation, "VAE_DBSCAN_Isolation_SGTR")

print("done...")
