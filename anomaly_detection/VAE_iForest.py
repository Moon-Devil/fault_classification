from VAE_iForest_function import *

normal_data, anomaly_data = read_data()
normal_data = normal_data.values
anomaly_data = anomaly_data.values

# "PowerR", "PRLL_accident", "PRSL_accident", "CL_LOCA_accident", "HL_LOCA_accident", "SG2L_accident", "SGTR_accident"

vae = VAE(25)
VAE_iForest_train_function(vae, "SGTR_accident", normal_data, anomaly_data)
