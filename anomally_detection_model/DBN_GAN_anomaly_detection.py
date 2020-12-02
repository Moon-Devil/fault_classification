from accident_classification_model.accident_classification_data import x_data
from anomally_detection_model.DBN_GAN_anomaly_detection_function import *
import os
import time

x_data = x_data[0][:5000, ]
start_time = time.time()

nn_model = Neural_network_model(x_data)
nn_model.create()
nn_model.train(5000)






# n_visible = 27
# n_hidden = 27
# rbm = RBM_model(x_train, n_visible, n_hidden)
#
# for epoch in range(training_epochs):
#     avg_cost = 0
#     cost, output = rbm.get_reconstruction_cost()
#     with tf.Session() as sess:
#         cost = sess.run(cost)
#         avg_cost = np.mean(cost)
#     persistent_chain = tf.cast(tf.zeros([batch_size, n_hidden]), dtype=tf.float32)
#     train_ops = rbm.get_train_ops(learning_rate=learning_rate, k=20, persistent=persistent_chain)
#
#     if epoch % display_step == 0:
#         print("Epoch {0} cost: {1}".format(epoch, avg_cost))
#
# end_time = time.time()
# training_time = end_time - start_time
# print("Finished!")
# print("The training ran for {0} minutes.".format(training_time/60,))

print('done...')
