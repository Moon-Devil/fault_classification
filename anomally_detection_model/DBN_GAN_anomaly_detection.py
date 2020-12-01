from accident_classification_model.accident_classification_data import x_data
from anomally_detection_model.DBN_GAN_anomaly_detection_function import *
import os
import timeit

x_train = x_data[0][:5000, ]

learning_rate = 0.001
batch_size = 5000

init = tf.global_variables_initializer()

training_epochs = 20
display_step = 1
start_time = timeit.default_timer()

n_visible = 27
n_hidden = 27
rbm = RBM(x_train, n_visible, n_hidden)

for epoch in range(training_epochs):
    avg_cost = 0
    cost, output = rbm.get_reconstruction_cost()
    with tf.Session() as sess:
        cost = sess.run(cost)
        avg_cost = np.mean(cost)
    persistent_chain = tf.cast(tf.zeros([batch_size, n_hidden]), dtype=tf.float32)
    train_ops = rbm.get_train_ops(learning_rate=learning_rate, k=20, persistent=persistent_chain)

    if epoch % display_step == 0:
        print("Epoch {0} cost: {1}".format(epoch, avg_cost))

end_time = timeit.default_timer()
training_time = end_time - start_time
print("Finished!")
print("The training ran for {0} minutes.".format(training_time/60,))
