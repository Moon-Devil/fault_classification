from accident_classification_model.accident_classification_data import x_data
from anomally_detection_model.DBN_GAN_anomaly_detection_function import *


node_visible = 27
node_hidden = 500
learning_rate = 0.01
training_epochs = 20
inner_epochs = 10

mse0 = DBN(x_data[0], x_data[3], node_visible, node_hidden, learning_rate, training_epochs, inner_epochs, 'w+')
mse1 = DBN(x_data[1], x_data[3], node_visible, node_hidden, learning_rate, training_epochs, inner_epochs, 'a')
mse2 = DBN(x_data[2], x_data[3], node_visible, node_hidden, learning_rate, training_epochs, inner_epochs, 'a')
mse4 = DBN(x_data[4], x_data[3], node_visible, node_hidden, learning_rate, training_epochs, inner_epochs, 'a')

print(str(mse0) + '\t' + str(mse1) + '\t' + str(mse2) + '\t' + str(mse4))
print('done...')
