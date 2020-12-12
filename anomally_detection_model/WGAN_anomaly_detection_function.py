from anomally_detection_model.anomaly_detection_data_1000 import x_data_set
import tensorflow as tf
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
import os
tf.enable_eager_execution()


scale_model = MinMaxScaler()
x_data_scale = scale_model.fit_transform(x_data_set)
x_train_data = x_data_scale[:1000, ]
x_test_data = np.vstack((x_data_scale[1000:2000, ], x_data_scale[2000:3000, ], x_data_scale[4000:5000, ]))

BATCH_SIZE = 32
learning_rate = 0.0001

father_path = os.path.abspath('..\\..\\..') + 'Calculations\\'
result_directory = father_path + 'anomaly_detection\\'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)


class Generator(tf.keras.models.Model):
    def __init__(self, input_nodes=1, output_nodes=1):
        super(Generator, self).__init__()
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_layer_1 = tf.keras.layers.Dense(self.input_nodes, activation=tf.keras.activations.relu)
        self.hidden_layer_2 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        self.hidden_layer_3 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        self.hidden_layer_4 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        self.output_layer = tf.keras.layers.Dense(self.output_nodes)

    def build(self, input_shape, **kwargs):
        super(Generator, self).build(input_shape)

    def call(self, x, **kwargs) -> object:
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.hidden_layer_3(x)
        x = self.hidden_layer_4(x)
        x = self.output_layer(x)

        return x


class Discriminator(tf.keras.models.Model):
    def __init__(self, input_nodes=1, output_nodes=1):
        super(Discriminator, self).__init__()
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_layer_1 = tf.keras.layers.Dense(self.input_nodes, activation=tf.keras.activations.relu)
        self.hidden_layer_2 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        self.hidden_layer_3 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        self.hidden_layer_4 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        self.output_layer = tf.keras.layers.Dense(self.output_nodes)

    def build(self, input_shape, **kwargs):
        super(Discriminator, self).build(input_shape)

    def call(self, x, **kwargs) -> object:
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.hidden_layer_3(x)
        x = self.hidden_layer_4(x)
        x = self.output_layer(x)

        return x


class Encoder(tf.keras.models.Model):
    def __init__(self, input_nodes=1, output_nodes=1):
        super(Encoder, self).__init__()
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_layer_1 = tf.keras.layers.Dense(self.input_nodes, activation=tf.keras.activations.relu)
        self.hidden_layer_2 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        self.hidden_layer_3 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        self.hidden_layer_4 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        self.output_layer = tf.keras.layers.Dense(self.output_nodes)

    def build(self, input_shape, **kwargs):
        super(Encoder, self).build(input_shape)

    def call(self, x, **kwargs) -> object:
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.hidden_layer_3(x)
        x = self.hidden_layer_4(x)
        x = self.output_layer(x)

        return x


def gradient_penalty_loss(gradient, gradient_penalty_weight) -> object:
    gradient_square = tf.keras.backend.square(gradient)
    gradient_sum = tf.keras.backend.sum(gradient_square, axis=np.arange(1, len(gradient_square.shape)))
    gradient_mean = tf.keras.backend.mean(gradient_sum)
    gradient_l2_norm = tf.keras.backend.sqrt(gradient_mean)
    gradient_penalty = gradient_penalty_weight * tf.keras.backend.square(1 - gradient_l2_norm)

    return gradient_penalty


def discriminator_loss(g_model, d_model, x_true, x_sample) -> object:
    for layer in d_model.layers:
        layer.trainable = True

    for layer in g_model.layers:
        layer.trainable = False

    d_model.trainable = True
    g_model.trainable = False

    generator_value = g_model(x_sample)
    random_value = tf.keras.backend.random_uniform(shape=[BATCH_SIZE, 1], maxval=1.0, minval=-1.0)
    discriminator_inputs = random_value * x_true + (1 - random_value) * generator_value
    discriminator_true = d_model(x_true)

    with tf.GradientTape() as tape:
        tape.watch(discriminator_inputs)
        discriminator_fake = d_model(discriminator_inputs)
    gradient = tape.gradient(discriminator_fake, discriminator_inputs)
    penalty_loss = gradient_penalty_loss(gradient, 10)
    loss = tf.reduce_mean(discriminator_fake) - tf.reduce_mean(discriminator_true) + penalty_loss

    return loss


def generator_loss(g_model, d_model, x_sample) -> object:
    for layer in g_model.layers:
        layer.trainable = True

    for layer in d_model.layers:
        layer.trainable = False

    g_model.trainable = True
    d_model.trainable = False

    generator_fake = g_model(x_sample)
    discriminator_fake = d_model(generator_fake)
    loss = - tf.reduce_mean(discriminator_fake)

    return loss


def train(epochs, g_model, d_model, x_train):
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)

    x_train_slice = tf.data.Dataset.from_tensor_slices(x_train)
    x_train_batches = tf.data.Dataset.batch(x_train_slice, batch_size=BATCH_SIZE, drop_remainder=True)

    d_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9)

    result_document = result_directory + 'WGAN_train_result.txt'
    if os.path.exists(result_document):
        os.remove(result_document)

    x_sample_shape = 0

    for epoch in np.arange(epochs):
        d_loss_sum = tf.constant(0.0)
        g_loss_sum = tf.constant(0.0)
        batches = 0

        epoch_start_time = time.time()
        for x_train_batch in x_train_batches:
            batches = batches + 1
            x_sample_shape = tf.shape(x_train_batch)
            x_sample_batch = tf.keras.backend.random_uniform(shape=x_sample_shape, maxval=1.0, minval=-1.0)

            with tf.GradientTape() as tape:
                d_loss = discriminator_loss(g_model, d_model, x_train_batch, x_sample_batch)
                d_loss_sum = d_loss_sum + d_loss
                d_loss_mean = tf.divide(d_loss_sum, batches)
            d_gradient = tape.gradient(d_loss_mean, d_model.trainable_variables)
            d_optimizer.apply_gradients(zip(d_gradient, d_model.trainable_variables))

        for i_value in np.arange(batches):
            divide_value = i_value + 1
            x_sample_batch = tf.keras.backend.random_uniform(shape=x_sample_shape, maxval=1.0, minval=-1.0)

            with tf.GradientTape() as tape:
                g_loss = generator_loss(g_model, d_model, x_sample_batch)
                g_loss_sum = g_loss_sum + g_loss
                g_loss_mean = tf.divide(g_loss_sum, divide_value)
            g_gradient = tape.gradient(g_loss_mean, g_model.trainable_variables)
            g_optimizer.apply_gradients(zip(g_gradient, g_model.trainable_variables))

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        g_predict_start = time.time()
        g_predict = g_model(x_train)
        g_mse = tf.reduce_mean(tf.sqrt(tf.square(g_predict - x_train)), axis=np.arange(len(tf.shape(g_predict))))
        g_predict_end = time.time()
        g_predict_time = g_predict_end - g_predict_start

        d_predict_start = time.time()
        d_predict = d_model(x_train)
        d_predict_end = time.time()

        d_label = tf.ones_like(d_predict)
        d_mse = tf.reduce_mean(tf.sqrt(tf.square(d_predict - d_label)), axis=np.arange(len(tf.shape(d_predict))))
        d_predict_time = d_predict_end - d_predict_start

        log = "epoch:{}\tepoch_time:{}\tloss:{}\tgenerator_loss:{}\tdiscriminator_loss:{}\tgenerator_predict_time:{}" \
              "\tdiscriminator_predict_time:{}".format(epoch, epoch_time, d_loss.numpy(), g_mse.numpy(), d_mse.numpy(),
                                                       g_predict_time, d_predict_time)
        print(log)

        if epoch == 0:
            with open(result_document, 'w+') as f:
                f.write("epoch\tepoch_time\tloss\tgenerator_loss\tdiscriminator_loss\tgenerator_predict_time\t"
                        "discriminator_predict_time\n")
                f.write(str(epoch) + '\t' + str(epoch_time) + '\t' + str(d_loss.numpy()) + '\t' + str(g_mse.numpy()) +
                        '\t' + str(d_mse.numpy()) + '\t' + str(g_predict_time) + '\t' + str(d_predict_time) + '\n')
        else:
            with open(result_document, 'a') as f:
                f.write(str(epoch) + '\t' + str(epoch_time) + '\t' + str(d_loss.numpy()) + '\t' + str(g_mse.numpy()) +
                        '\t' + str(d_mse.numpy()) + '\t' + str(g_predict_time) + '\t' + str(d_predict_time) + '\n')


def anoGan_loss(ano_model, g_model, x_true, x_fake, dimension, loss_penalty) -> object:
    generator_value = g_model(x_fake)
    loss_difference = tf.square(ano_model(x_true) - ano_model(generator_value)) * loss_penalty
    loss = tf.divide(tf.square(x_true - generator_value) + loss_difference, dimension)

    return tf.reduce_mean(loss)


def anoGan_train(epochs, g_model, ano_model, x_train, lr=0.01):
    dimension = x_train.shape[1]

    result_document = result_directory + 'anoGAN_train_result.txt'
    if os.path.exists(result_document):
        os.remove(result_document)

    for epoch in np.arange(epochs):
        epoch_start_time = time.time()
        batches = 0
        loss_sum = tf.constant(0.0)
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        x_train_slice = tf.data.Dataset.from_tensor_slices(x_train)
        x_train_batches = tf.data.Dataset.batch(x_train_slice, batch_size=BATCH_SIZE, drop_remainder=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        for x_train_batch in x_train_batches:
            batches = batches + 1
            x_sample_shape = tf.shape(x_train_batch)
            x_sample_batch = tf.keras.backend.random_uniform(shape=x_sample_shape, maxval=1.0, minval=-1.0)
            with tf.GradientTape() as tape:
                loss = anoGan_loss(ano_model, g_model, x_train_batch, x_sample_batch, dimension, 10)
                loss_sum = loss_sum + loss
                loss_mean = tf.divide(loss_sum, batches)
            gradient = tape.gradient(loss_mean, ano_model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, ano_model.trainable_variables))
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        log = "epoch:{}\tepoch_time:{}\tloss:{}\t".format(epoch, epoch_time, loss.numpy())
        print(log)

        if epoch == 0:
            with open(result_document, 'w+') as f:
                f.write("epoch\tepoch_time\tloss\n")
                f.write(str(epoch) + '\t' + str(epoch_time) + '\t' + str(loss.numpy()) + '\n')
        else:
            with open(result_document, 'a') as f:
                f.write(str(epoch) + '\t' + str(epoch_time) + '\t' + str(loss.numpy()) + '\n')


def predict(ano_model, x_true):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    x_true = tf.convert_to_tensor(x_true, dtype=tf.float32)
    x_fake = tf.keras.backend.random_uniform(shape=[3000, 27], maxval=1.0, minval=-1.0)
    x_predict_true = ano_model(x_true)
    x_predict_fake = ano_model(x_fake)

    predict_start_time = time.time()
    x_predict_true = x_predict_true.numpy()
    x_predict_fake = x_predict_fake.numpy()

    x_predict_true_format = np.array([format(x_predict_true[i_value][0], '.4f') for i_value in
                                      np.arange(x_predict_true.shape[0])])
    x_predict_fake_format = np.array([format(x_predict_fake[i_value][0], '.4f') for i_value in
                                      np.arange(x_predict_fake.shape[0])])
    predict_end_time = time.time()
    predict_time = predict_end_time - predict_start_time

    flag = x_predict_true[0][0]
    for i_value in np.arange(len(x_predict_true_format)):
        if x_predict_true[i_value][0] == flag:
            TP = TP + 1
        else:
            TN = TN + 1

    for i_value in np.arange(len(x_predict_fake_format)):
        if x_predict_fake[i_value][0] == flag:
            FP = FP + 1
        else:
            FN = FN + 1

    result_document = result_directory + 'anoGAN_discriminate_result.txt'
    if os.path.exists(result_document):
        os.remove(result_document)

    with open(result_document, 'w+') as f:
        f.write("TP\tTN\tFP\tFN\n")
        f.write(str(TP) + '\t' + str(TN) + '\t' + str(FP) + '\t' + str(FN) + '\n')
        f.write("predict_time\t" + str(predict_time) + "\n")
