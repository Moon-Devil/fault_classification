from anomally_detection_model.WGAN_anomaly_detection_function import *


generator = Generator(input_nodes=x_train_data.shape[1], output_nodes=x_train_data.shape[1])
generator.build(input_shape=(None, x_train_data.shape[1]))
generator.summary()

discriminator = Discriminator(input_nodes=x_train_data.shape[1], output_nodes=1)
discriminator.build(input_shape=(None, x_train_data.shape[1]))
discriminator.summary()

anoGan = Encoder(input_nodes=x_train_data.shape[1], output_nodes=1)
anoGan.build(input_shape=(None, x_train_data.shape[1]))
anoGan.summary()

train(200, generator, discriminator, x_train_data)
anoGan_train(200, generator, anoGan, x_train_data)
predict(anoGan, x_train_data)

print('done...')
