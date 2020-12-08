from anomally_detection_model.anomaly_detection_data import x_data
import tensorflow as tf


x_train = x_data[0]
model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(27, activation=tf.keras.activations.sigmoid, input_shape=(27, )))
model.add(tf.keras.layers.Dense(27, activation=tf.keras.activations.relu, input_shape=(27, )))
model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(27, activation=tf.keras.activations.relu))
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mape, metrics=[tf.keras.metrics.mse])
model.summary()

history = model.fit(x_train, x_train, batch_size=5, epochs=200, validation_split=0.2,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2)])

print(history.history['loss'])
print('done')
