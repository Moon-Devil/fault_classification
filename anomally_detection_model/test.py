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
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss=tf.keras.losses.mape,
              metrics=[tf.keras.metrics.mean_squared_error])
model.summary()

history = model.fit(x_train, x_train, batch_size=1, epochs=5)

print(history.history['loss'])
