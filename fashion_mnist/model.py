import tensorflow as tf

model = tf.keras.Sequential()

model.add(tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28, )))
model.add(tf.keras.layers.Conv2D(32, (5, 5)))
model.add(tf.keras.layers.BatchNormalization(3))
model.add(tf.keras.layers.Activation(tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D((2, 2, )))
model.add(tf.keras.layers.Conv2D(64, (5, 5)))
model.add(tf.keras.layers.BatchNormalization(3))
model.add(tf.keras.layers.Activation(tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D((2, 2, )))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation(tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(512))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation(tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(512))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation(tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(512))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation(tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

