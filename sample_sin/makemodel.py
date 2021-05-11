import numpy as np
import tensorflow as tf



x = np.arange(-1, 1, 0.01)
y = np.sin(-1*np.pi*x)

model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(8, activation='relu',input_dim=1 ),
                tf.keras.layers.Dense(4, activation='relu'),
                tf.keras.layers.Dense(1, activation='linear'),
            ]
        )
optm = tf.keras.optimizers.SGD(lr=0.1)
model.compile(loss='mean_squared_error', optimizer=optm)

model.fit(x, y, epochs=60, verbose=0)
model.save('model.h5')


