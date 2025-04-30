import tensorflow as tf

class build:

    @staticmethod

    def create_model():

        model = tf.keras.Sequential([

            tf.keras.layers.InputLayer(input_shape = (28, 28, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation = 'relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation = 'relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation = 'relu'),
            tf.keras.layers.Dense(10, activation = 'softmax')
            
        ])

        model.compile(
            optimizer = 'adam',
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
        )

        return model

