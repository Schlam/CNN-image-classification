'''
Author: Samuel Bellenchia
CNN image classifier trained using the MNIST fashion dataset  
'''

import tensorflow as tf
print(tf.__version__)


# Custom callbacks function to end model training after 99.8% accuracy is reached
class myCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= 0.998):
            print("Desired accuracy reached!")
            self.model.stop_training = True
callbacks = myCallbacks()

# Load dataset
fashion = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = fashion.load_data()


# Reshape and normalize data
training_images = training_images.reshape(len(training_images), 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(len(test_images), 28, 28, 1)
test_images = test_images / 255.0

# Define our model
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, (3,3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Fit model to training data
history = model.fit(
    training_images,
    training_labels,
    epochs = 20,
    callbacks = [callbacks]
)

# Print results
print(history.epoch, history.history['accuracy'][-1])
