from tensorflow import keras
from keras.optimizers import RMSprop
import tensorflow as tf
print(tf.__version__)

# Custom callback function
class myCallbacks(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')==None):
      print("nonetype")
    elif(logs.get('acc')>0.998):
      print("\nAchieved 99.8% Accuracy!")
      self.model.stop_training = True

callbacks = myCallbacks()

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (4,4), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.Conv2D(32, (4,4), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  #tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(training_images, training_labels, epochs=20, callbacks=[callbacks])
test_loss = model.evaluate(test_images, test_labels)

print(test_loss)


# Excercis notebook


# GRADED FUNCTION: train_mnist_conv
def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE
    !pip install tensorflow==2.0.0-alpha0
    class myBabyGotCallbacks(tf.keras.callbacks.Callback):
        df on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')==None):
                print("none")
            elif(logs.get('acc')>0.998):
                print("Reached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True
    callbacks = myCallbacks()
    # YOUR CODE ENDS HERE

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)
    # YOUR CODE STARTS HERE
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0
    test_images = test_images.reshape(60000, 28, 28, 1)
    test_images = test_images / 255.0
    # YOUR CODE ENDS HERE

    model = tf.keras.models.Sequential([
            # YOUR CODE STARTS HERE
        tf.keras.layers.Conv2D(8, (3,3), activation='relu' input_shape = (28, 28, 1)),
        tf.keras.layers.Conv2D(8, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(8, (3,3), activation = 'relu'),
        tf.keras.layers.Flatten()
        tf.keras.layers.Dense(128, activation = 'relu')
        tf.keras.layers.Dense(10, activation = 'softmax')
            # YOUR CODE ENDS HERE
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model fitting
    history = model.fit(
        # YOUR CODE STARTS HERE
        training_images, training_labels, epochs=20, callbacks=[callbacks]
        # YOUR CODE ENDS HERE
    )
    # model fitting
    return history.epoch, history.history['acc'][-1]




# GRADED FUNCTION: train_mnist_conv
def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE
    
    class myCallbacks(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy') >= 0.998):
                print("Reached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True
    callbacks = myCallbacks()
    
    # YOUR CODE ENDS HERE

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)
    # YOUR CODE STARTS HERE
    
    training_images = training_images.reshape(len(training_images), 28, 28, 1)
    training_images = training_images / 255.0
    test_images = test_images.reshape(len(test_images), 28, 28, 1)
    test_images = test_images / 255.0
    
    # YOUR CODE ENDS HERE

    model = tf.keras.models.Sequential([
            # YOUR CODE STARTS HERE
        
            tf.keras.layers.Conv2D(8, (3,3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        
            # YOUR CODE ENDS HERE
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model fitting
    history = model.fit(
        # YOUR CODE STARTS HERE
    
        training_images, training_labels, epochs = 20, callbacks = [callbacks]
        
        # YOUR CODE ENDS HERE
    )
    # model fitting
    return history.epoch, history.history['accuracy'][-1]
