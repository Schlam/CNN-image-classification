# GRADED FUNCTION: train_mnist_conv
def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE
    class myCallbacks(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            accuracy = logs.get('accuracy')
            if(accuracy >= 0.1):
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
    return history.epoch, history.history['acc'][-1]

# GRADED FUNCTION: train_mnist_conv
def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE
    
    class myCallbacks(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(float(logs.get('accuracy')) >= 0.998):
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
