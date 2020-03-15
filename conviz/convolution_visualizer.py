'''

Examine the output image from a Conv2D layer
Useful for opening the lid on a "black box" neural network.
'''
import matplotlib.pyplot as plt
import tensorflow as tf

def visulalize_convolution(test_images, indices, convolution, model=model, layers = model.layers):
    '''
    This function prints the output image from the model layers of a CNN
    '''
    fig, axs = plt.subplots(3,4)
    layer_outputs = [layer.output for layer in layers]
    activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
    for ind in indices:
        for i in range(0,4):

            output = activation_model.predict(test_images[ind].reshape(1, 28, 28, 1))[i]
            axs[0,i].imshow(output[0, : , :, convolution], cmap='viridis')
            axs[0,i].grid(False)