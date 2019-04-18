from mlxtend.data import loadlocal_mnist
import tensorflow as tf


def data_input_fn(images, labels=None, batch_size=None, num_epochs=None):

    return tf.estimator.inputs.numpy_input_fn(
        x=images,
        y=labels,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=batch_size is not None and 0 < batch_size < len(images['image']),
        queue_capacity=1000,
        num_threads=4)


img, lab = loadlocal_mnist(images_path='./MNIST/train-images-idx3-ubyte',
                           labels_path='./MNIST//train-labels-idx1-ubyte')

images = {'image': img/255}
