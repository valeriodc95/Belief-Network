# Belief-Network
TensorFlow implementation of a Belief Network, with the exploitation of the tf.Estimator interface.

Following the Colab notebook flow, the model first builds and trains a Restricted Boltzmann Machine with stochastic binary activation (simplified under the assumption that we can take the probability itself for each neuron).

After the training of the RBM, a softmax layer is built and trained by reloading the best RBM exported during the design cycle.
