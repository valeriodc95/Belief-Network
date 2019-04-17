import tensorflow as tf


class DeepBeliefNetwork:

    def __init__(self, hidden_layers, n_classes):
        self.n_rbm = len(hidden_layers)
        self.n_hidden = hidden_layers
        self.n_classes = n_classes
        self.n_trained_rbm = 0

    def _model_fn_(self, model_input):
        curr_fn = model_input

        for i in range(self.n_trained_rbm):
            with tf.variable_scope('rbm_'+str(i)):
                weights = tf.get_variable(name='weights',
                                          shape=(curr_fn.shape[1], self.n_hidden[i]),
                                          dtype=tf.float64,
                                          initializer=tf.random_uniform_initializer,
                                          trainable=True)
                bias = tf.get_variable(name='bias',
                                       shape=(1, self.n_hidden),
                                       dtype=tf.float64,
                                       initializer=tf.zeros_initializer,
                                       trainable=True)
                curr_fn = tf.math.sigmoid((curr_fn @ weights) + bias, name='out')

        if self.n_trained_rbm < self.n_rbm:
            return curr_fn
        else:
            with tf.variable_scope('softmax'):
                softmax_layer = tf.layers.Dense(inputs=curr_fn,
                                                units=self.n_classes)
                return softmax_layer


    def _custom_model_fn_(self, model_input, labels=None, mode=None, params=None):

        # -------- PREDICT Options --------- #
        if mode == tf.estimator.ModeKeys.PREDICT:
            logits = self._model_fn_(model_input)
            predictions = {
                'classes': tf.argmax(logits),
                'probabilities': tf.nn.softmax(logits)
            }
            export_outputs = {
                'predict_output': tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                              predictions=predictions,
                                              export_outputs=export_outputs)

        # -------- TRAIN Options --------- #
        if mode == tf.estimator.ModeKeys.TRAIN:
            if self.n_trained_rbm < self.n_rbm:
                h_i =
            else:
                logits = self._model_fn_(model_input)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
                loss = tf.reduce_mean(cross_entropy)
                lr = tf.train.exponential_decay(learning_rate=params['lr'],
                                                global_step=tf.train.get_or_create_global_step(),
                                                decay_steps=params['dec_steps'],
                                                decay_rate=params['dec_rate'],
                                                staircase=params['staircase'])

                nn_opt = tf.train.MomentumOptimizer(learning_rate=params['lr'], momentum=params['momentum'])
                nn_train_op = nn_opt.minimize(loss=loss,
                                              global_step=tf.train.get_or_create_global_step(),
                                              var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                         scope='neural_network'))
                accuracy = tf.metrics.accuracy(labels=tf.cast(labels, tf.int32),
                                               predictions=tf.argmax(logits, axis=1),
                                               name='accuracy')
                logging_hook = tf.train.LoggingTensorHook({"loss": loss,
                                                           "accuracy": accuracy}, every_n_iter=1)
                tf.summary.scalar('train_loss', loss)
                tf.summary.scalar('train_accuracy', accuracy)

                return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN,
                                                  loss=loss,
                                                  train_op=nn_train_op,
                                                  training_hooks=[logging_hook])

        # -------- EVAL Options --------- #
        if mode == tf.estimator.ModeKeys.EVAL:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            loss = tf.reduce_mean(cross_entropy)
            accuracy = tf.metrics.accuracy(labels=tf.cast(labels, tf.int32),
                                           predictions=tf.argmax(logits, axis=1),
                                           name='accuracy')
            eval_metric = {
                'accuracy': accuracy
            }
            logging_hook = tf.train.LoggingTensorHook({"loss": loss,
                                                       "accuracy": accuracy}, every_n_iter=1)
            tf.summary.scalar('eval_loss', loss)
            tf.summary.scalar('eval_accuracy', accuracy)
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions,
                                              loss=loss,
                                              eval_metric_ops=eval_metric,
                                              evaluation_hooks=[logging_hook])