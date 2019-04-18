import tensorflow as tf


class BeliefNetwork:

    def __init__(self, hidden_neurons, n_classes, model_dir=None, tr_params=None, warm_start_from=None):
        self.hidden_neurons = hidden_neurons
        self.n_classes = n_classes
        self.trained_rbm = False
        distribute = tf.distribute.DistributeConfig(train_distribute=tf.distribute.ParameterServerStrategy(),
                                                    eval_distribute=tf.distribute.MirroredStrategy())
        run_config = tf.estimator.RunConfig(save_summary_steps=10,
                                            save_checkpoints_steps=100,
                                            experimental_distribute=distribute)
        self.estimator = tf.estimator.Estimator(model_fn=self._custom_model_fn_,
                                                model_dir=model_dir,
                                                config=run_config,
                                                params=tr_params,
                                                warm_start_from=warm_start_from)

    def _model_fn_(self, model_input):

        stoch_spike = tf.random_uniform(model_input['image'].shape, maxval=1, dtype=tf.float64)
        v_spike = tf.math.ceil(model_input['image'] - stoch_spike)

        if self.trained_rbm:
            with tf.variable_scope('rbm'):
                weights = tf.get_variable(name='weights')
                bias = tf.get_variable(name='bias')

                rbm_act = tf.math.sigmoid((v_spike @ weights) + bias)
                stoch_spike = tf.random_uniform(rbm_act.shape, maxval=1, dtype=tf.float64)
                rbm_spike = tf.math.ceil(rbm_act - stoch_spike)

            with tf.variable_scope('softmax'):
                softmax_layer = tf.layers.Dense(inputs=rbm_spike,
                                                units=self.n_classes)
                return softmax_layer
        else:
            return v_spike

    def _custom_model_fn_(self, model_input, labels=None, mode=None, params=None):
        model_fn = self._model_fn_(model_input)

        # -------- TRAIN Options --------- #
        if mode == tf.estimator.ModeKeys.TRAIN:

            # RBM to be trained and stacked
            if self.trained_rbm:
                with tf.variable_scope('rbm'):
                    global_step = tf.get_variable(name='rbm_global_step',
                                                  dtype=tf.int32,
                                                  initializer=tf.constant(0),
                                                  trainable=False)
                    lr = tf.train.exponential_decay(learning_rate=params['lr'],
                                                    global_step=global_step,
                                                    decay_steps=params['dec_steps'],
                                                    decay_rate=params['dec_rate'],
                                                    staircase=params['staircase'])
                    weights = tf.get_variable(name='weights',
                                              shape=(model_fn.shape[1], self.hidden_neurons),
                                              dtype=tf.float64,
                                              initializer=tf.random_normal_initializer(stddev=0.01, dtype=tf.float64),
                                              trainable=True)
                    h_bias = tf.get_variable(name='bias',
                                             shape=(1, self.hidden_neurons),
                                             dtype=tf.float64,
                                             initializer=tf.zeros_initializer,
                                             trainable=True)
                    # Computing the wake part
                    h_act = tf.math.sigmoid((model_fn @ weights) + h_bias)
                    wake = tf.expand_dims(model_fn, axis=2) @ tf.expand_dims(h_act, axis=1)

                    # Reconstructing input
                    stoch_spike = tf.random_uniform(h_act.shape, maxval=1, dtype=tf.float64)
                    h_spike = tf.math.ceil(h_act - stoch_spike)
                    v_bias = tf.get_variable(name='tmp_v_bias',
                                             shape=(1, model_fn.shape[1]),
                                             dtype=tf.float64,
                                             initializer=tf.zeros_initializer,
                                             trainable=True)
                    recon_input = tf.math.sigmoid((h_spike @ tf.transpose(weights)) + v_bias)

                    # Spiking neurons of reconstructed input
                    stoch_spike = tf.random_uniform(model_input.shape, maxval=1, dtype=tf.float64)
                    recon_spike = tf.math.ceil(recon_input - stoch_spike)

                    # Computing the dream part
                    recon_h_act = tf.math.sigmoid((recon_spike @ weights) + h_bias)
                    dream = tf.expand_dims(recon_spike, axis=2) @ tf.expand_dims(recon_h_act, axis=1)

                    err = tf.losses.mean_squared_error(model_fn, recon_spike)

                    with tf.control_dependencies([wake, dream]):
                        w_update = tf.assign_add(weights, lr * tf.reduce_mean(wake - dream, axis=0))
                        hb_update = tf.assign_add(h_bias, lr * tf.reduce_mean(h_act - recon_h_act, axis=0))
                        vb_update = tf.assign_add(v_bias, lr * tf.reduce_mean(model_fn - recon_input, axis=0))

                    with tf.control_dependencies([w_update, hb_update, vb_update]):
                        gs_update = tf.assign_add(global_step, 1)

                    return tf.estimator.EstimatorSpec(mode=mode,
                                                      loss=err,
                                                      train_op=[w_update, hb_update, vb_update, gs_update])

            # Softmax layer to be stacked and trained
            else:
                logits = self._model_fn_(model_input)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
                loss = tf.reduce_mean(cross_entropy)

                lr = tf.train.exponential_decay(learning_rate=params['lr'],
                                                global_step=tf.train.get_or_create_global_step(),
                                                decay_steps=params['dec_steps'],
                                                decay_rate=params['dec_rate'],
                                                staircase=params['staircase'])
                nn_opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=params['momentum'])
                nn_train_op = nn_opt.minimize(loss=loss,
                                              global_step=tf.train.get_or_create_global_step(),
                                              var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                         scope='softmax'))
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

        if mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL:
            predictions = {
                'classes': tf.argmax(model_fn),
                'probabilities': tf.nn.softmax(model_fn)
            }

            # -------- PREDICT Options --------- #
            if mode == tf.estimator.ModeKeys.PREDICT:
                export_outputs = {
                    'predict_output': tf.estimator.export.PredictOutput(predictions)
                }
                return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                                  predictions=predictions,
                                                  export_outputs=export_outputs)

            else:
                # -------- EVAL Options --------- #
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=model_fn)
                loss = tf.reduce_mean(cross_entropy)
                accuracy = tf.metrics.accuracy(labels=tf.cast(labels, tf.int32),
                                               predictions=tf.argmax(model_fn, axis=1),
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

    def train(self, train_data, val_data=None):
        # RBM Training
        rbm_early_stopping = tf.estimator.stop_if_no_decrease_hook(estimator=self.estimator,
                                                                   metric_name='loss',
                                                                   max_steps_without_decrease=500,
                                                                   min_steps=100)
        self.estimator.train(input_fn=train_data,
                             hooks=[out_early_stopping])
        self.trained_rbm = True

        # Output layer training
        out_early_stopping = tf.estimator.stop_if_no_decrease_hook(estimator=self.estimator,
                                                                   metric_name='loss',
                                                                   max_steps_without_decrease=500,
                                                                   min_steps=100)
        train_spec = tf.estimator.TrainSpec(input_fn=train_data,
                                            max_steps=None,
                                            hooks=[out_early_stopping])
        if val_data is not None:
            def serving_input_rec_fn():
                serving_features = {
                    'tree': tf.placeholder(shape=[None, None, self.L+3], dtype=tf.int32),
                    'limits': tf.placeholder(shape=[None, None], dtype=tf.int32)}
                return tf.estimator.export.ServingInputReceiver(features=serving_features,
                                                                receiver_tensors=serving_features)

            exporter = tf.estimator.BestExporter(name="best_exporter",
                                                 serving_input_receiver_fn=serving_input_rec_fn,
                                                 exports_to_keep=3)
            eval_spec = tf.estimator.EvalSpec(input_fn=val_data,
                                              steps=None,
                                              name='Validation',
                                              hooks=None,
                                              exporters=[exporter],
                                              start_delay_secs=120)
            tf.estimator.train_and_evaluate(estimator=self.estimator,
                                            train_spec=train_spec,
                                            eval_spec=eval_spec)
        else:
            pass