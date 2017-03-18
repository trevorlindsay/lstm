import tensorflow as tf
import numpy as np


class LSTM(object):

    def __init__(self, is_training, modelConfig, dataConfig):

        self.batch_size = batch_size = np.int32(modelConfig['BatchSize'])
        self.num_steps = num_steps = np.int32(modelConfig['StepSize'])
        self.num_features = num_features = np.int32(modelConfig['NumFeatures'])

        # Set the activation function
        activations = {'tanh': tf.tanh, 'relu': tf.nn.relu, 'sigmoid': tf.sigmoid, 'softmax': tf.nn.softmax}
        try:
            self.activation = activation = activations[modelConfig['Activation']]
        except KeyError:
            raise KeyError("The activation function {} is not recognized".format(modelConfig['Activation']))

        if num_steps == -1:
            self.num_steps = num_steps = np.int32(modelConfig['MaxGroupSize'])

        # Tensors to hold the input and target arrays
        self._input_data = inputs = tf.placeholder(tf.float32, [batch_size, num_steps, num_features])
        self._targets = tf.placeholder(tf.float32, [batch_size, num_steps])
        self._sample_weights = sample_weights = tf.placeholder(tf.float32, [batch_size, num_steps])

        # Get sequence lengths (model will run the timesteps for each instrument up to the sequence length)
        lengths = [self.getSequenceLength(input) for input in tf.split(0, batch_size, inputs)]
        lengths = tf.squeeze(tf.pack(lengths), [1])

        # Create the recurrent layers
        rnn_sizes = [int(x.strip()) for x in modelConfig['RNN'].split(',')]
        rnn_layers = [tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_size, forget_bias=0.0,
                                                   state_is_tuple=True, activation=activation) for rnn_size in rnn_sizes]

        # Wrap the recurrent layers in dropout wrappers (for outputs)
        if is_training and modelConfig['KeepProb'] < 1:
            keep_prob = modelConfig['KeepProb']
            rnn_layers = [tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob,
                                                        input_keep_prob=1.0) for cell in rnn_layers]

        # Stack the recurrent layers
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers, state_is_tuple=True)

        # Initialize the state -- it will hold the last output, h_t, as well as the memory state, c_t
        self._initial_state = stacked_cell.zero_state(batch_size=tf.constant(batch_size), dtype=tf.float32)

        # Run inputs through the RNN
        outputs, state = tf.nn.dynamic_rnn(stacked_cell, inputs, initial_state=self._initial_state,
                                          sequence_length=lengths, time_major=False)

        # Separate the outputs into individual tensors for each instrument
        outputs = [tf.squeeze(output_, [0]) for output_ in tf.split(0, batch_size, outputs)]
        predictions = []

        # Feed each output from the RNN to the fully-connected layer, one at a time
        for i, output in enumerate(outputs):

            if i == 0:

                with tf.variable_scope('dense_layers'):

                    last_layer = None
                    dense_output = None

                    for j, layer in enumerate(modelConfig['Dense'].split(',')):

                        layer = int(layer)

                        if j == 0:
                            weights = tf.get_variable('dense_w{}'.format(j), shape=[rnn_sizes[-1], layer])
                            biases = tf.get_variable('dense_b{}'.format(j), shape=[layer])
                            last_layer = layer
                        else:
                            weights = tf.get_variable('dense_w{}'.format(j), shape=[last_layer, layer])
                            biases = tf.get_variable('dense_b{}'.format(j), shape=[layer])
                            last_layer = layer

                        if j == 0 and layer != 1:
                            dense_output = activation(tf.matmul(output, weights) + biases)
                        elif j == 0 and layer == 1:
                            dense_output = tf.matmul(output, weights) + biases
                        elif layer == 1:
                            dense_output = tf.matmul(dense_output, weights) + biases
                        else:
                            dense_output = activation(tf.matmul(dense_output, weights) + biases)

            else:

                with tf.variable_scope('dense_layers', reuse=True):

                    last_layer = None
                    dense_output = None

                    for j, layer in enumerate(modelConfig['Dense'].split(',')):

                        layer = int(layer)

                        if j == 0:
                            weights = tf.get_variable('dense_w{}'.format(j), shape=[rnn_sizes[-1], layer])
                            biases = tf.get_variable('dense_b{}'.format(j), shape=[layer])
                            last_layer = layer
                        else:
                            weights = tf.get_variable('dense_w{}'.format(j), shape=[last_layer, layer])
                            biases = tf.get_variable('dense_b{}'.format(j), shape=[layer])
                            last_layer = layer

                        if j == 0 and layer != 1:
                            dense_output = activation(tf.matmul(output, weights) + biases)
                        elif j == 0 and layer == 1:
                            dense_output = tf.matmul(output, weights) + biases
                        elif layer == 1:
                            dense_output = tf.matmul(dense_output, weights) + biases
                        else:
                            dense_output = activation(tf.matmul(dense_output, weights) + biases)

            predictions.append(dense_output)

        # Pack all of the predictions together into one tensor
        self._predictions = predictions = tf.squeeze(tf.pack(predictions), [2])

        # Compute the R^2
        numerator = tf.reduce_sum(tf.square(tf.sub(self.targets, self.predictions)))
        denominator = tf.reduce_sum(tf.square(tf.sub(self.targets, tf.reduce_mean(self.targets))))
        self.r2 = r2 = tf.sub(1.0, tf.div(numerator, denominator))

        # Weighted Mean Squared Error cost function
        squared_error = tf.square(tf.sub(self.targets, self.predictions))
        cost = tf.mul(squared_error, sample_weights)
        self._cost = cost = tf.reduce_sum(cost) / tf.reduce_sum(sample_weights)

        self._final_state = state

        if is_training:

            self._lr = tf.Variable(float(modelConfig['LearningRate']), trainable=False)

            # Compute the gradients
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(t_list=tf.gradients(cost, tvars),
                                              clip_norm=int(modelConfig['MaxGradNorm']))

            # Adjust the parameters based on optimizer and gradients
            optimizer = tf.train.AdamOptimizer(self.lr)
            self._train_op = optimizer.apply_gradients(grads_and_vars=zip(grads, tvars))

            # Summaries for Tensorboard
            cost_summ = tf.scalar_summary('mean squared error', cost)
            r2_summ = tf.scalar_summary('r-squared', r2)
            self.summary = tf.merge_all_summaries()

        else:
            # Ignore this -- put here so errors are prevented when running model not in training mode
            self.summary = predictions

        return

    @staticmethod
    def getSequenceLength(inputs):
        used = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        return tf.cast(length, tf.int32)

    @staticmethod
    def save_model(session, save_path):
        saver = tf.train.Saver()
        saver.save(session, save_path + '/model')

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def sample_weights(self):
        return self._sample_weights

    @property
    def predictions(self):
        return self._predictions

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op