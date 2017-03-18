import tensorflow as tf
import numpy as np


class LSTM(object):

    def __init__(self, is_training, args):

        self.batch_size = batch_size = np.int32(args['batch_size'])
        self.max_steps = max_steps = np.int32(args['max_group_size'])
        self.num_features = num_features = args['num_features']
        self.dense1 = dense1 = 30
        self.dense2 = dense2 = 15
        self.dense3 = dense3 = 1
        self.hidden_size = size = np.int32(args['hidden_size'])

        if self.hidden_size < 10:
            print 'hidden_size should not be less than 10 -- setting to 10'
            self.hidden_size = size = 10

        if is_training:
            print 'Initiating input tensors of shape: {}'.format((batch_size, max_steps, num_features))

        self._input_data = inputs = tf.placeholder(tf.float32, [batch_size, max_steps, num_features])
        self._targets = tf.placeholder(tf.float32, [batch_size, max_steps])

        # Get sequence lengths (model will run the timesteps for each instrument up to the sequence length)
        lengths = [self.getSequenceLength(input) for input in tf.split(0, batch_size, inputs)]
        lengths = tf.squeeze(tf.pack(lengths), [1])

        # Memory cell to use in model
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=size, forget_bias=0.0, state_is_tuple=True)

        if is_training:
            print 'Memory Cell: {}'.format(type(lstm_cell))

        # Wrap the memory cell in a dropout layer (for outputs)
        if is_training and args['keep_prob'] < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=args['keep_prob'])

        # Create the RNN with 'num_layers' layers
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * args['num_layers'], state_is_tuple=True)

        # Initialize the state -- it will hold the last output, h_t, as well as the memory state, c_t
        # Shape will be [batch_size, num_units x 2] -- splitting on dimension 1 will separate output and memory
        self._initial_state = stacked_cell.zero_state(batch_size=tf.constant(batch_size), dtype=tf.float32)

        # Computes dropout for inputs
        if is_training and args['keep_prob'] < 1:
            inputs = tf.nn.dropout(inputs, args['keep_prob'])

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
                    self.dense_w1 = dense_w1 = tf.get_variable('dense_w1', shape=[size, dense1])
                    self.dense_b1 = dense_b1 = tf.get_variable('dense_b1', shape=[dense1])
                    self.dense_w2 = dense_w2 = tf.get_variable('dense_w2', shape=[dense1, dense2])
                    self.dense_b2 = dense_b2 = tf.get_variable('dense_b2', shape=[dense2])
                    self.dense_w3 = dense_w3 = tf.get_variable('dense_w3', shape=[dense2, dense3])
                    self.dense_b3 = dense_b3 = tf.get_variable('dense_b3', shape=[dense3])
            else:
                with tf.variable_scope('dense_layers', reuse=True):
                    self.dense_w1 = dense_w1 = tf.get_variable('dense_w1', shape=[size, dense1])
                    self.dense_b1 = dense_b1 = tf.get_variable('dense_b1', shape=[dense1])
                    self.dense_w2 = dense_w2 = tf.get_variable('dense_w2', shape=[dense1, dense2])
                    self.dense_b2 = dense_b2 = tf.get_variable('dense_b2', shape=[dense2])
                    self.dense_w3 = dense_w3 = tf.get_variable('dense_w3', shape=[dense2, dense3])
                    self.dense_b3 = dense_b3 = tf.get_variable('dense_b3', shape=[dense3])

            output1 = tf.tanh(tf.matmul(output, dense_w1) + dense_b1)
            output2 = tf.tanh(tf.matmul(output1, dense_w2) + dense_b2)
            prediction = tf.matmul(output2, dense_w3) + dense_b3

            predictions.append(prediction)

        # Bring all of the predictions together from a list of tensors of shape [max_steps]
        # to one large tensor of shape [batch_size, max_steps]
        self._predictions = predictions = tf.squeeze(tf.pack(predictions), [2])

        # Compute the R^2
        numerator = tf.reduce_sum(tf.square(tf.sub(self.targets, self.predictions)))
        denominator = tf.reduce_sum(tf.square(tf.sub(self.targets, tf.reduce_mean(self.targets))))
        self.r2 = r2 = tf.sub(1.0, tf.div(numerator, denominator))

        # MSE cost function
        self._cost = cost = tf.reduce_mean(tf.square(tf.sub(self.targets, self.predictions)))
        self._final_state = state

        # Variable for state (for when saving model)
        self.save_state = tf.Variable(tf.zeros([args['num_layers'], 2, batch_size, size]))
        self.save_state.assign(state)

        if is_training:

            self._lr = tf.Variable(0.0, trainable=False)

            # Compute the gradients
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(t_list=tf.gradients(cost, tvars),
                                              clip_norm=args['max_grad_norm'])

            # Adjust the parameters based on optimizer and gradients
            optimizer = tf.train.AdamOptimizer(self.lr)
            self._train_op = optimizer.apply_gradients(grads_and_vars=zip(grads, tvars))

            # Summaries for Tensorboard
            cost_summ = tf.scalar_summary('mean squared error', cost)
            r2_summ = tf.scalar_summary('r-squared', r2)
            # state_summ = tf.histogram_summary('states', state)
            # pred_summ = tf.histogram_summary('predictions', predictions)
            self.summary = tf.merge_all_summaries()

        else:
            # Ignore this -- put here so errors are prevented when running model not in training mode
            self.summary = predictions

        return

    @staticmethod
    def getSequenceLength(inputs):
        used = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def save_model(self, session, save_path):
        saver = tf.train.Saver([self.save_state, self.dense_w1, self.dense_b1,
                                self.dense_w2, self.dense_b2, self.dense_w3, self.dense_b3])
        saver.save(session, save_path + '/model')

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

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