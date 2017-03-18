import tensorflow as tf
import config as c
from load_data import load_data
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import multiprocessing


class LSTM(object):

    def __init__(self, is_training, config):

        self.batch_size = batch_size = np.int32(config.max_group_size)
        self.num_steps = num_steps = config.num_steps
        self.num_features = num_features = config.num_features
        self.dense_units = dense_units = config.dense_units
        size = np.int32(config.hidden_size)

        print 'Type of batch_size: {}'.format(type(batch_size))
        print 'Type of num_units: {}'.format(type(size))

        if is_training:
            msg = 'Initiating input tensors of shape: {}'.format((num_steps, batch_size, num_features))
            print msg

        self._input_data = inputs = tf.placeholder(tf.float32, [num_steps, batch_size, num_features])
        self._targets = tf.placeholder(tf.float32, [num_steps, batch_size])

        # lstm_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=size, input_size=[batch_size, num_steps, num_features])
        # lstm_cell = tf.nn.rnn_cell.GRUCell(num_units=size, input_size=[batch_size, num_steps, num_features])
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=size, forget_bias=0.0)

        print 'Using RNN Cell: {}'.format(type(lstm_cell))

        # Wrap the cell in a dropout layer
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,
                                                      output_keep_prob=config.keep_prob)

        # Creates a stacked model with num_layers number of lstm cells
        # Output of first layer is input of second and so on
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

        # Initialize state as 2D Tensor of shape [batch_size x state_size] filled with zeros
        # State size is total number of units between all cells (i.e. size * num_layers)
        self._initial_state = stacked_cell.zero_state(batch_size=tf.constant(batch_size), dtype=tf.float32)

        inputs = [tf.squeeze(input, [0]) for input in tf.split(0, num_steps, inputs)]

        # Computes dropout
        if is_training and config.keep_prob < 1:
            inputs = [tf.nn.dropout(x, config.keep_prob) for x in inputs]

        # Builds the RNN
        outputs, state = tf.nn.rnn(stacked_cell, inputs, initial_state=self._initial_state)

        # Joins all output tensors and creates shape with width 'size'
        output = tf.reshape(tf.concat(1, outputs), shape=[-1, size])

        # Add a fully-connected layer
        self.dense_w = dense_w = tf.get_variable('dense_w', shape=[size, dense_units])
        self.dense_b = dense_b = tf.get_variable('dense_b', shape=[dense_units])

        # Feed the output from the RNN to the fully-connected layer with no activation function
        # The sum of all the outputs is the predicted target
        self._predictions = predictions = tf.matmul(output, dense_w) + dense_b
        self._predictions = predictions = tf.reshape(tf.reduce_sum(self.predictions, 1), shape=[num_steps, batch_size])

        numerator = tf.reduce_sum(tf.square(tf.sub(self.targets, self.predictions)))
        denominator = tf.reduce_sum(tf.square(tf.sub(self.targets, tf.reduce_mean(self.targets))))
        self.r2 = r2 = tf.sub(1.0, tf.div(numerator, denominator))

        # MSE cost function
        self._cost = cost = tf.reduce_mean(tf.square(tf.sub(self.targets, self.predictions)))
        self._final_state = state

        # Variable for state for when saving model
        self.save_state = tf.Variable(tf.zeros([batch_size, size * config.num_layers * 2]))
        self.save_state.assign(state)

        if is_training:

            self._lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(t_list=tf.gradients(cost, tvars),
                                              clip_norm=config.max_grad_norm)

            optimizer = tf.train.AdamOptimizer(self.lr)
            # optimizer = tf.train.GradientDescentOptimizer(self.lr)

            msg = 'Optimizer: {}'.format(type(optimizer))
            print msg

            self._train_op = optimizer.apply_gradients(grads_and_vars=zip(grads, tvars))

            # Summaries for Tensorboard
            cost_summ = tf.scalar_summary('mean squared error', cost)
            r2_summ = tf.scalar_summary('r-squared', r2)
            state_summ = tf.histogram_summary('states', state)
            pred_summ = tf.histogram_summary('predictions', predictions)
            self.summary = tf.merge_all_summaries()

        else:
            self.summary = predictions

        return

    # Set new learning rate
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def save_model(self, session, save_path):
        saver = tf.train.Saver([self.save_state, self.dense_w, self.dense_b])
        saver.save(session, save_path + '/Model1')

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


def run_epoch(session, m, data, eval_op, verbose=False):

    """ Runs the model on the given data """

    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    predictions = []
    state = m.initial_state.eval()
    data.prepBatches(m.batch_size, m.num_steps)

    for step, (x, y) in enumerate(data):

        targets, state, summary,  _ = session.run(fetches=[m.predictions, m.final_state, m.summary, eval_op],
                                                feed_dict={m.input_data: x,
                                                           m.targets: y,
                                                           m.initial_state: state})

        timesteps = [m.num_steps * step + i for i in range(m.num_steps)]
        pads = []
        for timestep in timesteps:
            try:
                pads.append(data.pads[timestep])
            except IndexError:
                pads.append(0)

        targets = [target[:,:-pad] if pad != 0 else target for pad, target in zip(pads, np.split(targets, m.num_steps))]
        targets = [np.reshape(target, -1).tolist() for target in targets]

        preds = []
        for target in targets:
            preds += target

        if len(data) - len(predictions) < len(preds):
            diff = len(data) - len(predictions)
            predictions += preds[0:diff]
        else:
            predictions += preds

    mse = np.mean(np.square(np.subtract(predictions, data.targets)))
    return predictions, mse, summary


def get_config(num_features, num_epochs, debug=False):

  """ Retrieve model configurations from config.py """

  if debug:
    return c.TestConfig(num_features, num_epochs)
  else:
    return c.ProductionConfig(num_features, num_epochs)


def run_model(train_data, test_data, id1_train, id1_test, num_epochs, NUM_CORES, debug=False):

    config = get_config(train_data.num_features, num_epochs, debug)
    config.max_group_size = train_data.max_group_size

    msg = 'Initiating the model with the following params:\n'
    msg += '\n'.join('{0} = {1}'.format(paramName, paramValue) for paramName, paramValue in config.params().iteritems())
    print msg

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES)

    with tf.Graph().as_default(), tf.Session(config=tf_config) as session:

        initializer = tf.random_uniform_initializer(-(1/np.sqrt(config.num_features)), 1/np.sqrt(config.num_features))
        print 'Initializer: {}'.format(tf.random_uniform_initializer)

        with tf.variable_scope('model', reuse=None, initializer=initializer):
            m = LSTM(True, config=config)

        with tf.variable_scope('model', reuse=True, initializer=initializer):
            mtest = LSTM(False, config=config)

        log_path  = train_data.config['OutputInSample'][:train_data.config['OutputInSample'].rfind('/')]
        writer = tf.train.SummaryWriter(log_path, session.graph)
        tf.initialize_all_variables().run()

        for epoch in range(num_epochs):

            # Decay the learning rate (currently not used when using Adam Optimizer)
            lr_decay = config.lr_decay ** max(epoch - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)
            current_lr = session.run(m.lr)

            predictions, mse, summary = run_epoch(session, m, train_data, m.train_op)
            r2 = compute_r2(train_data.targets, predictions)
            writer.add_summary(summary, epoch)

            msg = 'Epoch: {} - learning rate: {:.3f} - train mse: {:.3f}e-03 - train r2: {:.3f}'.format(
                epoch, current_lr, mse * 10**3, r2)
            print msg

        writer.close()
        m.save_model(session, log_path)

        ytrain_pred, _, _ = run_epoch(session, mtest, train_data, tf.no_op())
        msg = 'Train r2: {:.3f}'.format(compute_r2(train_data.targets, ytrain_pred))
        print msg

        ytest_pred, _, _ = run_epoch(session, mtest, test_data, tf.no_op())
        msg = 'Test r2: {:.3f}'.format(compute_r2(test_data.targets, ytest_pred))
        print msg

        pd.DataFrame({'id1': id1_train.id1.values,
                      '{}_prediction'.format(train_data.config['Target']): ytrain_pred},
                     index=train_data.X.index).to_csv(train_data.config['OutputInSample'])

        pd.DataFrame({'id1': id1_test.id1.values,
                      '{}_prediction'.format(train_data.config['Target']): ytest_pred},
                     index=test_data.X.index).to_csv(test_data.config['OutputOutSample'])

        msg = 'Wrote predictions to csv'
        print msg


def compute_r2(ytrue, ypred):

    """ Compute the r2 value """

    numerator = np.sum(np.square(np.subtract(ytrue, ypred)))
    denominator = np.sum(np.square(np.subtract(ytrue, np.mean(ytrue))))

    return 1 - (numerator / denominator)


def main(debug=False):

    warnings.filterwarnings(action='ignore')

    NUM_CORES = 48

    # Do not buffer the output (needed to get the full log file)
    unbuffered = os.fdopen(sys.stdout.fileno(), 'w', 0)
    sys.stdout = unbuffered

    try:
        available_cores = multiprocessing.cpu_count()
    except:
        available_cores = 'unknown'
    
    print 'Initiating model at {}'.format(datetime.now())
    print 'Model is using {} CPUs out of {} available'.format(NUM_CORES, available_cores)

    train_data, test_data, id1_train, id1_test = load_data(debug=debug)
    run_model(train_data, test_data, id1_train, id1_test, num_epochs=50, NUM_CORES=NUM_CORES, debug=debug)