from lstm import LSTM
import config as c
from load_data import load_data

import tensorflow as tf
import pandas as pd
import numpy as np

from datetime import datetime
import multiprocessing
import warnings
import os
import sys


def run_model(train_data, test_data, id1_train, id1_test, num_epochs, NUM_CORES, debug=False):

    # Get model configurations based on whether in debug mode
    config = get_config(train_data.num_features, num_epochs, debug)

    # Set the max_group_size (which will end up being the batch size)
    config.max_group_size = train_data.max_group_size

    msg = 'Initiating the model with the following params:\n'
    msg += '\n'.join('{0} = {1}'.format(paramName, paramValue) for paramName, paramValue in config.params().iteritems())
    print msg

    # Set number of cores for TensorFlow to use
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES)

    with tf.Graph().as_default(), tf.Session(config=tf_config) as session:

        # Initializer for weights and biases
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.variable_scope('model', reuse=None, initializer=initializer):
            m = LSTM(True, config=config)

        with tf.variable_scope('model', reuse=True, initializer=initializer):
            mtest = LSTM(False, config=config)

        # Path to save everything
        save_path = train_data.config['OutputInSample'][:train_data.config['OutputInSample'].rfind('/')]

        # Writer for tensorboard summaries
        writer = tf.train.SummaryWriter(save_path, session.graph)

        # Prepare the data to be iterated over
        train_data.prepBatches(m.num_steps)
        test_data.prepBatches(m.num_steps)

        tf.initialize_all_variables().run()

        for epoch in range(num_epochs):

            # Decay the learning rate (begins decaying at config.decay_epoch)
            lr_decay = config.lr_decay ** max(epoch - config.decay_epoch, 1.0)
            m.assign_lr(session, config.learning_rate * lr_decay)
            if epoch > config.decay_epoch: print current_lr
            current_lr = session.run(m.lr)

            # Run epoch and retrieve results
            predictions, mse, summary = run_epoch(session, m, train_data, m.train_op)
            r2 = compute_r2(np.reshape(train_data.Y.values, len(train_data.Y)), predictions)
            writer.add_summary(summary, epoch)

            print 'Epoch: {} - learning rate: {:.3f} - train mse: {:.3f}e-03 - train r2: {:.3f}'.format(
                epoch, current_lr, mse * 10 ** 3, r2)

        writer.close()
        m.save_model(session, save_path)

        # Train predictions to write to CSV
        ytrain_pred, _, _ = run_epoch(session, mtest, train_data, tf.no_op())
        print 'Train r2: {:.3f}'.format(compute_r2(np.reshape(train_data.Y.values, len(train_data.Y)), ytrain_pred))

        # Test predictions to write to CSV
        ytest_pred, _, _ = run_epoch(session, mtest, test_data, tf.no_op())
        print 'Test r2: {:.3f}'.format(compute_r2(np.reshape(test_data.Y.values, len(test_data.Y)), ytest_pred))

        # Write predictions to csv
        pd.DataFrame({'id1': id1_train.id1.values,
                      '{}_prediction'.format(train_data.config['Target']): ytrain_pred},
                     index=train_data.X.index).to_csv(train_data.config['OutputInSample'])

        pd.DataFrame({'id1': id1_test.id1.values,
                      '{}_prediction'.format(train_data.config['Target']): ytest_pred},
                     index=test_data.X.index).to_csv(test_data.config['OutputOutSample'])

        print 'Wrote predictions to csv'


def run_epoch(session, m, data, eval_op, verbose=False):

    # List to hold all of the predictions
    predictions = []

    # Initial state is final state from last epoch; if first epoch, initial state is zero state
    state = m.initial_state.eval()

    for step, (x, y) in enumerate(data):

        targets, state, summary, _ = session.run(fetches=[m.predictions, m.final_state, m.summary, eval_op],
                                                 feed_dict={m.input_data: x,
                                                            m.targets: y,
                                                            m.initial_state: state})

        # Ensures that the correct targets are added to the predictions list and in the right order
        timesteps = [m.num_steps * step + i for i in range(m.num_steps)]
        pads = []

        # Find out where actual predictions end and those for padded values begin
        for timestep in timesteps:
            try:
                pads.append(data.pads[timestep])
            except IndexError:
                pads.append(0)

        # Only keep actual targets and not those for padded values
        targets = [target[:, :-pad] if pad != 0 else target for pad, target in zip(pads, np.split(targets, m.num_steps))]
        targets = [np.reshape(target, -1).tolist() for target in targets]

        preds = []
        for target in targets:
            preds += target

        # For final round of iteration -- makes sure not to include predictions for padded values
        if len(data) - len(predictions) < len(preds):
            diff = len(data) - len(predictions)
            predictions += preds[0:diff]
        else:
            predictions += preds

    # Compute mean squared error
    mse = np.mean(np.square(np.subtract(predictions, np.reshape(data.Y.values, len(data.Y)))))

    return predictions, mse, summary


def compute_r2(ytrue, ypred):

    numerator = np.sum(np.square(np.subtract(ytrue, ypred)))
    denominator = np.sum(np.square(np.subtract(ytrue, np.mean(ytrue))))

    return 1 - (numerator / denominator)


def get_config(num_features, num_epochs, debug=False):
    return c.ProductionConfig(num_features, num_epochs) if not debug else c.TestConfig(num_features, num_epochs)


def main(debug=False):

    warnings.filterwarnings(action='ignore')

    # Do not buffer the output (needed to get the full log file)
    unbuffered = os.fdopen(sys.stdout.fileno(), 'w', 0)
    sys.stdout = unbuffered

    try:
        available_cores = multiprocessing.cpu_count()
    except:
        available_cores = None

    NUM_CORES = available_cores if available_cores else 48

    print 'Initiating model at {}'.format(datetime.now())
    print 'Model is using {} CPUs out of {} available'.format(NUM_CORES, available_cores)

    train_data, test_data, id1_train, id1_test = load_data(debug=debug)
    run_model(train_data, test_data, id1_train, id1_test, num_epochs=200, NUM_CORES=NUM_CORES, debug=debug)