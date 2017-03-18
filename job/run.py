from lstm import LSTM
import tensorflow as tf
import pandas as pd
import numpy as np
from collections import deque
from scipy.stats import pearsonr


def compute_r2(ytrue, ypred, pearson=False):

    numerator = np.sum(np.square(np.subtract(ytrue, ypred)))
    denominator = np.sum(np.square(np.subtract(ytrue, np.mean(ytrue))))
    r2 = 1 - (numerator / denominator)

    return r2 if not pearson else pearsonr(ytrue, ypred)[0]


def run_epoch(session, m, data, eval_op, verbose=False):

    # List to hold all of the predictions
    predictions = pd.Series()

    # List to hold states in model
    state = []

    # Helper variables
    preds = None
    step = 0

    # Initial state is final state from last epoch; if first epoch, initial state is zero state
    for c, h in m.initial_state:
        state.append((c.eval(), h.eval()))

    for x, y, w in data:

        # Variables to fetch
        fetches = [m.predictions, m.summary, eval_op]
        for c, h in m.final_state:
            fetches.append(c)
            fetches.append(h)

        # Variables to feed
        feed_dict = {m.input_data: x, m.targets: y, m.sample_weights: w}
        for i, (c, h) in enumerate(m.initial_state):
            feed_dict[c], feed_dict[h] = state[i]

        results = session.run(fetches=fetches,
                              feed_dict=feed_dict)

        output = results[0]
        summary = results[1]
        state_flat = results[3:]
        state = [state_flat[i : (i + 2)] for i in range(0, len(state_flat), 2)]

        # Combine all outputs until all time steps have been passed through model
        if not isinstance(preds, np.ndarray):
            preds = output
            if not preds.shape[1] == data.max_group_size:
                continue
        elif (output.shape[1] + preds.shape[1]) <= data.max_group_size:
            preds = np.concatenate((preds, output), axis=1)
            if not preds.shape[1] == data.max_group_size:
                continue
        else:
            diff = data.max_group_size - preds.shape[1]
            preds = np.concatenate((preds, output[:,-diff:]), axis=1)

        # The index position (in data.pads) of the instruments than ran through the model
        instruments = [(m.batch_size * step) + i for i in range(m.batch_size)]
        pads = deque([])
        extra_steps = 0

        # Find out where actual predictions end and those for padded values begin
        for instrument in instruments:
            try:
                pads.append(data.pads[instrument])
            except IndexError:
                # Sometimes the __iter__ function of data object has to add extra time steps as padding
                extra_steps += 1
                pads.appendleft(0)

        # Only keep actual targets and not those for padded values
        if not extra_steps:
            targets = [target[:, :-pad] if pad != 0 else target for pad, target in
                       zip(pads, np.split(preds, m.batch_size))]
        else:
            targets = []
            for i, (pad, target) in enumerate(zip(pads, np.split(preds, m.batch_size))):
                if i >= extra_steps:
                    if pad != 0:
                        targets.append(target[:, :-pad])
                    else:
                        targets.append(target)

        targets = [pd.Series(target[0]) for target in targets]

        # For final round of iteration -- makes sure not to include predictions for padded values
        for df in targets:
            predictions = predictions.append(df)

        # Reset helper variables
        step += 1
        preds = None

    # Compute weighted mean squared error
    squared_error = np.square(np.subtract(predictions, pd.Series(data.Y.values[:,0])))
    weighted_error = np.matmul(squared_error, data.weights)
    mse = np.sum(weighted_error) / np.sum(data.weights)

    return predictions, mse, summary, state


def run_model(modelConfig, dataConfig, train_data, test_data):

    modelConfig['MaxGroupSize'] = dataConfig['MaxGroupSize']

    # Set number of cores for TensorFlow to use
    try:
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=int(modelConfig['NumCores']),
                                   intra_op_parallelism_threads=int(modelConfig['NumCores']))
    except KeyError:
        raise UserWarning("Number of cores to use not specified! Setting to TensorFlow default.")

    with tf.Graph().as_default(), tf.Session(config=tf_config) as session:

        # Initializer for weights and biases
        scale = float(modelConfig['InitScale'])
        initializer = tf.random_uniform_initializer(-scale, scale)

        with tf.variable_scope('Model', reuse=None, initializer=initializer):
            m = LSTM(True, modelConfig, dataConfig)

        with tf.variable_scope('Model', reuse=True, initializer=initializer):
            mtest = LSTM(False, modelConfig, dataConfig)

        if modelConfig['OutputDirectory'] == "":
            save_path = train_data.config['OutputInSample'][:train_data.config['OutputInSample'].rfind('/')]
        else:
            save_path = modelConfig['OutputDirectory']

        # Writer for tensorboard summaries
        writer = tf.train.SummaryWriter(save_path, session.graph)

        # Prepare the data to be iterated over
        train_data.prepBatches(m.batch_size, m.num_steps)
        ytrain = pd.Series(train_data.Y.values[:, 0])

        tf.initialize_all_variables().run()

        for epoch in range(int(modelConfig['NumEpochs'])):

            # Run epoch and retrieve results
            predictions, mse, summary, state = run_epoch(session, m, train_data, m.train_op)
            r2 = compute_r2(ytrain, predictions)
            writer.add_summary(summary, epoch)

            print 'Epoch: {} - train mse: {:.3f}e-03 - train r2: {:.5f}'.format(
                epoch, mse * 10 ** 3, r2)

        writer.close()
        m.save_model(session, save_path)

        # Train predictions
        ytrain_pred, _, _, _ = run_epoch(session, mtest, train_data, tf.no_op())
        print 'Train r2: {}'.format(compute_r2(ytrain, ytrain_pred))

        # Free up memory then prep batches for testing data
        train_data.reset()
        test_data.prepBatches(m.batch_size, m.num_steps)
        ytest = pd.Series(test_data.Y.values[:, 0])

        # Test predictions
        ytest_pred, _, _, _ = run_epoch(session, mtest, test_data, tf.no_op())
        print 'Test r2: {}'.format(compute_r2(ytest, ytest_pred))

        return ytrain_pred, ytrain, ytest_pred, ytest


def load_model(modelConfig, dataConfig, data):

    # Set number of cores for TensorFlow to use
    try:
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=int(modelConfig['NumCores']),
                                   intra_op_parallelism_threads=int(modelConfig['NumCores']))
    except KeyError:
        tf_config = tf.ConfigProto()
        raise UserWarning("Number of cores to use not specified! Setting to TensorFlow default.")

    with tf.Graph().as_default(), tf.Session(config=tf_config) as session:

        with tf.variable_scope('Model', reuse=None):
            m = LSTM(False, modelConfig, dataConfig)

        saver = tf.train.Saver()
        saver.restore(session, modelConfig['InputDirectory'])
        print 'Model successfuly restored!'

        data.prepBatches(m.batch_size, m.num_steps)
        ypred, _, _, _ = run_epoch(session, m, data, tf.no_op())

    return ypred

