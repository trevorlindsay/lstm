from lstm import LSTM
import tensorflow as tf
import pandas as pd
import numpy as np
from collections import deque
from scipy.stats import pearsonr


def run_model(args, train_data, test_data, id1_train, id1_test, num_epochs, NUM_CORES, debug=False, final_run=False):

    # Set the max_group_size (which will end up being the max_steps)
    args['max_group_size'] = train_data.max_group_size

    msg = 'Initiating the model with the following params:\n'
    msg += '\n'.join('{0} = {1}'.format(paramName, paramValue) for paramName, paramValue in args.iteritems())
    print msg

    # Set number of cores for TensorFlow to use
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES)

    with tf.Graph().as_default(), tf.Session(config=tf_config) as session:

        # Initializer for weights and biases
        initializer = tf.random_uniform_initializer(-args['init_scale'], args['init_scale'])

        with tf.variable_scope('model', reuse=None, initializer=initializer):
            m = LSTM(True, args)

        with tf.variable_scope('model', reuse=True, initializer=initializer):
            mtest = LSTM(False, args)

        # Path to save everything
        save_path = train_data.config['OutputInSample'][:train_data.config['OutputInSample'].rfind('/')]

        # Writer for tensorboard summaries
        writer = tf.train.SummaryWriter(save_path, session.graph)

        # Prepare the data to be iterated over
        train_data.prepBatches(m.batch_size)

        tf.initialize_all_variables().run()

        for epoch in range(num_epochs):

            # Decay the learning rate (begins decaying at config.decay_epoch)
            lr_decay = args['lr_decay'] ** max(epoch - args['decay_epoch'], 0.0)
            m.assign_lr(session, args['learning_rate'] * lr_decay)
            current_lr = session.run(m.lr)

            # Run epoch and retrieve results
            predictions, mse, summary = run_epoch(session, m, train_data, m.train_op)
            r2 = compute_r2(pd.Series(train_data.Y.values[:,0]), predictions)
            writer.add_summary(summary, epoch)

            if final_run:
                print 'Epoch: {} - learning rate: {:.3f} - train mse: {:.3f}e-03 - train corr: {:.5f}'.format(
                    epoch, current_lr, mse * 10 ** 3, r2)
            elif epoch % 10 == 0:
                print 'Epoch: {} - learning rate: {:.3f} - train mse: {:.3f}e-03 - train corr: {:.5f}'.format(
                    epoch, current_lr, mse * 10 ** 3, r2)

        writer.close()

        if final_run:
            m.save_model(session, save_path)

        # Train predictions
        ytrain_pred, _, _ = run_epoch(session, mtest, train_data, tf.no_op())
        print 'Train corr: {}'.format(compute_r2(pd.Series(train_data.Y.values[:,0]), ytrain_pred))
        print 'Train r2: {}'.format(compute_r2(pd.Series(train_data.Y.values[:, 0]), ytrain_pred, pearson=False))

        # Free up memory then prep batches for testing data
        train_data.reset()
        test_data.prepBatches(m.batch_size)

        # Test predictions
        ytest_pred, _, _ = run_epoch(session, mtest, test_data, tf.no_op())
        print 'Test corr: {}'.format(compute_r2(pd.Series(test_data.Y.values[:,0]), ytest_pred))
        print 'Test r2: {}'.format(compute_r2(pd.Series(test_data.Y.values[:, 0]), ytest_pred, pearson=False))

        return ytrain_pred, \
               pd.Series(train_data.Y.values[:, 0]), \
               ytest_pred, \
               pd.Series(test_data.Y.values[:, 0])


def run_epoch(session, m, data, eval_op, verbose=False):

    # List to hold all of the predictions
    predictions = pd.Series()

    # List to hold states in model
    state = []

    # Initial state is final state from last epoch; if first epoch, initial state is zero state
    for c, h in m.initial_state:
        state.append((c.eval(), h.eval()))

    for step, (x, y) in enumerate(data):

        # Variables to fetch
        fetches = [m.predictions, m.summary, eval_op]
        for c, h in m.final_state:
            fetches.append(c)
            fetches.append(h)

        # Variables to feed
        feed_dict = {m.input_data: x, m.targets: y}
        for i, (c, h) in enumerate(m.initial_state):
            feed_dict[c], feed_dict[h] = state[i]

        results = session.run(fetches=fetches,
                              feed_dict=feed_dict)

        preds = results[0]
        summary = results[1]
        state_flat = results[3:]
        state = [state_flat[i : (i + 2)] for i in range(0, len(state_flat), 2)]

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

    # Compute mean squared error
    mse = np.mean(np.square(np.subtract(predictions, pd.Series(data.Y.values[:,0]))))

    return predictions, mse, summary


def compute_r2(ytrue, ypred, pearson=True):

    numerator = np.sum(np.square(np.subtract(ytrue, ypred)))
    denominator = np.sum(np.square(np.subtract(ytrue, np.mean(ytrue))))
    r2 = 1 - (numerator / denominator)

    return pearsonr(ytrue, ypred)[0] if pearson else r2