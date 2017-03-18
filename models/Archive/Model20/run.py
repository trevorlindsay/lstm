from lstm import LSTM
import tensorflow as tf
import numpy as np


def run_model(args, train_data, test_data, id1_train, id1_test, num_epochs, NUM_CORES, debug=False, final_run=False):

    # Set the max_group_size (which will end up being the batch size)
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
        train_data.prepBatches(m.num_steps)
        test_data.prepBatches(m.num_steps)

        tf.initialize_all_variables().run()

        for epoch in range(num_epochs):

            # Decay the learning rate (begins decaying at config.decay_epoch)
            lr_decay = args['lr_decay'] ** max(epoch - args['decay_epoch'], 0.0)
            m.assign_lr(session, args['learning_rate'] * lr_decay)
            current_lr = session.run(m.lr)

            # Run epoch and retrieve results
            predictions, mse, summary = run_epoch(session, m, train_data, m.train_op)
            r2 = compute_r2(np.reshape(train_data.Y.values, len(train_data.Y)), predictions)
            writer.add_summary(summary, epoch)

            if final_run:
                print 'Epoch: {} - learning rate: {:.3f} - train mse: {:.3f}e-03 - train r2: {:.5f}'.format(
                    epoch, current_lr, mse * 10 ** 3, r2)
            elif epoch % 10 == 0:
                print 'Epoch: {} - learning rate: {:.3f} - train mse: {:.3f}e-03 - train r2: {:.5f}'.format(
                    epoch, current_lr, mse * 10 ** 3, r2)

        writer.close()

        if final_run:
            m.save_model(session, save_path)

        # Train predictions
        ytrain_pred, _, _ = run_epoch(session, mtest, train_data, tf.no_op())
        print 'Train r2: {}'.format(compute_r2(np.reshape(train_data.Y.values, len(train_data.Y)), ytrain_pred))

        # Test predictions
        ytest_pred, _, _ = run_epoch(session, mtest, test_data, tf.no_op())
        print 'Test r2: {}'.format(compute_r2(np.reshape(test_data.Y.values, len(test_data.Y)), ytest_pred))

        return ytrain_pred, \
               np.reshape(train_data.Y.values, len(train_data.Y)), \
               ytest_pred, \
               np.reshape(test_data.Y.values, len(test_data.Y))


def run_epoch(session, m, data, eval_op, verbose=False):

    # List to hold all of the predictions
    predictions = []

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

        targets = results[0]
        summary = results[1]
        state_flat = results[3:]
        state = [state_flat[i : (i + 2)] for i in range(0, len(state_flat), 2)]

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
            predictions += preds[-diff:]
        else:
            predictions += preds

    # Compute mean squared error
    mse = np.mean(np.square(np.subtract(predictions, np.reshape(data.Y.values, len(data.Y)))))

    return predictions, mse, summary


def compute_r2(ytrue, ypred):

    numerator = np.sum(np.square(np.subtract(ytrue, ypred)))
    denominator = np.sum(np.square(np.subtract(ytrue, np.mean(ytrue))))

    return 1 - (numerator / denominator)