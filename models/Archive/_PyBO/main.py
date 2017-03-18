from pybo import solve_bayesopt

import numpy as np
import pandas as pd

from load_data import load_data
from run import run_model

from datetime import datetime
from collections import deque
import cPickle as pickle
import os
import sys
import warnings


def objective(x):

    global train_data, test_data, id1_train, id1_test, DEBUG, NUM_CORES, NUM_EPOCHS

    args = {'num_layers': int(x[0]),
            'hidden_size': int(x[1]),
            'learning_rate': x[2],
            'keep_prob': x[3],
            'num_steps': int(x[4]),
            'init_scale': x[5],
            'max_grad_norm': int(x[6]),
            'decay_epoch': NUM_EPOCHS,
            'lr_decay': 1.0,
            'num_features': train_data.num_features}

    _, _, ypred, ytrue = run_model(args, train_data, test_data, id1_train, id1_test, NUM_EPOCHS, NUM_CORES, DEBUG)
    return compute_mse(ytrue, ypred)


def compute_r2(ytrue, ypred):

    numerator = np.sum(np.square(np.subtract(ytrue, ypred)))
    denominator = np.sum(np.square(np.subtract(ytrue, np.mean(ytrue))))

    return 1 - (numerator / denominator)


def compute_mse(ytrue, ypred):
    return np.mean(np.square(np.subtract(ypred, ytrue)))


def main(debug=False):

    warnings.filterwarnings('ignore')
    unbuffered = os.fdopen(sys.stdout.fileno(), 'w', 0)
    sys.stdout = unbuffered

    print 'Initiating hyperopt at {}'.format(datetime.now())

    global train_data, test_data, id1_train, id1_test, DEBUG, NUM_CORES, NUM_EPOCHS

    DEBUG = debug
    NUM_CORES = 48
    NUM_EPOCHS = 50
    train_data, test_data, id1_train, id1_test = load_data(debug=DEBUG)
    num_features = train_data.num_features

    bounds = [
        [2, 4], # num_layers
        [10, num_features], # hidden_size
        [0.001, 0.01], # learning_rate
        [0.5, 1.0], # keep_prob
        [1, 1], # num_steps
        [0.0, 0.05], # init_scale
        [15, 15] # max_grad_norm
    ]

    xbest, model, info = solve_bayesopt(objective, bounds, niter=100, verbose=True)
    run_best_model(xbest)

    save_path = train_data.config['OutputInSample'][:train_data.config['OutputInSample'].rfind('/')]
    pickle.dump(info, open(save_path + '/trials', 'wb'))


def run_best_model(x):

    global train_data, test_data, id1_train, id1_test, DEBUG, NUM_CORES, NUM_EPOCHS

    args = {'num_layers': int(x[0]),
            'hidden_size': int(x[1]),
            'learning_rate': x[2],
            'keep_prob': x[3],
            'num_steps': int(x[4]),
            'lr_decay': x[5],
            'init_scale': x[6],
            'max_grad_norm': int(x[7]),
            'decay_epoch': 200,
            'num_features': train_data.num_features}

    ytrain_pred, ytrain_true, ytest_pred, ytest_true = run_model(args,
                                                                 train_data,
                                                                 test_data,
                                                                 id1_train,
                                                                 id1_test,
                                                                 200,
                                                                 NUM_CORES,
                                                                 DEBUG,
                                                                 final_run=True)

    # Write predictions to csv
    pd.DataFrame({'id1': id1_train.id1.values,
                  '{}_prediction'.format(train_data.config['Target']): ytrain_pred},
                 index=train_data.X.index).to_csv(train_data.config['OutputInSample'])

    pd.DataFrame({'id1': id1_test.id1.values,
                  '{}_prediction'.format(train_data.config['Target']): ytest_pred},
                 index=test_data.X.index).to_csv(test_data.config['OutputOutSample'])

    print 'Wrote predictions to csv'

def data_transformation_test():

    warnings.filterwarnings('ignore')
    unbuffered = os.fdopen(sys.stdout.fileno(), 'w', 0)
    sys.stdout = unbuffered

    num_steps = 1

    train_data, test_data, id1_train, id1_test = load_data(debug=True)
    train_data.prepBatches(num_steps)

    predictions = pd.Series()

    # This section of code comes from the run_epoch function in run.py
    # Normally, x and y are fed into the model then the model's predictions are
    # what this section of code handles
    for step, (x, y) in enumerate(train_data):

        # The index position (in data.pads) of the timesteps than ran through the model
        timesteps = [(num_steps * step) + i for i in range(num_steps)]
        pads = deque([])
        extra_steps = 0

        # Find out where actual predictions end and those for padded values begin
        for timestep in timesteps:
            try:
                pads.append(train_data.pads[timestep])
            except IndexError:
                # Sometimes the __iter__ function of data object has to add extra time steps as padding
                extra_steps += 1
                pads.appendleft(0)

        # Only keep actual targets and not those for padded values
        if not extra_steps:
            targets = [target[:, :-pad] if pad != 0 else target for pad, target in
                       zip(pads, np.split(y, num_steps))]
        else:
            targets = []
            for i, (pad, target) in enumerate(zip(pads, np.split(y, num_steps))):
                if i >= extra_steps:
                    if pad != 0:
                        targets.append(target[:, :-pad])
                    else:
                        targets.append(target)

        targets = [pd.Series(target[0]) for target in targets]

        # For final round of iteration -- makes sure not to include predictions for padded values
        for df in targets:
            predictions = predictions.append(df)

    before_transform = pd.Series(train_data.Y)
    after_transform = predictions

    # If 0 then transformation was executed correctly
    print (before_transform != after_transform).sum()