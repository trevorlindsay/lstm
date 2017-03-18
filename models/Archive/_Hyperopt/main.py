from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
import tensorflow as tf
import pandas as pd

from load_data import load_data
from run import run_model

from datetime import datetime
import os
import sys
import warnings
import cPickle as pickle


def initiate(args):

    global train_data, test_data, id1_train, id1_test, DEBUG, NUM_CORES, NUM_EPOCHS

    args['num_features'] = train_data.num_features
    _, _, ypred, ytrue = run_model(args, train_data, test_data, id1_train, id1_test, NUM_EPOCHS, NUM_CORES, DEBUG)

    return {'loss': -compute_r2(ytrue, ypred),
            'mse': compute_mse(ytrue, ypred),
            'status': STATUS_OK,
            'eval_time': datetime.now()}


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

    trials = Trials()

    space = hp.choice('model', [
        {'num_layers': hp.choice('num_layers', [1, 2]),
         'num_features': num_features,
         'hidden_size': hp.quniform('hidden_size', 10, num_features, 5),
         'learning_rate': hp.quniform('learning_rate', 0.001, 0.01, 0.001),
         'keep_prob': hp.quniform('keep_prob', 0.5, 1, 0.05),
         'num_steps': hp.quniform('num_steps', 1, 201, 10) if not debug else hp.quniform('num_steps', 1, 2, 1),
         'optimizer': tf.train.AdamOptimizer,
         'lr_decay': hp.quniform('lr_decay', 0.5, 1, 0.05),
         'init_scale': hp.quniform('init_scale', 0, 0.05, 0.01),
         'max_grad_norm': hp.choice('max_grad_norm', [5, 10, 15, 20]),
         'decay_epoch': hp.quniform('decay_epoch', 10, NUM_EPOCHS, 1)}
    ])

    best = fmin(initiate,
                space=space,
                algo=tpe.suggest,
                max_evals=100 if not debug else 50,
                trials=trials)

    best = paramHelper(best, num_features)
    print best

    # Set these params for model
    best['num_features'] = train_data.num_features
    best['optimizer'] = tf.train.AdamOptimizer

    # Re-train best model with 200 epochs
    ytrain_pred, ytrain_true, ytest_pred, ytest_true = run_model(best, train_data, test_data, id1_train, id1_test, 200, NUM_CORES, DEBUG, final_run=True)

    print 'Train r2: {}'.format(compute_r2(ytrain_true, ytrain_pred))
    print 'Test r2: {}'.format(compute_r2(ytest_true, ytest_pred))

    # Write predictions to csv
    pd.DataFrame({'id1': id1_train.id1.values,
                  '{}_prediction'.format(train_data.config['Target']): ytrain_pred},
                 index=train_data.X.index).to_csv(train_data.config['OutputInSample'])

    pd.DataFrame({'id1': id1_test.id1.values,
                  '{}_prediction'.format(train_data.config['Target']): ytest_pred},
                 index=test_data.X.index).to_csv(test_data.config['OutputOutSample'])

    print 'Wrote predictions to csv'

    # Save trials object
    save_path = train_data.config['OutputInSample'][:train_data.config['OutputInSample'].rfind('/')]
    pickle.dump(trials, save_path + '/trials')


def paramHelper(best_args, num_features):

    best_args['num_layers'] = [1, 2][best_args['num_layers']]
    best_args['max_grad_norm'] = [5, 10, 15, 20][best_args['max_grad_norm']]

    return best_args

