import numpy as np
import pandas as pd

from load_data import load_data
from run import run_model

from datetime import datetime
import os
import sys
import warnings


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

    print 'Initiating Model17 at {}'.format(datetime.now())

    DEBUG = debug
    train_data, test_data, id1_train, id1_test = load_data(debug=DEBUG)
    num_features = train_data.num_features

    xbest = [2, 45, 0.006, 0.6, 1, 0.03, 15, 24, 0.6]
    run_best_model(xbest, DEBUG)


def run_best_model(x, DEBUG):

    train_data, test_data, id1_train, id1_test = load_data(debug=DEBUG)
    NUM_CORES = 48

    args = {'num_layers': int(x[0]),
            'hidden_size': int(x[1]),
            'learning_rate': x[2],
            'keep_prob': x[3],
            'num_steps': int(x[4]),
            'init_scale': x[5],
            'max_grad_norm': int(x[6]),
            'decay_epoch': int(x[7]),
            'lr_decay': x[8],
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


main(True)
