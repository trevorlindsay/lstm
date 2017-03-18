from load_data import load_data
from run import run_model

import pandas as pd
from datetime import datetime
import os
import sys
import warnings


def main(debug=False):

    warnings.filterwarnings('ignore')
    unbuffered = os.fdopen(sys.stdout.fileno(), 'w', 0)
    sys.stdout = unbuffered

    print 'Initiating Model22 at {}'.format(datetime.now())

    NUM_CORES = 48
    NUM_EPOCHS = 50

    xbest = [1, 150, 0.001, 0.75, 50, 0.65, 0.04, 10, 35]
    run_best_model(xbest, debug)


def run_best_model(x, DEBUG=False):

    NUM_CORES = 48
    NUM_EPOCHS = 50

    train_data, test_data, id1_train, id1_test = load_data(debug=DEBUG)

    args = {'num_layers': int(x[0]),
            'hidden_size': int(x[1]),
            'learning_rate': x[2],
            'keep_prob': x[3],
            'batch_size': int(x[4]) if not DEBUG else 1,
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
                                                                 NUM_EPOCHS,
                                                                 NUM_CORES,
                                                                 DEBUG,
                                                                 final_run=True)

    # Write predictions to csv
    pd.DataFrame({'{}_prediction'.format(train_data.config['Target']): ytrain_pred.values},
                 index=train_data.X.index).sort_index(level=0).swaplevel(-2, -1).to_csv(train_data.config['OutputInSample'])


    pd.DataFrame({'{}_prediction'.format(train_data.config['Target']): ytest_pred.values},
             index=test_data.X.index).sort_index(level=0).swaplevel(-2, -1).to_csv(test_data.config['OutputOutSample'])

    print 'Wrote predictions to csv'
