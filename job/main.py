from load_data import load_data
from run import run_model, load_model
from config import loadConfig

import pandas as pd
import os
import sys


def main():

    unbuffered = os.fdopen(sys.stdout.fileno(), 'w', 0)
    sys.stdout = unbuffered

    # Ignore SettingWithCopy warnings
    pd.options.mode.chained_assignment = None

    try:
        configFile = sys.argv[1]
    except IndexError:
        configFile = 'ToyNewConfig.txt'

    dataConfig, modelConfig = loadConfig(configFile)

    # Check if a model is being read from file
    if modelConfig['InputDirectory'] == "":

        (train_data, test_data), dataConfig = load_data(dataConfig)
        modelConfig['NumFeatures'] = train_data.X.shape[1]

        ytrain_pred, ytrain_true, ytest_pred, ytest_true = run_model(modelConfig, dataConfig, train_data, test_data)

        # Write predictions to csv
        pd.DataFrame({'{}_prediction'.format(train_data.config['Target']): ytrain_pred.values},
                     index=train_data.X.index).sort_index(level=0).swaplevel(-2, -1).to_csv(train_data.config['OutputInSample'])

        pd.DataFrame({'{}_prediction'.format(train_data.config['Target']): ytest_pred.values},
                 index=test_data.X.index).sort_index(level=0).swaplevel(-2, -1).to_csv(test_data.config['OutputOutSample'])

    else:

        (data,), dataConfig = load_data(dataConfig, new_model=False)
        modelConfig['NumFeatures'] = data.X.shape[1]
        ypred = load_model(modelConfig, dataConfig, data)

        pd.DataFrame({'alpha': ypred.values},
                     index=data.X.index).sort_index(level=0).swaplevel(-2, -1).to_csv(data.config['AlphaDirectory'])

    print 'Wrote predictions to csv'