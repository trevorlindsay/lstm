import config as c
import pandas as pd
import numpy as np
from collections import Counter


class DataSet(object):

    def __init__(self, X, Y, config):

        self._X = X.sort_index(level=('id1', 'timeindex'))
        self._Y = Y.sort_index(level=('id1', 'timeindex'))
        self._features = None
        self._targets = None
        self._timesteps = timesteps = tuple(X.index.unique())
        self._config = config
        self.num_steps = None
        self.num_features = X.shape[1]
        self.batch_size = None
        self.len = len(Y)
        self.max_group_size = config['max_group_size']
        self.pads = None

    def prepBatches(self, batch_size):

        features = self.X
        targets = self.Y
        self.batch_size = batch_size
        max_group_size = self.max_group_size
        num_features = self.num_features

        # Separate observations into input dataframes by instrument
        features = [group for _, group in features.groupby(level=1)]

        # Determine how much padding is required for each dataframe (max_steps will be length of largest group size)
        self.pads = pads = [max_group_size - group.shape[0] for group in features]

        # Pad the bottom of the dataframes with 0
        features = [group.append([pd.DataFrame(np.zeros((1, num_features)), columns=group.columns, index=[(0,0)])] * pads[i])
                    if pads[i] != 0 else group for i, group in enumerate(features)]

        # Separate targets into target series by instrument
        targets = [group for _, group in targets.groupby(level=1)]

        # Pad the series to the bottom with the mean
        targets = [group.append([pd.DataFrame(np.zeros((1, group.shape[1])), columns=group.columns, index=[(0,0)])] * pads[i])
                   if pads[i] != 0 else group for i, group in enumerate(targets)]

        self._features = features
        self._targets = targets


    def __len__(self):
        return self.X.shape[0]

    def __iter__(self):

        # Converts list of dataframes to 3D numpy array
        # features.shape = (num_instruments, max_group_size, num_features)
        # targets.shape = (num_instruments, max_group_size)
        features = np.asarray([df.values for df in self.features])
        targets = np.asarray([df.values[:,0] for df in self.targets])

        num_instruments = features.shape[0]
        batch_size = self.batch_size
        epoch_size = num_instruments // batch_size

        for i in range(epoch_size):
            x = features[i * batch_size : (i + 1) * batch_size]
            y = targets[i * batch_size : (i + 1) * batch_size]
            yield (x, y)

        # For when num_instruments % batch_size != 0, pad the matrices to the left with old values
        while epoch_size * batch_size < num_instruments:
            inser_x = features[epoch_size * batch_size:]
            inser_y = targets[epoch_size * batch_size:]
            self.inser_y = inser_y

            x[-inser_x.shape[0]:, -inser_x.shape[1]:] = inser_x
            y[-inser_y.shape[0]:, -inser_y.shape[1]:] = inser_y

            epoch_size += 1
            yield(x, y)

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def timesteps(self):
        return self._timesteps

    @property
    def features(self):
        return self._features

    @property
    def targets(self):
        return self._targets

    @property
    def config(self):
        return self._config


def load_data(debug=False):

    # Retrieve configuration information
    config, feature_indices, target_indices, ind, id1 = getDataConfig(debug=debug)

    # These values are timeindexes since index of dataframe is timeindex
    start_in_sample = int(config['StartInSample'])
    start_out_sample = int(config['StartOutSample'])
    end_in_sample = int(config['EndInSample'])
    end_out_sample = int(config['EndOutSample'])

    # Read in the csv file, sort the data and get a list of the unique IDs
    features = pd.read_csv(config['Data'], usecols=feature_indices, index_col=ind, verbose=False)
    print 'Created dataframe with columns: ', features.columns.values.tolist()

    target = pd.read_csv(config['Data'], usecols=target_indices, index_col=ind, verbose=False)
    print 'Created dataframe with target: ', target.columns.values

    # Stats about data
    msg = 'The dataset has {} periods.\n' \
          'There are {} targets and {} features.\n' \
          'The targets range from {:.3f} to {:.3f}.\n' \
          'The mean is {:.3f}\n' \
          'The standard deviation is {:.3f}.'.format(features.index.levels[0].shape[0],
                                                     target.shape[0], features.shape[1], target.min()[0],
                                                     target.max()[0], target.mean()[0], target.std()[0])

    print msg

    # Add dummy variables for categorical data
    features = pd.get_dummies(features, columns=['Category0'])

    # Impute all missing values with the mean along the column
    xtrain = features.ix[start_in_sample : end_in_sample]
    xtrain.fillna(xtrain.mean(), inplace=True)

    xtest = features.ix[start_out_sample : end_out_sample]
    xtest.fillna(xtest.mean(), inplace=True)

    ytrain = target.ix[start_in_sample : end_in_sample]
    ytrain.fillna(ytrain.mean(), inplace=True)

    ytest = target.ix[start_out_sample : end_out_sample]
    ytest.fillna(ytest.mean(), inplace=True)

    # Determine which instrument has the max number of timesteps -- the number of timesteps will be max_steps
    max_instruments = [Counter(zip(*frame.index.values.tolist())[1]).most_common()[0][-1] for frame in (xtrain, xtest)]
    config['max_group_size'] = np.max(max_instruments)

    # Create DataSet variables for each set of data
    train_data = DataSet(xtrain, ytrain, config)
    test_data = DataSet(xtest, ytest, config)

    instruments = pd.read_csv(config['Data'], usecols=ind, index_col=ind[0])
    id1_train = instruments.ix[start_in_sample : end_in_sample]
    id1_test = instruments.ix[start_out_sample : end_out_sample]

    return train_data, test_data, id1_train, id1_test


def getDataConfig(debug=False):

    config = c.getParams(debug=debug)

    feature_indices = list()
    target_indices = list()
    ind = -1

    with open(config['Data'], 'rb') as f:

        header = f.readline()
        features = [config['Features'].split(',')[i].strip().lower() for i in range(len(config['Features'].split(',')))]
        targets = [config['Target'].split(',')[i].strip().lower() for i in range(len(config['Target'].split(',')))]

        ind = [-1, -1]

        for i in range(len(header.split(','))):
            col = header.split(',')[i].strip()
            if col == 'timeindex':
                ind[0] = i
            if col == 'id1':
                id1 = i
                ind[1] = i
            if 'feature' in col or col in ('timeindex', 'risk', 'Category0', 'id1'):
                feature_indices.append(i)
            if col in targets or col in ('timeindex', 'id1'):
                target_indices.append(i)

    return config, feature_indices, target_indices, ind, id1