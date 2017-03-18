import config as c
import pandas as pd
import numpy as np
from collections import Counter


class DataSet(object):

    def __init__(self, X, Y, config):

        self._X = X
        self._Y = Y
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

    def prepBatches(self, num_steps):

        self.num_steps = num_steps
        max_group_size = self.max_group_size
        num_features = self.num_features

        features = [group for _, group in self.X.groupby(level=0)]
        self.pads = pads = [max_group_size - group.shape[0] for group in features]
        features = np.asarray([np.pad(group, ((0, pads[i]), (0, 0)), 'mean') for i, group in enumerate(features)])

        targets = [group for _, group in self.Y.groupby(level=0)]
        targets = np.asarray([np.pad(group, ((0, pads[i]), (0, 0)), 'mean') for i, group in enumerate(targets)])
        targets = np.reshape(targets, (len(self.timesteps), max_group_size))

        self._features = features
        self._targets = targets

    def __len__(self):
        return self.X.shape[0]

    def __iter__(self):

        features = self.features
        targets = self.targets
        num_steps = self.num_steps
        epoch_size = len(self.timesteps) // num_steps

        if epoch_size == 0:
            raise ValueError('Epoch size equals 0! Decrease the number of steps')

        for i in range(epoch_size):
            x = features[i * num_steps : (i + 1) * num_steps]
            y = targets[i * num_steps : (i + 1) * num_steps]
            yield (x,y)

        while epoch_size * num_steps < len(self.timesteps):
            inser_x = features[(epoch_size + 2) * num_steps:]
            inser_y = targets[(epoch_size + 1) * num_steps:]

            x[:inser_x.shape[0], :inser_x.shape[1]] = inser_x
            y[:inser_y.shape[0]] = inser_y

            epoch_size += 1
            yield(x, y)

        """
        # Arrays to hold feature data and target data
        feature_data = np.zeros(shape=(batch_size, batch_len, num_features), dtype=np.float32)
        target_data = np.zeros(shape=(batch_size, batch_len), dtype=np.float32)

        # Split the data based on the number of batches
        for i in range(batch_size):
            feature_data[i] = features[(batch_len * i): (batch_len * (i + 1))]
            target_data[i] = targets[(batch_len * i): (batch_len * (i + 1))]

        epoch_size = (batch_len - 1) // num_steps

        if epoch_size == 0:
            raise ValueError('Epoch size equals 0! Decrease the batch size or number of steps')

        for i in range(epoch_size):
            x = feature_data[:, (i * num_steps) : ((i + 1) * num_steps)]
            y = target_data[:, (i * num_steps) : ((i + 1) * num_steps)]
            yield(x, y)

        while epoch_size * (num_steps * batch_size) < data_len:
            inser_x = feature_data[:, (epoch_size + 2) * num_steps : ]
            inser_y = target_data[:, (epoch_size + 1) * num_steps: ]

            x[:inser_x.shape[0], :inser_x.shape[1]] = inser_x
            y[:inser_y.shape[0], :inser_y.shape[1]] = inser_y

            epoch_size += 1
            yield (x, y)
        """


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
          'The standard deviation is {:.3f}.'.format(len(features.index.unique()),
                                                     target.shape[0], features.shape[1], target.min()[0],
                                                     target.max()[0], target.mean()[0], target.std()[0])

    print msg

    xtrain = features.ix[start_in_sample : end_in_sample]
    xtrain.fillna(xtrain.mean(), inplace=True)

    xtest = features.ix[start_out_sample : end_out_sample]
    xtest.fillna(xtest.mean(), inplace=True)

    ytrain = target.ix[start_in_sample : end_in_sample]
    ytrain.fillna(ytrain.mean(), inplace=True)

    ytest = target.ix[start_out_sample : end_out_sample]
    ytest.fillna(ytest.mean(), inplace=True)

    max_timesteps = [Counter(frame.index.values).most_common()[0][-1] for frame in (xtrain, xtest)]
    config['max_group_size'] = np.max(max_timesteps)

    # Create DataSet variables for each set of data
    train_data = DataSet(xtrain, ytrain, config)
    test_data = DataSet(xtest, ytest, config)

    instruments = pd.read_csv(config['Data'], usecols=[ind, id1], index_col=ind)
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

        for i in range(len(header.split(','))):
            col = header.split(',')[i].strip()
            if col == 'timeindex':
                ind = i
            if col == 'id1':
                id1 = i
            if 'feature' in col or col == 'timeindex' or 'risk' in col or 'cat' in col:
                feature_indices.append(i)
            if col in targets or col  == 'timeindex':
                target_indices.append(i)

    return config, feature_indices, target_indices, ind, id1