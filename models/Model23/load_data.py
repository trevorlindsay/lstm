import pandas as pd
import numpy as np
from collections import Counter


class DataSet(object):

    def __init__(self, X, Y, config):

        self._X = X.sort_index(level=('id1', 'timeindex')) # Sort in this order -- not order of index
        self._Y = Y.sort_index(level=('id1', 'timeindex')) # Sort in this order -- not order of index
        self._timesteps = timesteps = tuple(X.index.unique())
        self._config = config
        self.max_group_size = config['MaxGroupSize']
        self.num_features = X.shape[1]
        self._features = None
        self._targets = None
        self.num_steps = None
        self.batch_size = None
        self.step_size = None
        self.pads = None
        self.weights = self._X[self.config['Weight'].strip()].map((lambda x: 1 / ((x + 1e-10) ** 2)))

    def reset(self):

        self._features = None
        self._targets = None

    def prepBatches(self, batch_size, step_size):

        self.batch_size = batch_size
        self.step_size = step_size

        features = self.X
        targets = self.Y
        num_features = self.num_features
        max_group_size = self.max_group_size

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

        # Create array to hold weights
        # Same shape as targets array below
        weight_col = self.config['Weight'].strip()
        weights = np.asarray([df[weight_col].map((lambda x: 1 / ((x + 1e-10) ** 2))).values for df in self.features])

        # Converts list of dataframes to 3D numpy array
        # features.shape = (num_instruments, max_group_size, num_features)
        # targets.shape = (num_instruments, max_group_size)
        features = np.asarray([df.values for df in self.features])
        targets = np.asarray([df.values[:,0] for df in self.targets])

        num_instruments = features.shape[0]
        batch_size = self.batch_size
        step_size = self.step_size
        batch_epoch_size = num_instruments // batch_size
        step_epoch_size = self.max_group_size // step_size

        for i in range(batch_epoch_size):

            for j in range(step_epoch_size):

                x = features[i * batch_size : (i + 1) * batch_size, j * step_size : (j + 1) * step_size]
                y = targets[i * batch_size : (i + 1) * batch_size, j * step_size : (j + 1) * step_size]
                w = weights[i * batch_size: (i + 1) * batch_size, j * step_size: (j + 1) * step_size]
                yield (x, y, w)

            # For when max_group_size % step_size != 0, pad the matrices to the left with old values
            if step_epoch_size * step_size < self.max_group_size:

                inser_x = features[i * batch_size : (i + 1) * batch_size, step_epoch_size * step_size :]
                inser_y = targets[i * batch_size : (i + 1) * batch_size, step_epoch_size * step_size :]
                inser_w = weights[i * batch_size: (i + 1) * batch_size, step_epoch_size * step_size:]
                x[-inser_x.shape[0]:, -inser_x.shape[1]:] = inser_x
                y[-inser_y.shape[0]:, -inser_y.shape[1]:] = inser_y
                w[-inser_w.shape[0]:, -inser_w.shape[1]:] = inser_w
                yield(x, y, w)

        # For when num_instruments % batch_size != 0, pad the matrices to the left with old values
        if batch_epoch_size * batch_size < num_instruments:

            for j in range(step_epoch_size):

                inser_x = features[batch_epoch_size * batch_size :, j * step_size : (j + 1) * step_size]
                inser_y = targets[batch_epoch_size * batch_size :, j * step_size : (j + 1) * step_size]
                inser_w = weights[batch_epoch_size * batch_size:, j * step_size: (j + 1) * step_size]
                x[-inser_x.shape[0]:, -inser_x.shape[1]:] = inser_x
                y[-inser_y.shape[0]:, -inser_y.shape[1]:] = inser_y
                w[-inser_w.shape[0]:, -inser_w.shape[1]:] = inser_w
                yield(x, y, w)

            # For when max_group_size % step_size != 0, pad the matrices to the left with old values
            if step_epoch_size * step_size < self.max_group_size:

                inser_x = features[batch_epoch_size * batch_size :, step_epoch_size * step_size :]
                inser_y = targets[batch_epoch_size * batch_size :, step_epoch_size * step_size :]
                inser_w = weights[batch_epoch_size * batch_size:, step_epoch_size * step_size:]
                x[-inser_x.shape[0]:, -inser_x.shape[1]:] = inser_x
                y[-inser_y.shape[0]:, -inser_y.shape[1]:] = inser_y
                w[-inser_w.shape[0]:, -inser_w.shape[1]:] = inser_w
                yield (x, y, w)

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


def load_data(config, new_model=True):

    # For a model read from file (i.e. new_model=False), the target dataframe will just be the 'Weight' column
    (features, target) = createDataFrames(config)

    if new_model:
        xtrain, xtest, ytrain, ytest = train_test_split(features, target, config)
        config['MaxGroupSize'] = getMaxGroupSize(frames=(xtrain, xtest))
        train_data = DataSet(xtrain, ytrain, config)
        test_data = DataSet(xtest, ytest, config)
        return (train_data, test_data), config

    else:
        features.fillna(features.mean(), inplace=True)
        config['MaxGroupSize'] = getMaxGroupSize(frames=(features,))
        data = DataSet(features, target, config)
        return (data,), config


def createDataFrames(config):

    feature_names = config['NumericalFeatures'].strip().split(',')
    if config['CategoricalFeatures'] != "":
        feature_names += config['CategoricalFeatures'].strip().split(',')

    target_name = [config['Target'].strip()] if 'Target' in config else [config['Weight'].strip()]
    index = [config['Time'].strip(), config['Id'].strip()]

    df = pd.read_csv(config['Input'])
    df.set_index(keys=index, inplace=True)

    # Exclude all samples with missing targets or missing weights
    null_rows = np.asarray(pd.isnull(df[target_name[0]])) | np.asarray([pd.isnull(df[config['Weight'].strip()])])
    df = df[np.negative(null_rows)[0]]
    print 'Removed {} samples with missing targets / weights'.format(np.sum(null_rows[0]))

    features = df[feature_names]
    print 'Created features dataframe with shape: ', features.shape

    target = df[target_name]
    print 'Created target dataframe with shape: ', target.shape

    # Drop original dataframe from memory
    del df

    # Add dummy variables for categorical data
    if config['CategoricalFeatures'] != "":
        to_dummy = config['CategoricalFeatures'].strip().split(',')
        features = pd.get_dummies(features, columns=to_dummy) if len(to_dummy) > 0 else features

    return features, target


def train_test_split(features, target, config):

    start_in_sample = int(config['StartInSample'])
    end_in_sample = int(config['EndInSample'])
    start_out_sample = int(config['StartOutSample'])
    end_out_sample = int(config['EndOutSample'])

    # Impute all missing values with the mean along the column
    xtrain = features.ix[start_in_sample: end_in_sample]
    xtrain.fillna(xtrain.mean(), inplace=True)

    xtest = features.ix[start_out_sample: end_out_sample]
    xtest.fillna(xtest.mean(), inplace=True)

    ytrain = target.ix[start_in_sample: end_in_sample]
    ytest = target.ix[start_out_sample: end_out_sample]

    return xtrain, xtest, ytrain, ytest


def getMaxGroupSize(frames):

    # Determine which instrument has the max number of timesteps -- the number of timesteps will be max_steps
    max_steps = [Counter(zip(*frame.index.values.tolist())[1]).most_common()[0][-1] for frame in frames]
    return np.max(max_steps





