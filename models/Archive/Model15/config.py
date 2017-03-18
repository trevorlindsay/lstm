import sys


def getParams(debug=False):

    if debug:
        configFile = 'ExampleDataConfig.txt'
    else:
        configFile = sys.argv[1]

    config = {}

    with open(configFile, 'rb') as f:
        for line in f:
            if line[0] == '#' or line.isspace():
                continue

            paramName = line.split('::')[0].strip()
            paramValue = line.split('::')[1].strip().split('#')[0].strip()
            config[paramName] = paramValue

    msg = '\n'.join('{0} = {1}'.format(param, value) for param, value in config.iteritems())
    print msg

    return config


class TestConfig(object):

    def __init__(self, num_features, num_epochs):

        self.num_features = num_features
        self.hidden_size = num_features // 2
        self.dense_units = 1
        self.init_scale = 0.1
        self.learning_rate = 0.01
        self.max_grad_norm = 5
        self.num_layers = 2
        self.num_steps = 1
        self.keep_prob = 0.70
        self.decay_epoch = 100
        self.lr_decay = 1 / 1.05
        self.max_group_size = self.batch_size = None

    def params(self):
        return {'num_features': self.num_features,
                'num_layers': self.num_layers,
                'hidden_size (# of units)': self.hidden_size,
                'num_steps (depth of tensor)': self.num_steps,
                'units in dense layer': self.dense_units,
                'initializer_range': self.init_scale,
                'initial learning rate': self.learning_rate,
                'max_grad_norm': self.max_grad_norm,
                'dropout probability': 1 - self.keep_prob,
                'learning rate decay': self.lr_decay}


class ProductionConfig(object):

    def __init__(self, num_features, num_epochs):

        self.num_features = num_features
        self.hidden_size = num_features // 2
        self.dense_units = 1
        self.init_scale = 0.04
        self.learning_rate = 0.01
        self.max_grad_norm = 5
        self.num_layers = 2
        self.num_steps = 200
        self.keep_prob = 0.70
        self.decay_epoch = 100
        self.lr_decay = 1 / 1.05
        self.max_group_size = self.batch_size = None

    def params(self):
        return {'num_features': self.num_features,
                'num_layers': self.num_layers,
                'hidden_size (# of units)': self.hidden_size,
                'num_steps (depth of tensor)': self.num_steps,
                'units in dense layer': self.dense_units,
                'initializer_range': self.init_scale,
                'initial learning rate': self.learning_rate,
                'max_grad_norm': self.max_grad_norm,
                'dropout probability': 1 - self.keep_prob,
                'learning rate decay': self.lr_decay}
