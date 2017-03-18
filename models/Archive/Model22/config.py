import sys


def getParams(debug=False):

    if debug:
        configFile = 'ToyExample.txt'
        # configFile = 'ExampleDataConfig.txt'
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

