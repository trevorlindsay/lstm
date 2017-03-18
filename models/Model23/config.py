from collections import defaultdict

def loadConfig(configFile):

    dataConfig = defaultdict(str)
    modelConfig = defaultdict(str)

    with open(configFile, 'rb') as f:

        for line in f:

            if line[0] == '#' or line.isspace():
                continue

            group, name, value = [x.strip() for x in line.split('::')]

            if group == 'Data':
                dataConfig[name] = value.split('#')[0].strip()
            elif group == 'Model':
                modelConfig[name] = value.split('#')[0].strip()
            else:
                raise UserWarning("The group ({}) specified for {} is not recognized".format(group, name))

    data_msg = "### Data Parameters ###\n" + "\n".join('{0} = {1}'.format(param, value)
                                            for param, value in dataConfig.iteritems())
    model_msg = "### Model Parameters ###\n" + "\n".join('{0} = {1}'.format(param, value)
                                              for param, value in modelConfig.iteritems())

    print data_msg
    print model_msg

    return dataConfig, modelConfig

