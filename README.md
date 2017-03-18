# RNN Project

## Model Configuration File

### Overview
The model is run and built based on the information in the configuration file. It is divided into two parts: Data and Model. Below are the meanings of each of the parameters.

### Data
#### Required
1. `Data::Input::` - path to csv input data
2. `Data::Id::` - instrument id column
3. `Data::Time::` - time column
4. `Data::Weight::` - column to use for sample weights
5. `Data::NumericalFeatures::` - comma-separated list of numerical features (including timeindex and instrument id)
6. `Data::CategoricalFeatures::` - comma-separated list of categorical features

#### Required Only for Training
7. `Data::OutputInSample::` - path to csv output predictions
8. `Data::OutputOutSample::` - path to csv output predictions
9. `Data::Target::` - target column to predict
10. `Data::StartInSample::` - time index of where to begin in-sample, can be omitted for alpha predictions
11. `Data::EndInSample::` - time index of where to end in-sample
12. `Data::StartOutSample::` - time index of where to begin out-sample
13. `Data::EndOutSample::` - time index of where to end out-sample

#### Required Only for Alpha Predictions (model read from file)
14. `Data::AlphaDirectory::` - path of where to savae output of predictions when model read from file

### Model
#### Required (even if model read from file)
1. `Model::RNN::` - number of neurons in each of the model’s RNN layers (comma-separated list, one number for each layer)
2. `Model::Dense::` - number of neurons in each of the model’s fully-connected layers (comma-separated list, one number for each layer)
3. `Model::Activation::` - activation function to use (MUST BE ONE OF THESE: tanh, relu, sigmoid, softmax)
4. `Model::BatchSize::` - number of instruments to feed through model at once
5. `Model::StepSize::` - number of timesteps to train model on at once (set to -1 to use the max available)

#### Required Only for Training
6. `Model::NumEpochs::` - number of epochs for training
7. `Model::LearningRate::` - learning rate for Adam optimizer
8. `Model::KeepProb::` - dropout probability between layers
9. `Model::InitScale::` - max of range for initializing weights
10. `Model::MaxGradNorm::` - max allowed gradient, anything above clipped to this value
11. `Model::OutputDirectory::` - directory to save model

#### Required Only When Model Read from File (must omit otherwise)
12. `Model::InputDirectory::` - path to saved model

#### Optional
13. `Model::NumCores::` - number of CPU cores to use when running model (if not specified, will be set to max available)


## Using Compute Framework with EC2 Instance

### Submitting Jobs
1. Ensure version of model you wish to run is in the <code>job</code> folder
2. Run <code>bash scripts/submit</code> (Note: the script won't work if you run it from within the <code>scripts</code> folder!)

### Checking Log of Active Job
1. Run <code>bash scripts/log</code> (Note: the script won't work if you run it from within the <code>scripts</code> folder!)
2. When prompted, enter the calc id for the job then enter the number of lines you'd like to see

### Retrieving the Results of Finished Job
1. Run <code>bash scripts/result</code> (Note: the script won't work if you run it from within the <code>scripts</code> folder!)
2. When prompted, enter the calc id for the job
3. New folder will be created in <code>~/compute_framework_results/</code> with the name <code>calcxxxx</code> (where xxxx will be the calc id)
4. The folder will also be copied to the <code>results</code> folder in the repo

## Using Tensorboard Summaries
Whenever a model is run, it will produce summaries that can be viewed with Tensorboard. The summaries track two scalar values (mean squared error and r-squared) while the model trains.

To view the summaries with Tensorboard, run the following command:
<code>tensorboard --logdir=$PATH/TO/SUMMARIES</code>

Then navigate to the path given (usually http://0.0.0.0:6006) in a web browser to view the graphs (note: there are known issues with using Tensorboard with Safari).

### Example of Tensorboard output:
![alt text](https://github.com/startupml/rnn/raw/master/screenshots/tensorboard1.png "View of Tensorboard")
