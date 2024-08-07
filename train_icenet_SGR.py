import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "icenet"))  # if using jupyter kernel
import numpy as np
import config
import json
import glob
import argparse
import scipy
import tensorflow as tf
import time
import re
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

from utils import IceNetDataLoader, make_exp_decay_lr_schedule
from metrics import ConstructLeadtimeAccuracy
from losses import construct_categorical_focal_loss, weighted_categorical_crossentropy

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import models

import wandb
from wandb.keras import WandbCallback
from callbacks import (
    IceNetPreTrainingEvaluator,
    BatchwiseWandbLogger,
    BatchwiseModelCheckpoint,
)

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

"""
Trains an IceNet network with optional pre-training on climate simulation data,
training (or fine-tuning) on observational data, and optional probability
calibration using temperature scaling.

Logic for which of those three stages to run, and details of the training
procedure, are controlled in the 'User input' section.

Hyperparameters for the learning rate, batch size, and number of filters are
controlled by command line inputs. The default values were determined from a
Bayesian hyperparameter search.

Passing the `--wandb` flag lets you track the `val_acc_mean` (mean validation
set accuracy) and various other metrics on Weights and Biases
(https://wandb.ai/home).

The `--seed` input sets the random seed for initialising the weights of the
network before training. Run this script multiple times with different
integers for `--seed` to build an ensemble.

Trained networks are checkpointed
and saved in `trained_networks/<dataloader_ID>/<architecture_ID>/networks/`.
The filename format used for saving networks is as follows:
- Pre-trained networks: `network_transfer_<seed>.h5`,
- Networks trained on observational data (possibly after pre-training):
`network_<seed>.h5`,
- Final temperature scaled networks: `network_tempscaled_<seed>.h5`
"""

#### Commandline args
####################################################################

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int)
parser.add_argument(
    "--wandb", help="Use Weights and Biases", default=False, action="store_true"
)
parser.add_argument("--learning_rate", default=0.0005, type=float)
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--n_filters_factor", default=1, type=float)

args = parser.parse_args()

seed = args.seed

#### Wandb SLURM config (https://docs.wandb.ai/guides/track/advanced/environment-variables)
####################################################################

# os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
os.environ["WANDB_DIR"] = config.WANDB_DIR
os.environ["WANDB_CONFIG_DIR"] = config.WANDB_CONFIG_DIR
os.environ["WANDB_CACHE_DIR"] = config.WANDB_CACHE_DIR

#### Set up wandb
####################################################################

# These values are used if not being controlled by a Wandb agent in a sweep
defaults = dict(
    learning_rate=args.learning_rate,
    filter_size=3,
    n_filters_factor=args.n_filters_factor,
    batch_size=args.batch_size,
)

if not args.wandb:
    print("\nNot using Weights and Biases.\n")
    wandb_mode = "disabled"
else:
    print("\nUsing Weights and Biases.\n")
    wandb_mode = "online"

wandb.init(
    project="icenet",
    entity="uit-lars-uebbing-org",
    config=defaults,
    allow_val_change=True,
    mode=wandb_mode,
)

print("\n\nUsing a seed of {}\n\n".format(seed))

print("\n\nHyperparams:")
print(wandb.config)
print("\n\n")

#### User input
####################################################################

dataloader_ID = "2023_02_06_1513_icenet_nature_communications"
architecture_ID = "unet_tempscale"

eager_mode = False  # Run TensorFlow in 'graph' or 'eager' mode

# Whether to pre-load and existing saved network file (e.g. for fine-tuning after
#   pre-training or temperature scaling after fine-tuning).
pre_load_network = False
pre_load_network_fname = "network_transfer_{}.h5".format(seed)  # From transfer learning

# Network fnames
transfer_network_fname = "network_transfer_{}.h5".format(seed)  # trained w/ cmip6
network_fname = "network" + "_{}".format(seed) + ".h5"  # trained w/ obs
temp_network_fname = "network_tempscaled_{}.h5".format(seed)  # temperature scaled

icenet_architecture = models.unet_batchnorm

# 2) Fine-tune on obs
train_on_observations = True

# 3) Whether to use temperature scaling
#   To train the T parameter without training network weights, set the following:
#   pre_load_network=True,
# train_on_observations=False
use_temp_scaling = True

# Amount by which to reduce the learning rate when fine-tuning begins
#   Only applies if the network has been pre-trained within this script
fine_tune_learning_rate_reduce_factor = 2

# Number of batches to validate for model checkpointing
#   Options: 'epoch' or int
pretrain_obs_validation_freq = 1000

# 1: Print progress bar. 2: Only print per epoch
verbose = 2

# Whether to load train/val sets in memory (numpy or tfrecords) when training on
#   observational data. If both are False, the data loader is used.
train_on_numpy = False
train_on_tfrecords = False


num_epochs = 50

steps_per_epoch = None

loss = construct_categorical_focal_loss(gamma=2.0)

# Whether to use multiprocessing for generating batches from the data loader
use_multiprocessing = False
workers = 3  # Number of CPUs to use
max_queue_size = 3  # Max number of batches held per CPU

esPatience = 10
esPatienceTransfer = np.inf  # No early stopping for the two transfer learning epochs

# Metric to monitor for early stopping
esMonitor = "val_acc_mean"
esMode = "max"

# Metric to monitor for model checkpointing
mcMonitor = "val_acc_mean"
mcMode = "max"

#### Custom object for loading trained models
####################################################################

custom_objects = {
    "categorical_focal_loss": loss,
    "ConstructLeadtimeAccuracy": ConstructLeadtimeAccuracy,
    "TemperatureScale": models.TemperatureScale,
}

metric = ConstructLeadtimeAccuracy(name="acc_mean", use_all_forecast_months=True)
metrics = [metric]
custom_objects[metric.name] = metric
for i in range(6):
    metric = ConstructLeadtimeAccuracy(
        name="acc_{}month".format(i + 1),
        use_all_forecast_months=False,
        single_forecast_leadtime_idx=i,
    )
    custom_objects[metric.name] = metric
    metrics.append(metric)

#### Data loaders; set up paths
###############################################################################

dataloader_config_fpath = os.path.join(
    config.dataloader_config_folder, dataloader_ID + ".json"
)

icenet_folder = os.path.join(config.networks_folder, dataloader_ID, architecture_ID)
if not os.path.exists(icenet_folder):
    os.makedirs(icenet_folder)

numpy_folder = os.path.join(
    config.networks_folder, dataloader_ID, "obs_train_val_data", "numpy"
)

tfrecords_folder = os.path.join(
    config.networks_folder, dataloader_ID, "obs_train_val_data", "tfrecords"
)

# Network paths
network_h5_files_folder = os.path.join(icenet_folder, "networks")
if not os.path.isdir(network_h5_files_folder):
    os.makedirs(network_h5_files_folder)
network_path = os.path.join(network_h5_files_folder, network_fname)
network_path_preload = os.path.join(network_h5_files_folder, pre_load_network_fname)
network_path_transfer = os.path.join(network_h5_files_folder, transfer_network_fname)
network_path_temp = os.path.join(network_h5_files_folder, temp_network_fname)

# Folder to save training history figure and JSON
training_logs_folder = os.path.join(icenet_folder, "training_logs")
if not os.path.isdir(training_logs_folder):
    os.makedirs(training_logs_folder)

# Load the training and validation data loader objects from the pickle file
print(
    "\nSetting up the training and validation data"
    "loaders with config file: {}\n\n".format(dataloader_ID)
)
dataloader = IceNetDataLoader(dataloader_config_fpath)
val_dataloader = IceNetDataLoader(dataloader_config_fpath)
val_dataloader.convert_to_validation_data_loader()
print("\n\nDone.\n")

input_shape = (*dataloader.config["raw_data_shape"], dataloader.tot_num_channels)

output_shape = (
    *dataloader.config["raw_data_shape"],
    3,
    dataloader.config["n_forecast_months"],
)

sample_weight_shape = (
    *dataloader.config["raw_data_shape"],
    1,
    dataloader.config["n_forecast_months"],
)

#### GPUs
###############################################################################

print("Inputs: {}\n\n".format(dataloader.determine_variable_names()))

dataloader.batch_size = wandb.config.batch_size
val_dataloader.batch_size = wandb.config.batch_size

# Set the seed
np.random.seed(seed)
tf.random.set_seed(seed)
dataloader.set_seed(seed)
dataloader.on_epoch_end()  # Randomly shuffle training samples

if eager_mode is True:
    tf.config.experimental_run_functions_eagerly(True)

#### Define model
###############################################################################

if pre_load_network:
    print("\nLoading network from {}... ".format(network_path_preload))
    network = load_model(network_path_preload, custom_objects=custom_objects)
    print("Done.\n")

else:
    network = icenet_architecture(
        input_shape=input_shape,
        loss=loss,
        weighted_metrics=metrics,
        learning_rate=wandb.config.learning_rate,
        filter_size=wandb.config.filter_size,
        n_filters_factor=wandb.config.n_filters_factor,
        n_forecast_months=dataloader.config["n_forecast_months"],
        use_temp_scaling=use_temp_scaling,
    )

##############################################################################

###############################################################################
#### Training on observational data
###############################################################################


obs_train_data = dataloader
obs_val_data = val_dataloader

print("Training network with dataloaders.\n\n")

obs_callbacks = []

obs_callbacks.append(
    ModelCheckpoint(
        network_path, monitor=mcMonitor, mode=mcMode, verbose=1, save_best_only=True
    )
)

obs_callbacks.append(
    EarlyStopping(monitor=esMonitor, mode=esMode, verbose=1, patience=esPatience)
)

if args.wandb:
    obs_callbacks.append(
        WandbCallback(
            monitor=mcMonitor, mode=mcMode, log_weights=False, log_gradients=False
        )
    )

lr_schedule = LearningRateScheduler(
    make_exp_decay_lr_schedule(
        rate=0.1,
        start_epoch=3,  # Start reducing LR after 3 epochs
        end_epoch=np.inf,
    )
)
obs_callbacks.append(lr_schedule)

print("\n\nTraining network on obervations.\n\n")

history = network.fit(
    obs_train_data,
    epochs=num_epochs,
    verbose=verbose,
    callbacks=obs_callbacks,
    validation_data=obs_val_data,
    max_queue_size=max_queue_size,
    workers=workers,
    use_multiprocessing=use_multiprocessing,
)
print("\n\nTraining on observational data complete.\n\n")

history = history.history

if "lr" in history.keys():
    # convert to string for JSON serialisation
    lrs = history["lr"]
    history["lr"] = [str(lr) for lr in lrs]

fpath = os.path.join(training_logs_folder, "history_{}.json".format(seed))
with open(fpath, "w") as outfile:
    json.dump(history, outfile)

