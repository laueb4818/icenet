# system
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "polar-modelling-icenet", "icenet"))

# data
import json
import pandas as pd
import numpy as np
import xarray as xr

# custom functions from the icenet repo
from utils import IceNetDataLoader, create_results_dataset_index, arr_to_ice_edge_arr

# modelling
from tensorflow.keras.models import load_model

# plotting
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.offsetbox import AnchoredText


# utils
import urllib.request
import re
from tqdm import tqdm
import calendar
from pprint import pprint

pd.options.display.max_columns = 10


# data folder
data_folder = "./data"

config = {
    "obs_data_folder": os.path.join(data_folder, "obs"),
    "mask_data_folder": os.path.join(data_folder, "masks"),
    "forecast_data_folder": os.path.join(data_folder, "forecasts"),
    "network_dataset_folder": os.path.join(data_folder, "network_datasets"),
    "dataloader_config_folder": "./polar-modelling-icenet/dataloader_configs",
    "network_h5_files_folder": "./polar-modelling-icenet/networks",
    "forecast_results_folder": "./polar-modelling-icenet/results",
}

# Generate the folder structure through a list of comprehension
[os.makedirs(val) for key, val in config.items() if not os.path.exists(val)]

url = "https://ramadda.data.bas.ac.uk/repository/entry/get/"

target_networks = list(range(36, 61))

# for network in target_networks:
#     urllib.request.urlretrieve(
#         url
#         + f"network_tempscaled_{network}.h5?entryid=synth%3A71820e7d-c628-4e32-969f-464b7efb187c%3AL25ldXJhbF9uZXR3b3JrX21vZGVsL25ldHdvcmtfdGVtcHNjYWxlZF8zNi5oNQ%3D%3D",
#         os.path.join(
#             config["network_h5_files_folder"], f"network_tempscaled_{network}.h5"
#         ),
#     )

network_regex = re.compile("^network_tempscaled_([0-9]*).h5$")

network_fpaths = [
    os.path.join(config["network_h5_files_folder"], f)
    for f in sorted(os.listdir(config["network_h5_files_folder"]))
    if network_regex.match(f)
]

ensemble_seeds = [
    network_regex.match(f)[1]
    for f in sorted(os.listdir(config["network_h5_files_folder"]))
    if network_regex.match(f)
]

networks = []
for network_fpath in network_fpaths:
    print("Loading model from {}... ".format(network_fpath), end="", flush=True)
    networks.append(load_model(network_fpath, compile=False))
    print("Done.")
model = "IceNet"

forecast_start = pd.Timestamp("2020-01-01")
forecast_end = pd.Timestamp("2020-12-01")

n_forecast_months = dataloader.config["n_forecast_months"]
print("\n# of forecast months: {}\n".format(n_forecast_months))

forecast_folder = os.path.join(
    config["forecast_data_folder"], "icenet", dataloader_ID, model
)

if not os.path.exists(forecast_folder):
    os.makedirs(forecast_folder)

print("Loading ground truth SIC... ", end="", flush=True)
true_sic_fpath = os.path.join(config["obs_data_folder"], "siconca_EASE.nc")
true_sic_da = xr.open_dataarray(true_sic_fpath)
print("Done.")

# define list of lead times
leadtimes = np.arange(1, n_forecast_months + 1)

# add ensemble to the list of models
ensemble_seeds_and_mean = ensemble_seeds.copy()
ensemble_seeds_and_mean.append("ensemble")

all_target_dates = pd.date_range(start=forecast_start, end=forecast_end, freq="MS")

all_start_dates = pd.date_range(
    start=forecast_start - pd.DateOffset(months=n_forecast_months - 1),
    end=forecast_end,
    freq="MS",
)

shape = (len(all_target_dates), *dataloader.config["raw_data_shape"], n_forecast_months)

coords = {
    "time": all_target_dates,  # To be sliced to target dates
    "yc": true_sic_da.coords["yc"],
    "xc": true_sic_da.coords["xc"],
    "lon": true_sic_da.isel(time=0).coords["lon"],
    "lat": true_sic_da.isel(time=0).coords["lat"],
    "leadtime": leadtimes,
    "seed": ensemble_seeds_and_mean,
    "ice_class": ["no_ice", "marginal_ice", "full_ice"],
}

# Probabilistic SIC class forecasts
dims = ("seed", "time", "yc", "xc", "leadtime", "ice_class")
shape = (len(ensemble_seeds_and_mean), *shape, 3)

model_forecast = xr.DataArray(
    data=np.zeros(shape, dtype=np.float32), coords=coords, dims=dims
)

for start_date in tqdm(all_start_dates):
    # Target forecast dates for the forecast beginning at this `start_date`
    target_dates = pd.date_range(
        start=start_date,
        end=start_date + pd.DateOffset(months=n_forecast_months - 1),
        freq="MS",
    )

    X, y, sample_weights = dataloader.data_generation([start_date])
    mask = sample_weights > 0
    pred = np.array([network.predict(X)[0] for network in networks])
    pred *= mask  # mask outside active grid cell region to zero
    # concat ensemble mean to the set of network predictions
    ensemble_mean_pred = pred.mean(axis=0, keepdims=True)
    pred = np.concatenate([pred, ensemble_mean_pred], axis=0)

    for i, (target_date, leadtime) in enumerate(zip(target_dates, leadtimes)):
        if target_date in all_target_dates:
            model_forecast.loc[:, target_date, :, :, leadtime] = pred[..., i]

print("Saving forecast NetCDF for {}... ".format(model), end="", flush=True)

forecast_fpath = os.path.join(
    forecast_folder, f"{model.lower()}_forecasts.nc".format(model.lower())
)
model_forecast.to_netcdf(forecast_fpath)  # export file as Net

print("Done.")

metric_compute_list = ["Binary accuracy", "SIE error"]

forecast_fpath = os.path.join(
    forecast_folder, f"{model.lower()}_forecasts.nc".format(model.lower())
)

chunks = {"seed": 1}
icenet_forecast_da = xr.open_dataarray(forecast_fpath, chunks=chunks)
icenet_seeds = icenet_forecast_da.seed.values

mask_fpath_format = os.path.join(
    config["mask_data_folder"], "active_grid_cell_mask_{}.npy"
)

month_mask_da = xr.DataArray(
    np.array(
        [
            np.load(mask_fpath_format.format("{:02d}".format(month)))
            for month in np.arange(1, 12 + 1)
        ],
    )
)

now = pd.Timestamp.now()
new_results_df_fname = now.strftime("%Y_%m_%d_%H%M%S_forecast_results.csv")
new_results_df_fpath = os.path.join(
    config["forecast_results_folder"], new_results_df_fname
)

print("New results will be saved to {}\n\n".format(new_results_df_fpath))

results_df_fnames = sorted(
    [
        f
        for f in os.listdir(config["forecast_results_folder"])
        if re.compile(".*.csv").match(f)
    ]
)
if len(results_df_fnames) >= 1:
    old_results_df_fname = results_df_fnames[-1]
    old_results_df_fpath = os.path.join(
        config["forecast_results_folder"], old_results_df_fname
    )
    print("\n\nLoading previous results dataset from {}".format(old_results_df_fpath))

# Load previous results, do not interpret 'NA' as NaN
results_df = pd.read_csv(old_results_df_fpath, keep_default_na=False, comment="#")

# Remove existing IceNet results
results_df = results_df[~results_df["Model"].str.startswith("IceNet")]

# Drop spurious index column if present
results_df = results_df.drop("Unnamed: 0", axis=1, errors="ignore")
results_df["Forecast date"] = [
    pd.Timestamp(date) for date in results_df["Forecast date"]
]

results_df = results_df.set_index(
    ["Model", "Ensemble member", "Leadtime", "Forecast date"]
)

# Add new models to the dataframe
multi_index = create_results_dataset_index(
    [model], leadtimes, all_target_dates, model, icenet_seeds
)
results_df = results_df.append(pd.DataFrame(index=multi_index)).sort_index()

icenet_sip_da = icenet_forecast_da.sel(ice_class=["marginal_ice", "full_ice"]).sum(
    "ice_class"
)

true_sic_fpath = os.path.join(config["obs_data_folder"], "siconca_EASE.nc")
true_sic_da = xr.open_dataarray(true_sic_fpath, chunks={})
true_sic_da = true_sic_da.load()
true_sic_da = true_sic_da.sel(time=all_target_dates)

if "Binary accuracy" in metric_compute_list:
    binary_true_da = true_sic_da > 0.15

months = [pd.Timestamp(date).month - 1 for date in all_target_dates]
mask_da = xr.DataArray(
    [month_mask_da[month] for month in months],
    dims=("time", "yc", "xc"),
    coords={
        "time": true_sic_da.time.values,
        "yc": true_sic_da.yc.values,
        "xc": true_sic_da.xc.values,
    },
)

print("Analysing forecasts: \n\n")

print("Computing metrics:")
print(metric_compute_list)

binary_forecast_da = icenet_sip_da > 0.5

compute_ds = xr.Dataset()
for metric in metric_compute_list:
    if metric == "Binary accuracy":
        binary_correct_da = (binary_forecast_da == binary_true_da).astype(np.float32)
        binary_correct_weighted_da = binary_correct_da.weighted(mask_da)

        # Mean percentage of correct classifications over the active
        #   grid cell area
        ds_binacc = binary_correct_weighted_da.mean(dim=["yc", "xc"]) * 100
        compute_ds[metric] = ds_binacc

    elif metric == "SIE error":
        binary_forecast_weighted_da = binary_forecast_da.astype(int).weighted(mask_da)
        binary_true_weighted_da = binary_true_da.astype(int).weighted(mask_da)

        ds_sie_error = (
            binary_forecast_weighted_da.sum(["xc", "yc"])
            - binary_true_weighted_da.sum(["xc", "yc"])
        ) * 25**2

        compute_ds[metric] = ds_sie_error

print("Writing to results dataset...")
for compute_da in iter(compute_ds.data_vars.values()):
    metric = compute_da.name

    compute_df_index = (
        results_df.loc[pd.IndexSlice[model, :, leadtimes, all_target_dates], metric]
        .droplevel(0)
        .index
    )

    # Ensure indexes are aligned for assigning to results_df
    compute_df = (
        compute_da.to_dataframe()
        .reset_index()
        .set_index(["seed", "leadtime", "time"])
        .reindex(index=compute_df_index)
    )

    results_df.loc[
        pd.IndexSlice[model, :, leadtimes, all_target_dates], metric
    ] = compute_df.values

print("\nCheckpointing results dataset... ", end="", flush=True)
results_df.to_csv(new_results_df_fpath)
print("Done.")

settings_lineplots = dict(
    padding=0.1,
    height=400,
    width=700,
    fontsize={"title": "120%", "labels": "120%", "ticks": "100%"},
)

# Reset index to preprocess results dataset
results_df = results_df.reset_index()

results_df["Forecast date"] = pd.to_datetime(results_df["Forecast date"])

month_names = np.array(
    [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sept",
        "Oct",
        "Nov",
        "Dec",
    ]
)
forecast_month_names = month_names[results_df["Forecast date"].dt.month.values - 1]
results_df["Calendar month"] = forecast_month_names

results_df = results_df.set_index(
    ["Model", "Ensemble member", "Leadtime", "Forecast date"]
)

# subset target period
results_df = results_df.loc(axis=0)[
    pd.IndexSlice[:, :, :, slice(forecast_start, forecast_end)]
]

results_df = results_df.sort_index()

results_df.head()

new_results_df_fname = now.strftime("%Y_%m_%d_%H%M%S_forecast_results_modified.csv")
new_results_df_fpath = os.path.join(
    config["forecast_results_folder"], new_results_df_fname
)

results_df.to_csv(new_results_df_fpath)
