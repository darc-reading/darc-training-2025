"""Change of observation network
"""
import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr


def plot_timeseries(DATA_DIR):
    # Load ERA5 data and compute spatial average
    with xr.open_dataset(os.path.join(DATA_DIR, 'era5_data', 'regional_tcwv_era5_monthly_1995-2005.nc')) as ds:
        ds_t4 = xr.Dataset(
            data_vars={"tcwv": ("time", ds.tcwv.mean(dim=['latitude', 'longitude'], skipna=True).values)},
            coords={"time": ds.tcwv.valid_time.values}
        )

    # Load NCEP data and compute spatial average
    ncep_data_raw = xr.open_dataset(os.path.join(DATA_DIR, 'ncep_data', 'pr_wtr.mon.mean.nc'))
    ncep_data = ncep_data_raw.sel(time=slice('1995-01-01', '2005-12-31'), lat=slice(-10, -50), lon=slice(-180, 180))
    ncep_spatial_mean = ncep_data.mean(dim=['lat', 'lon'], skipna=True)
    ncep_time = ncep_spatial_mean.time.values
    ncep_data_raw.close()


    # Create a 2x1 subplot layout
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # --- Top subplot: ERA5 and NCEP TCWV time series ---
    # Plot both datasets on the same subplot
    axs[0].plot(ds_t4.time, ds_t4.tcwv, 'b-', linewidth=1.5, label='ERA5 TCWV')
    axs[0].plot(ncep_spatial_mean.time, ncep_spatial_mean.pr_wtr, 'g-', linewidth=1.5, label='NCEP TCWV')
    axs[0].axvline(x=np.datetime64('1998-07-01'), color='r', linestyle='--', label='ATOVS Introduction')

    # Calculate means before and after for both datasets
    mean_before_era5 = ds_t4.tcwv.sel(time=slice('1995-01-01', '1998-06-30')).mean().values
    mean_after_era5 = ds_t4.tcwv.sel(time=slice('1998-07-01', '2005-12-31')).mean().values
    mean_before_ncep = ncep_spatial_mean.pr_wtr.sel(time=slice('1995-01-01', '1998-06-30')).mean().values
    mean_after_ncep = ncep_spatial_mean.pr_wtr.sel(time=slice('1998-07-01', '2005-12-31')).mean().values

    # Add title and labels
    axs[0].set_title('Total Column Water Vapor Time Series', fontsize=12)
    axs[0].set_ylabel('TCWV (kg/m²)', fontsize=10)
    axs[0].grid(True, alpha=0.3)

    # Add text annotations showing mean values
    axs[0].text(0.02, 0.98, f'ERA5 mean: Before={mean_before_era5:.2f}, After={mean_after_era5:.2f} kg/m²',
             transform=axs[0].transAxes, fontsize=9, verticalalignment='top', color='blue')
    axs[0].text(0.02, 0.92, f'NCEP mean: Before={mean_before_ncep:.2f}, After={mean_after_ncep:.2f} kg/m²',
             transform=axs[0].transAxes, fontsize=9, verticalalignment='top', color='green')
    axs[0].legend(loc='upper right', frameon=True, fontsize=9)

    # --- Bottom subplot: ERA5 and NCEP variability ---
    # Calculate rolling standard deviation
    rolling_std_era5 = ds_t4.tcwv.rolling(time=12, center=True).std()
    rolling_std_ncep = ncep_spatial_mean.pr_wtr.rolling(time=12, center=True).std()

    # Plot both datasets' variability
    axs[1].plot(ds_t4.time, rolling_std_era5, 'b-', linewidth=1.5, label='ERA5 12-month Rolling Std. Dev.')
    axs[1].plot(ncep_spatial_mean.time, rolling_std_ncep, 'g-', linewidth=1.5, label='NCEP 12-month Rolling Std. Dev.')
    axs[1].axvline(x=np.datetime64('1998-07-01'), color='r', linestyle='--', label='ATOVS Introduction')

    # Calculate standard deviations before and after
    std_before_era5 = ds_t4.tcwv.sel(time=slice('1995-01-01', '1998-06-30')).std().values
    std_after_era5 = ds_t4.tcwv.sel(time=slice('1998-07-01', '2005-12-31')).std().values
    std_before_ncep = ncep_spatial_mean.pr_wtr.sel(time=slice('1995-01-01', '1998-06-30')).std().values
    std_after_ncep = ncep_spatial_mean.pr_wtr.sel(time=slice('1998-07-01', '2005-12-31')).std().values

    # Add title and labels
    axs[1].set_title('TCWV Variability (12-month Rolling Standard Deviation)', fontsize=12)
    axs[1].set_xlabel('Year', fontsize=10)
    axs[1].set_ylabel('Standard Deviation (kg/m²)', fontsize=10)
    axs[1].grid(True, alpha=0.3)

    # Add text annotations showing standard deviation values
    axs[1].text(0.02, 0.98, f'ERA5 std: Before={std_before_era5:.2f}, After={std_after_era5:.2f} kg/m²',
             transform=axs[1].transAxes, fontsize=9, verticalalignment='top', color='blue')
    axs[1].text(0.02, 0.92, f'NCEP std: Before={std_before_ncep:.2f}, After={std_after_ncep:.2f} kg/m²',
             transform=axs[1].transAxes, fontsize=9, verticalalignment='top', color='green')
    axs[1].legend(loc='upper right', frameon=True, fontsize=9)

    # Add a main title for the entire figure
    fig.suptitle('Comparison of ERA5 and NCEP Total Column Water Vapor over Southern Ocean (10°S-50°S)',
                 fontsize=16, y=0.98)

    # Adjust spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3)

    # Display the figure
    plt.show()
