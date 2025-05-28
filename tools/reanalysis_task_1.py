"""plotting 500hPa geopotential height from ERA5 reanalysis data"""
import os

import cartopy.crs as ccrs # type: ignore
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr

# Define our region of interest - North Atlantic/UK region
REGION_EAST = 10
REGION_WEST = -30
REGION_NORTH = 65
REGION_SOUTH = 40

def plot_500_geopotential(data_dir):
    """Plot 500hPa geopotential height from ERA5 reanalysis data.
    Args:
        data_dir (str): Path to the directory containing the ERA5 reanalysis data.
    """
    # read reanalysis data
    ds_t1 = xr.open_dataset(os.path.join(data_dir, 'era5_data', 'reanalysis.nc'))
    # Convert to decameters for plotting (standard in meteorology)
    z500_dam = ds_t1['z'][0,0,:,:] / 9.80665
    # Extract lon/lat values
    lon = ds_t1.longitude.values
    lat = ds_t1.latitude.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    ds_t1.close()

    # Create figure and axes
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    # Plot the geopotential height contours
    contourf = ax.contourf(lon2d, lat2d, z500_dam/10., levels=15, cmap='viridis',
                          transform=ccrs.PlateCarree())
    contour = ax.contour(lon2d, lat2d, z500_dam/10., levels=15, colors='black',
                          linewidths=0.5, linestyles='solid',
                          transform=ccrs.PlateCarree())
    ax.clabel(contour, fontsize=10)
    plt.colorbar(contourf, label='Geopotential Height (dam)', shrink=0.8)
    # Plot coastlines
    ax.coastlines(color='black', linewidth=0.5)

    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(np.arange(min(lon), max(lon)+1, 20))
    ax.set_yticks(np.arange(min(lat), max(lat)+1, 10))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
    ax.set_xlim(REGION_WEST, REGION_EAST)  # Approximate longitudes for North Atlantic/UK
    ax.set_ylim(REGION_SOUTH,REGION_NORTH)   # Approximate latitudes for North Atlantic/UK
    ax.set_title('ERA5 500hPa Geopotential Height - North Atlantic/UK Region, 1-3-2025',
                 fontsize=14)
    plt.tight_layout()
    plt.show()  # Show the plot

if __name__ == "__main__":
    # Define the directory containing the ERA5 reanalysis data
    DATA_DIR = 'test_sukun_practical/working/da_practical_data'
    plot_500_geopotential(DATA_DIR)
