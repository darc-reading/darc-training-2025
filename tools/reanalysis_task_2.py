"""A simple optimal interpolation scheme
"""
import os

import cartopy.crs as ccrs # type: ignore
import matplotlib.colors as mcolors
import matplotlib.gridspec as mgs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import scipy.signal # type: ignore
import xarray as xr

import ipywidgets as widgets # type: ignore
import IPython.display

# Define our region of interest - North Atlantic/UK region
REGION_EAST = 10
REGION_WEST = -30
REGION_NORTH = 65
REGION_SOUTH = 40

def haversine(lat1, lon1, lat2_rad_grid, lon2_rad_grid, lat2_rad_grid_cos):
    """
    lat1, lon1: scalar (reference point in degrees)
    lat2_grid, lon2_grid: 2D arrays (grid of lat/lon in degrees)
    Returns: 2D array of distances in km
    """
    R = 6371.0  # Earth radius in kilometers

    # Convert degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)

    dlat = lat2_rad_grid - lat1_rad
    dlon = lon2_rad_grid - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * lat2_rad_grid_cos * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def plot_correlation(data_dir):
    """
    Plot the spatial correlation function based on a Gaussian function
    with a spatial lengthscale L.
    """
    ds = xr.open_dataset(os.path.join(data_dir, 'era5_data', 'reanalysis.nc'))
    lat = ds.latitude
    lon = ds.longitude
    nlat = ds.sizes['latitude']
    nlon = ds.sizes['longitude']
    ds.close()

    lon, lat = np.meshgrid(lon, lat)
    lat_rad = np.radians(lat)
    lat_cos = np.cos(lat_rad)
    lon_rad = np.radians(lon)

    def do_plot(L:int):
        r_p = haversine(lat[nlat//2, nlon//2], lon[nlat//2, nlon//2],
            lat_rad, lon_rad, lat_cos)
        B = np.exp(-0.5* r_p*r_p/L/L)

        fig = plt.figure('B', figsize=(8, 5))
        fig.clf()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        pc = ax.pcolormesh(
            lon, lat, B, cmap='PuBu', transform=ccrs.PlateCarree(),
            vmin=0, vmax=1)
        ax.coastlines(color='black', linewidth=0.5)
        ax.set_extent([REGION_WEST, REGION_EAST, REGION_SOUTH, REGION_NORTH])
        fig.colorbar(pc, ax=ax, shrink=0.8)
        plt.title('Error covariance correlation function'
                  ' with L = {:.1f} km'.format(L))

        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xticks(np.arange(np.min(lon), np.max(lon)+1, 20))
        ax.set_yticks(np.arange(np.min(lat), np.max(lat)+1, 10))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
        ax.set_xlim(REGION_WEST, REGION_EAST)  # Approximate longitudes for North Atlantic/UK
        ax.set_ylim(REGION_SOUTH,REGION_NORTH)   # Approximate latitudes for North Atlantic/U
        plt.tight_layout()
        plt.show()

    # Create interactive slider for lengthscale L
    float_slider = widgets.FloatSlider(value=1000,
        min=0,
        max=3000,
        step=10.,
        description='Spatial lengthscale (km):',
        layout=widgets.Layout(width='500px'),
        style={'description_width': 'initial'},
    )
    out = widgets.interactive_output(do_plot, {'L': float_slider})
    outbox = widgets.VBox([float_slider, out])
    IPython.display.display(outbox)


def gen_back_noise_mesh(lat, dlat, dlon, nlat, nlon):
    cos_lat = np.cos(np.radians(np.mean(lat)))
    # km/deg for latitude (approx constant)
    dy = 110.574*dlat
    # avg km/deg for longitude
    dx = float(111.320 * np.cos(np.radians(np.mean(lat))))*dlon
    # the domain is extended for better convolution near domain edges
    # Create background field (geopotential height)
    x = np.linspace(-nlon//2, nlon//2+1, nlon)*dx
    y = np.linspace(-nlat//2, nlat//2+1, nlat)*dy
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def gen_background(truth, nlat, nlon, kernel, xx, yy, L):
    slice_y = slice(nlat // 2, nlon + nlat // 2)
    slice_x = slice(nlon // 2, nlat + nlon // 2)
    kernel[slice_y, slice_x] = np.exp(-0.5 * (xx*xx + yy*yy)/L/L)
    kernel[slice_y, slice_x] /= np.sqrt(np.sum(kernel[slice_y, slice_x]**2))
    rng = np.random.default_rng(42)
    nx = nlon//2 + nlon//2 + nlat
    ny = nlon + nlat//2 + nlat//2
    noise = rng.standard_normal(nx*ny).reshape(ny, nx)
    b_noise = scipy.signal.fftconvolve(
        noise, kernel, mode='same')[slice_y, slice_x]
    return truth + b_noise


def gen_obs(truth, lat, lon, omega):
    """
    Generate synthetic observation data for the North Atlantic region
    from ERA5 reanalysis data.
    """
    # Create synthetic observations (scattered points)
    # number of observations
    n_obs = 150
    # randomly choose index of observations locations
    rng = np.random.default_rng(42)
    obs_indices = rng.choice(len(lat) * len(lon), size=n_obs, replace=False)
    obs_y, obs_x = np.unravel_index(obs_indices, (len(lat), len(lon)))

    # Observation values (background + random error + slight bias in some regions)
    obs_values = truth[obs_y, obs_x] + omega * rng.standard_normal(n_obs)

    return obs_y, obs_x, obs_values


def calculate_analysis(background, obs_y, obs_x, obs_values, llat, llon,
                       lat_rad, lon_rad, lat_cos,
                       loc_radius=5.0, obs_error_factor=1.0, bkg_corr_scale=1.0):
    """
    Calculate analysis based on background and observations with adjustable parameters
    """
    n_obs = len(obs_y)
    # Create a grid for the analysis
    # Start with background as initial analysis
    analysis = background.copy()

    # Apply assimilation with tunable parameters
    for i in range(n_obs):
        # Calculate distances from this observation to all grid points
        dist = haversine(llat[obs_y[i], obs_x[i]], llon[obs_y[i], 
            obs_x[i]], lat_rad, lon_rad, lat_cos)
        # Apply localization with adjustable radius
        loc_factor = np.maximum(0, (1.0 - dist/(loc_radius)))

        # Apply background error correlation with adjustable scale
        bkg_corr = np.exp(-0.5 * (dist/(bkg_corr_scale))**2)

        # Calculate innovation
        obs_innovation = obs_values[i] - background[obs_y[i], obs_x[i]]
        # Kalman gain factor (simplified) - depends on observation error
        # Higher obs_error_factor means less trust in observations
        kalman_factor = 1.0 / (1.0 + obs_error_factor)

        # Update analysis
        analysis += kalman_factor * loc_factor * bkg_corr * obs_innovation

    return analysis


# Create interactive visualization
def interactive_da_params(data_dir):
    """Create interactive widget to explore data assimilation parameters"""

    ds = xr.open_dataset(os.path.join(data_dir, 'era5_data', 'reanalysis.nc'))
    lat = ds.latitude
    lon = ds.longitude
    dlon = np.abs(float(lon[1] - lon[0]))
    dlat = np.abs(float(lat[1] - lat[0]))
    nlat = ds.sizes['latitude']
    nlon = ds.sizes['longitude']
    truth = ds['z'][0,0,:,:].values / 9.80665
    ds.close()

    # these variables are used to generate background error noise
    xx, yy = gen_back_noise_mesh(lat, dlat, dlon, nlat, nlon)
    kernel = np.zeros((nlon + nlat//2 + nlat//2, nlon//2 + nlon//2 + nlat))
    # these variables are used to compute distance
    llon, llat = np.meshgrid(lon, lat)
    lat_rad = np.radians(llat)
    lat_cos = np.cos(lat_rad)
    lon_rad = np.radians(llon)

    # Define the interactive function
    def update_analysis(loc_radius, obs_error, bkg_corr_scale):
        background = gen_background(truth, nlon, nlat, kernel, xx, yy, bkg_corr_scale)
        obs_y, obs_x, obs_values = gen_obs(truth, lat, lon, obs_error)
        analysis = calculate_analysis(
            background, obs_y, obs_x, obs_values,
            llat, llon, lat_rad, lon_rad, lat_cos,
            loc_radius, obs_error, bkg_corr_scale
        )

        # Create output figure
        fig = plt.figure('task2', figsize=(16, 10))
        gs = mgs.GridSpec(2, 2, figure=fig)
        # plot background field
        ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
        im = ax.contourf(
                    lon, lat, background/10., levels=15,
                    cmap='viridis', transform=ccrs.PlateCarree()
                )
        contour = ax.contour(lon, lat, background/10.,
                             levels=15, colors='black',
                            linewidths=0.5, linestyles='solid',
                            transform=ccrs.PlateCarree())
        ax.clabel(contour, fontsize=10)
        ax.coastlines(color='black', linewidth=0.5)
        ax.set_extent([REGION_WEST, REGION_EAST, REGION_SOUTH, REGION_NORTH])
        ax.set_title('Background')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xticks(np.arange(np.min(lon), np.max(lon)+1, 20))
        ax.set_yticks(np.arange(np.min(lat), np.max(lat)+1, 10))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
        ax.set_xlim(REGION_WEST, REGION_EAST)  # Approximate longitudes for North Atlantic/UK
        ax.set_ylim(REGION_SOUTH,REGION_NORTH)   # Approximate latitudes for North Atlantic/U
        fig.colorbar(im, ax=ax, shrink=0.8, label='Geopotential Height (dam)')

        # plot observations
        ax = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
        sc = ax.scatter(
                lon[obs_x], lat[obs_y], c=obs_values/10.,
                cmap='viridis', s=10,  transform=ccrs.PlateCarree()
            )
        ax.coastlines(color='black', linewidth=0.5)
        ax.set_extent([REGION_WEST, REGION_EAST, REGION_SOUTH, REGION_NORTH])
        ax.set_title('Observation')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xticks(np.arange(np.min(lon), np.max(lon)+1, 20))
        ax.set_yticks(np.arange(np.min(lat), np.max(lat)+1, 10))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
        ax.set_xlim(REGION_WEST, REGION_EAST)  # Approximate longitudes for North Atlantic/UK
        ax.set_ylim(REGION_SOUTH,REGION_NORTH)   # Approximate latitudes for North Atlantic/U
        fig.colorbar(sc, ax=ax, shrink=0.8, label='Geopotential Height (dam)')

        # Calculate new analysis
        ax = fig.add_subplot(gs[2], projection=ccrs.PlateCarree())
        im = ax.contourf(
                lon, lat, analysis/10., levels=15,
                cmap='viridis', transform=ccrs.PlateCarree()
            )
        contour = ax.contour(lon, lat, analysis/10.,
                             levels=15, colors='black',
                            linewidths=0.5, linestyles='solid',
                            transform=ccrs.PlateCarree())
        ax.clabel(contour, fontsize=10)
        ax.coastlines(color='black', linewidth=0.5)
        ax.set_extent([REGION_WEST, REGION_EAST, REGION_SOUTH, REGION_NORTH])
        ax.set_title('Analysis')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xticks(np.arange(np.min(lon), np.max(lon)+1, 20))
        ax.set_yticks(np.arange(np.min(lat), np.max(lat)+1, 10))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
        ax.set_xlim(REGION_WEST, REGION_EAST)  # Approximate longitudes for North Atlantic/UK
        ax.set_ylim(REGION_SOUTH,REGION_NORTH)   # Approximate latitudes for North Atlantic/U
        fig.colorbar(im, ax=ax, shrink=0.8, label='Geopotential Height (dam)')

        # Calculate increment
        increment = analysis - background
        ax = fig.add_subplot(gs[3], projection=ccrs.PlateCarree())
        norm = mcolors.CenteredNorm(vcenter=0, halfrange=1.0)
        im = ax.pcolormesh(
                lon, lat, increment/10., norm=norm,
                cmap='RdBu_r', transform=ccrs.PlateCarree()
            )
        ax.scatter(
                lon[obs_x], lat[obs_y], c='grey', s=3,
                edgecolor='k', alpha=0.7, transform=ccrs.PlateCarree()
            )
        ax.coastlines(color='black', linewidth=0.5)
        ax.set_extent([REGION_WEST, REGION_EAST, REGION_SOUTH, REGION_NORTH])
        ax.set_title('Analysis Increment')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xticks(np.arange(np.min(lon), np.max(lon)+1, 20))
        ax.set_yticks(np.arange(np.min(lat), np.max(lat)+1, 10))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
        ax.set_xlim(REGION_WEST, REGION_EAST)  # Approximate longitudes for North Atlantic/UK
        ax.set_ylim(REGION_SOUTH,REGION_NORTH)   # Approximate latitudes for North Atlantic/U
        fig.colorbar(im, ax=ax, shrink=0.8, label='Analysis Increment (dam)')

        plt.tight_layout()
        plt.show()

    # Create interactive slider for lengthscale L
    # # Create sliders
    bkg_corr_slider = widgets.FloatSlider(
        min=0., max=3000.0, step=100, value=1000.0,
        description='Background error lengthscale:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')  # Set explicit width in pixels
    )

    loc_slider = widgets.FloatSlider(
        min=100.0, max=1000.0, step=10, value=500.0,
        description='Localisation radius:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )

    obs_err_slider = widgets.FloatSlider(
        min=0., max=2.0, step=0.1, value=1.0,
        description='Observation error variance:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )

    out = widgets.interactive_output(update_analysis, 
                                     {'loc_radius': loc_slider, 
                                     'obs_error': obs_err_slider, 
                                     'bkg_corr_scale': bkg_corr_slider
                                     }
                                     )
    outbox = widgets.VBox([loc_slider, obs_err_slider, bkg_corr_slider, out])
    IPython.display.display(outbox)


if __name__ == "__main__":
    # read reanalysis data
    DATA_DIR = 'working/da_practical_data'
    # plot_correlation(DATA_DIR)
    interactive_da_params(DATA_DIR)