import os

import cartopy.crs as ccrs  # type: ignore
import matplotlib.gridspec as mgs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr

import ipywidgets as widgets  # type: ignore
import IPython.display

# Define our region of interest - North Atlantic/UK region
REGION_EAST = 10
REGION_WEST = -30
REGION_NORTH = 65
REGION_SOUTH = 40

def get_data(DATA_DIR):
    # Load pre-downloaded ERA5 ensemble mean and spread
    # Extract ensemble mean
    data = xr.open_dataset(os.path.join(DATA_DIR, 'era5_data', 'ensemble_mean.nc'))
    lat = data.latitude
    lon = data.longitude
    ens_mean = data['z'][0,0,:,:].values / 9.80665

    # Extract ensemble spread
    data = xr.open_dataset(os.path.join(DATA_DIR, 'era5_data', 'ensemble_spread.nc'))
    ens_spread = data['z'][0,0,:,:].values / 9.80665

    # Extract ensemble spread
    data = xr.open_dataset(os.path.join(DATA_DIR, 'era5_data', 'ensemble_members.nc'))
    n_ens = 10
    ensemble_members = np.zeros((n_ens, len(lat), len(lon)))
    ensemble_members = data['z'][:,0,0,:,:].values / 9.80665

    # Create xarray dataset
    ds_t3 = xr.Dataset(
        data_vars={
            "z_mean": (["latitude", "longitude"], ens_mean),
            "z_spread": (["latitude", "longitude"], ens_spread),
            "z_members": (["member", "latitude", "longitude"], ensemble_members),
        },
        coords={
            "latitude": lat,
            "longitude": lon,
            "member": np.arange(1, n_ens+1),
        }
    )

    return ds_t3


def plot_ensemble(ds_t3):
    # Visualize ensemble mean and spread
    fig = plt.figure('ensemble', figsize=(16, 5))
    axes = fig.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot ensemble mean
    im1 = axes[0].contourf(
        ds_t3.longitude, ds_t3.latitude, ds_t3.z_mean/10.,
        levels=15, cmap='viridis', transform=ccrs.PlateCarree()
    )
    contour = axes[0].contour(ds_t3.longitude, ds_t3.latitude, ds_t3.z_mean/10.,
                         levels=15, colors='black',
                        linewidths=0.5, linestyles='solid',
                        transform=ccrs.PlateCarree())
    axes[0].clabel(contour, fontsize=10)
    axes[0].set_extent([REGION_WEST, REGION_EAST, REGION_SOUTH, REGION_NORTH])
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].set_xticks(np.arange(np.min(ds_t3.longitude), np.max(ds_t3.longitude)+1, 20))
    axes[0].set_yticks(np.arange(np.min(ds_t3.latitude), np.max(ds_t3.latitude)+1, 10))
    axes[0].xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
    axes[0].set_xlim(REGION_WEST, REGION_EAST)  # Approximate longitudes for North Atlantic/UK
    axes[0].set_ylim(REGION_SOUTH,REGION_NORTH)   # Approximate latitudes for North Atlantic/U
    axes[0].coastlines(color='black', linewidth=0.5)
    axes[0].set_title('ERA5 Ensemble Mean - 500hPa Geopotential', fontsize=14)
    plt.colorbar(im1, ax=axes[0], shrink=0.8, label='Geopotential Height (dam)')

    # Plot ensemble spread
    im2 = axes[1].contourf(
        ds_t3.longitude, ds_t3.latitude, ds_t3.z_spread/10.,
        levels=15, cmap='magma', transform=ccrs.PlateCarree()
    )
    contour = axes[1].contour(ds_t3.longitude, ds_t3.latitude, ds_t3.z_spread/10.,
                         levels=5, colors='black',
                        linewidths=0.5, linestyles='solid',
                        transform=ccrs.PlateCarree())
    axes[1].clabel(contour, fontsize=10)
    axes[1].set_extent([REGION_WEST, REGION_EAST, REGION_SOUTH, REGION_NORTH])
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].set_xticks(np.arange(np.min(ds_t3.longitude), np.max(ds_t3.longitude)+1, 20))
    axes[1].set_yticks(np.arange(np.min(ds_t3.latitude), np.max(ds_t3.latitude)+1, 10))
    axes[1].xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
    axes[1].set_xlim(REGION_WEST, REGION_EAST)  # Approximate longitudes for North Atlantic/UK
    axes[1].set_ylim(REGION_SOUTH,REGION_NORTH)   # Approximate latitudes for North Atlantic/U
    axes[1].coastlines(color='black', linewidth=0.5)
    axes[1].set_title('ERA5 Ensemble Spread - 500hPa Geopotential', fontsize=14)
    plt.colorbar(im2, ax=axes[1], shrink=0.8, label='Ensemble Spread (dam)')

    plt.tight_layout()
    plt.show()


# Create interactive ensemble size emulation
def changing_ensemble_size(ds_t3):
    # Extract data
    members_data = ds_t3.z_members.values

    def do_plot(ensemble_size):
        # Select random subset of ensemble members
        if ensemble_size < 10:
            rng = np.random.default_rng(42)
            selected_members = sorted(rng.choice(10, size=ensemble_size, replace=False))
        else:
            selected_members = list(range(10))
        # Extract subset
        subset_members = members_data[selected_members, :, :]
        # Calculate spread from subset
        subset_spread = np.std(subset_members, axis=0)
        full_spread = ds_t3.z_spread.values
        spread_diff = subset_spread - full_spread

        fig = plt.figure('ensemble_size', figsize=(25, 12))
        fig.clf()
        gs = mgs.GridSpec(4, 15, figure=fig)
        # Only show as many plots as members selected
        for i in selected_members:
            j = i//5
            k = i%5
            ax = fig.add_subplot(gs[j, 3*k:3*(k+1)], projection= ccrs.PlateCarree())
            im = ax.contourf(
                ds_t3.longitude, ds_t3.latitude, members_data[i, :, :]/10.,
                levels=15, cmap='viridis', transform=ccrs.PlateCarree()
            )
            contour = ax.contour(ds_t3.longitude, ds_t3.latitude, members_data[i, :, :]/10.,
                                 levels=15, colors='black',
                                linewidths=0.5, linestyles='solid',
                                transform=ccrs.PlateCarree())
            ax.clabel(contour, fontsize=10)
            ax.set_extent([REGION_WEST, REGION_EAST, REGION_SOUTH, REGION_NORTH])
            ax.grid(True, linestyle='--', alpha=0.7)
            # ax.set_xticks(np.arange(np.min(ds_t3.longitude), np.max(ds_t3.longitude)+1, 20))
            # ax.set_yticks(np.arange(np.min(ds_t3.latitude), np.max(ds_t3.latitude)+1, 10))
            # ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
            # ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
            ax.set_xlim(REGION_WEST, REGION_EAST)  # Approximate longitudes for North Atlantic/UK
            ax.set_ylim(REGION_SOUTH,REGION_NORTH)   # Approximate latitudes for North Atlantic/U
            ax.coastlines(color='black', linewidth=0.5)
            ax.set_title(f'Member {i+1}')
            # plt.colorbar(im, ax=axes[i], shrink=0.3, label='Geopotential Height (dam)')

        # Plot spread comparison
        ax = fig.add_subplot(gs[2:4, 0:5], projection=ccrs.PlateCarree())
        # Plot spread from subset
        levels = np.linspace(0, np.max(full_spread)/10., 15)
        im1 = ax.contourf(
            ds_t3.longitude, ds_t3.latitude, subset_spread/10.,
            levels=levels, cmap='magma', transform=ccrs.PlateCarree()
        )
        ax.set_extent([REGION_WEST, REGION_EAST, REGION_SOUTH, REGION_NORTH])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xticks(np.arange(np.min(ds_t3.longitude), np.max(ds_t3.longitude)+1, 20))
        ax.set_yticks(np.arange(np.min(ds_t3.latitude), np.max(ds_t3.latitude)+1, 10))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
        ax.set_xlim(REGION_WEST, REGION_EAST)  # Approximate longitudes for North Atlantic/UK
        ax.set_ylim(REGION_SOUTH,REGION_NORTH)   # Approximate latitudes for North Atlantic/U
        ax.coastlines(color='black', linewidth=0.5)
        ax.set_title(f'Spread from {ensemble_size} Members', fontsize=14)
        fig.colorbar(im1, ax=ax, shrink=0.8, label='Ensemble Spread (dam)')

        ax = fig.add_subplot(gs[2:4, 5:10], projection=ccrs.PlateCarree())
        im2 = ax.contourf(
            ds_t3.longitude, ds_t3.latitude, full_spread/10.,
            levels=levels, cmap='magma', transform=ccrs.PlateCarree()
        )
        ax.set_extent([REGION_WEST, REGION_EAST, REGION_SOUTH, REGION_NORTH])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xticks(np.arange(np.min(ds_t3.longitude), np.max(ds_t3.longitude)+1, 20))
        ax.set_yticks(np.arange(np.min(ds_t3.latitude), np.max(ds_t3.latitude)+1, 10))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
        ax.set_xlim(REGION_WEST, REGION_EAST)  # Approximate longitudes for North Atlantic/UK
        ax.set_ylim(REGION_SOUTH,REGION_NORTH)   # Approximate latitudes for North Atlantic/U
        ax.coastlines(color='black', linewidth=0.5)
        ax.set_title(f'Spread from All 10 Members (Reference)', fontsize=14)
        fig.colorbar(im2, ax=ax, shrink=0.8, label='Ensemble Spread (dam)')
        fig.tight_layout()

        ax = fig.add_subplot(gs[2:4, 10:15], projection=ccrs.PlateCarree())
        # Create a diverging colormap for difference
        levels = np.linspace(-0.2, 0.2, 21)
        im = ax.contourf(
            ds_t3.longitude, ds_t3.latitude, spread_diff/10.,
            levels=levels, cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree()
        )
        # contour = ax.contour(ds_t3.longitude, ds_t3.latitude, spread_diff/10.,
        #                      levels=levels, colors='black',
        #                     linewidths=0.5, linestyles='solid',
        #                     transform=ccrs.PlateCarree())
        # ax.clabel(contour, fontsize=10)
        ax.set_extent([REGION_WEST, REGION_EAST, REGION_SOUTH, REGION_NORTH])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xticks(np.arange(np.min(ds_t3.longitude), np.max(ds_t3.longitude)+1, 20))
        ax.set_yticks(np.arange(np.min(ds_t3.latitude), np.max(ds_t3.latitude)+1, 10))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
        ax.set_xlim(REGION_WEST, REGION_EAST)  # Approximate longitudes for North Atlantic/UK
        ax.set_ylim(REGION_SOUTH,REGION_NORTH)   # Approximate latitudes for North Atlantic/U
        ax.coastlines(color='black', linewidth=0.5)
        ax.set_title(f'Spread Difference: {ensemble_size} Members - Full Ensemble', fontsize=14)
        fig.colorbar(im, ax=ax, shrink=0.8, label='Spread Difference (dam)')

        # Print statistics
        mean_error = np.mean(np.abs(spread_diff))
        max_error = np.max(np.abs(spread_diff))
        ax.annotate(f"Mean Abs Error: {mean_error:.2f} dam\nMax Abs Error: {max_error:.2f} dam",
                    xy=(0.05, 0.05), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.show()

    # Create interactive slider for lengthscale L
    int_slider = widgets.IntSlider(value=10,
        min=1,
        max=10,
        step=1,
        description='Ensemble size',
        layout=widgets.Layout(width='500px'),
        style={'description_width': 'initial'},
    )
    out = widgets.interactive_output(do_plot, {'ensemble_size': int_slider})
    outbox = widgets.VBox([int_slider, out])
    IPython.display.display(outbox)

if __name__ == '__main__':
    DATA_DIR = 'working/da_practical_data'
    ds = get_data(DATA_DIR)
    # plot_ensemble(ds)
    changing_ensemble_size(ds)