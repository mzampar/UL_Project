import numpy as np
import pandas as pd
from datetime import datetime
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

def compute_patch_centers(Y, X, patch_size, stride=None):
    """
    Y, X : grid size
    patch_size : size of extracted patches (e.g. 8)
    stride : distance between patch centers (default = patch_size//2)
    """
    half = patch_size // 2
    if stride is None:
        stride = patch_size // 2  # default overlap = 50%
    centers = []
    for i in range(half, Y - half + 1, stride):
        for j in range(half, X - half + 1, stride):
            centers.append((i, j))

    return centers


def make_multi_step_channels(arr, step):
    """
    arr   : (T, Y, X, C)
    step  : integer number of steps before/after t

    Returns:
        (T, Y, X, C * (2*step ))
    """
    T, Y, X, C = arr.shape

    # Offsets: -step ... 0 ... +step
    offsets = list(range(-step, step ))

    # Pad time dimension so indexing never goes out of bounds
    arr_padded = np.pad(
        arr,
        pad_width=((step, step), (0, 0), (0, 0), (0, 0)),
        mode="edge"
    )

    stacked = []
    for off in offsets:
        # Slice from padded array
        start = off + step      # shift because of padding
        stacked.append(arr_padded[start : start + T])

    # Concatenate along channel axis
    return np.concatenate(stacked, axis=-1)

def gen_patches(ds, patch_size, num_steps=3):
    half_patch = patch_size // 2

    # ds has dims: time × south_north × west_east, variables = list of vars
    vars_list = list(ds.data_vars)  # all variables
    arr = np.stack([ds[var].values for var in vars_list], axis=-1)  # shape: (time, y, x, channels)

    T, Y, X, C = arr.shape
    # pad time dimension at start and end
    arr_padded = np.pad(arr, ((1,1),(0,0),(0,0),(0,0)), mode='edge')  # shape: (T+2, Y, X, C)
    arr_chan = make_multi_step_channels(arr, step=num_steps)

    patch_centers = compute_patch_centers(Y, X, patch_size, stride=patch_size//4)
    #patch_centers = compute_patch_centers(Y, X, patch_size, stride=1)

    # initialize list to store patches
    patches = []

    for t in range(T):
        dt = pd.Timestamp(ds['time'].values[t])  # convert to datetime
        if dt.month != 6:
            continue
        for (i,j) in patch_centers:
            patch = arr_chan[
                t,
                i-half_patch:i+half_patch,
                j-half_patch:j+half_patch,
                :
            ]
            patches.append(patch)

    # convert to numpy array
    patches = np.array(patches)
    print("Patches shape:", patches.shape)
    # Reshape to (N, C, H, W)
    patches = np.transpose(patches, (0, 3, 1, 2))  # (time, vars, lon, lat)
    return patches, patch_centers

def reconstruct_cluster_maps(clusters, patch_centers, patch_size, Y, X):
    """
    clusters: array of length (T * n_centers)
    patch_centers: list of (i, j) positions
    patch_size: int
    Y, X: output spatial shape (same as lat/lon)

    Returns:
        cluster_maps: (T, Y, X) with NaN where no patch covers
    """

    half = patch_size // 2
    n_centers = len(patch_centers)

    # --- infer T automatically ---
    if clusters.shape[0] % n_centers != 0:
        raise ValueError("clusters length must be T * n_centers")

    T = clusters.shape[0] // n_centers

    # initialize output
    cluster_maps = np.full((T, Y, X), np.nan)

    idx = 0

    for t in range(T):
        for (i, j) in patch_centers:

            cluster_val = clusters[idx]
            idx += 1

            # compute patch bounds
            y1, y2 = i - half, i + half
            x1, x2 = j - half, j + half

            # ensure bounds do not exceed array
            y1c, y2c = max(0, y1), min(Y, y2)
            x1c, x2c = max(0, x1), min(X, x2)

            cluster_maps[t, y1c:y2c, x1c:x2c] = cluster_val

    return cluster_maps


def compute_majority_cluster(cluster_maps):
    """
    cluster_maps: (T, Y, X) with NaN for uncovered points

    Output:
      majority: (Y, X) array of cluster IDs
    """

    T, Y, X = cluster_maps.shape
    majority = np.full((Y, X), np.nan)

    for y in range(Y):
        for x in range(X):
            vals = cluster_maps[:, y, x]
            vals = vals[~np.isnan(vals)]   # remove NaN (uncovered)

            if len(vals) == 0:
                print(f'no values for point {x}-{y}')
                continue  # remains nan

            # compute most frequent cluster
            unique, counts = np.unique(vals, return_counts=True)
            majority[y, x] = unique[np.argmax(counts)]

    return majority

def plot_majority_clusters(
    majority_map,
    lon,
    lat,
    outfig,
    n_clusters,
    title="Majority Cluster Map",
    show_orography=True,
    orography_source="stadia" 
):

    fig = plt.figure(figsize=(11, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # --- TILE SELECTION ---
    tiles = None
    if show_orography:
        if orography_source == "stadia":
            # The correct replacement for Stamen Terrain
            #tiles = cimgt.StadiaMapsTiles(style="stamen_terrain_background")
            tiles = cimgt.StadiaMapsTiles(
            style="stamen_terrain_background",
            apikey="9f50bc16-63a2-4e1a-b495-f866773b6c1d"
            )
        elif orography_source == "osm":
       

            tiles = cimgt.OSM()

    # --- Add image tiles (terrain background) ---
    if tiles is not None:
        try:
            ax.add_image(tiles, 8)
        except Exception as e:
            print("Could not load terrain tiles:", e)

    # --- Base map features ---
    ax.coastlines(resolution="10m", linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    ax.add_feature(cfeature.LAKES, edgecolor="black", facecolor="lightblue")
    ax.add_feature(cfeature.RIVERS, linewidth=0.4)

    if not show_orography:
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

    # --- Extent ---
    buffer = 0.1
    ax.set_extent([
        lon.min() - buffer, lon.max() + buffer,
        lat.min() - buffer, lat.max() + buffer
    ])

    # --- Discrete cmap ---
    cmap = plt.get_cmap("tab20", n_clusters)
    bounds = np.arange(n_clusters + 1) - 0.5
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # --- Cluster map ---
    pcm = ax.pcolormesh(
        lon,
        lat,
        majority_map,
        cmap=cmap,
        norm=norm,
        shading="auto",
        alpha=0.55 if show_orography else 1.0,
        zorder=5
    )

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray",
                      alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    #ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(outfig, dpi=300)
    plt.close()
    print(f"Saved {outfig}")

"""
def preprocess_dataset(ds_train, ds_test, log_var="dif_precip_g"):
    #Applies:
    #- log transform to log_var (in both train & test)
    #- global mean/std normalization:
    #      x_norm = (x - mean_global) / std_global
    #  where mean/std are computed over the entire time+space domain.

    ds_train, ds_test must contain exact same variables.

    ds_train_proc = ds_train.copy()
    ds_test_proc = ds_test.copy()

    for var in ds_train.data_vars:

        # ---------- 1. Optional log-transform  ----------
        if var == log_var:
            # avoid log(0)
            ds_train_proc[var] = np.log(ds_train_proc[var] + 1e-6)
            ds_test_proc[var]  = np.log(ds_test_proc[var]  + 1e-6)

        # ---------- 2. Compute global mean & std ----------
        # flatten over all dims except variable name
        vals_train = ds_train_proc[var].values

        mean_global = np.nanmean(vals_train)      # scalar
        std_global  = np.nanstd(vals_train) + 1e-6  # avoid division by zero

        # ---------- 3. Apply standardization ----------
        ds_train_proc[var] = (ds_train_proc[var] - mean_global) / std_global
        ds_test_proc[var]  = (ds_test_proc[var]  - mean_global) / std_global

        print(f"Variable: {var:20s}  mean={mean_global:.4f}  std={std_global:.4f}")

    return ds_train_proc, ds_test_proc
"""

def preprocess_dataset_monthly_minmax(ds_train, ds_test):
    """
    Monthly min-max normalization (global min/max per month, per variable).

    For each variable:
        For each month (1..12):
            Compute GLOBAL min/max over:
                time (subset to month), south_north, west_east
            using TRAIN dataset only.

    Apply same normalization to both train and test:
        x_scaled = (x - min) / (max - min)

    Returns:
        ds_train_norm, ds_test_norm
    """

    ds_train_proc = ds_train.copy()
    ds_test_proc  = ds_test.copy()

    months = np.arange(1, 13)

    stats = {var: {} for var in ds_train.data_vars}

    # Compute monthly min/max
    for var in ds_train.data_vars:
        print(f"\nComputing monthly global min/max for {var}...")

        for month in months:
            sel = ds_train[var].sel(time=ds_train["time.month"] == month)

            if sel.time.size == 0:
                raise ValueError(f"No training data found for month {month} in {var}")

            # Global min/max over entire grid + time in that month
            min_val = float(sel.min(dim=("time", "south_north", "west_east")))
            max_val = float(sel.max(dim=("time", "south_north", "west_east")))

            stats[var][month] = (min_val, max_val)

            print(f"  Month {month}: min={min_val:.4f}, max={max_val:.4f}")

    # Apply min-max scaling
    def apply_minmax(ds, name):
        ds_scaled = ds.copy()
        print(f"\nApplying monthly min/max scaling to {name} dataset...")

        for var in ds.data_vars:
            for month in months:

                min_v, max_v = stats[var][month]

                mask = (ds["time.month"] == month)

                # Normalize only the selected month
                ds_scaled[var].loc[dict(time=mask)] = (
                    ds[var].loc[dict(time=mask)] - min_v
                ) / (max_v - min_v)

                if name == 'TEST':
                    ds_scaled[var].loc[dict(time=mask)] = ds_scaled[var].loc[dict(time=mask)].clip(min=0.0, max=1.0)

        return ds_scaled

    ds_train_norm = apply_minmax(ds_train_proc, "TRAIN")
    ds_test_norm  = apply_minmax(ds_test_proc,  "TEST")


    return ds_train_norm, ds_test_norm



def plot_history(history, test_loss_dict, outfig):
    """
    history: dict with keys:
        'epoch', 'train_loss', 'train_recon', 'train_cluster',
        'val_loss', 'val_recon', 'val_cluster'

    test_loss_dict:
        {
            'test_loss': float,
            'test_recon': float,
            'test_cluster': float
        }

    outfig: str → path to output figure
    """

    epochs = history["epoch"]
    last_idx = -1  # last row
    train_recon_last   = history["train_recon"].iloc[last_idx]
    val_recon_last     = history["val_recon"].iloc[last_idx]
    test_recon         = test_loss_dict["test_recon"]
    
    train_cluster_last = history["train_cluster"].iloc[last_idx]
    val_cluster_last   = history["val_cluster"].iloc[last_idx]
    test_cluster       = test_loss_dict["test_cluster"]
    
    train_loss_last    = history["train_loss"].iloc[last_idx]
    val_loss_last      = history["val_loss"].iloc[last_idx]
    test_loss          = test_loss_dict["test_loss"]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle("Training History", fontsize=18)

    ax = axes[0]
    ax.plot(epochs, history["train_recon"],
            label=f"Train Recon (last={train_recon_last:.4f})")
    ax.plot(epochs, history["val_recon"],
            label=f"Val Recon (last={val_recon_last:.4f})")

    ax.axhline(test_recon, color="black", linestyle="--",
               label=f"Test Recon = {test_recon:.4f}")

    ax.set_title("Reconstruction Loss")
    ax.set_xlabel("Epoch")
    #ax.set_ylabel("Loss")
    ax.grid(True)
    ax.set_yscale("log")   # logarithmic y-axis
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, history["train_cluster"],
            label=f"Train Cluster (last={train_cluster_last:.4f})")
    ax.plot(epochs, history["val_cluster"],
            label=f"Val Cluster (last={val_cluster_last:.4f})")

    ax.axhline(test_cluster, color="black", linestyle="--",
               label=f"Test Cluster = {test_cluster:.4f}")

    ax.set_title("Cluster Loss")
    ax.set_xlabel("Epoch")
    ax.set_yscale("log")   # logarithmic y-axis
    ax.grid(True)
    ax.legend()

    ax = axes[2]
    ax.plot(epochs, history["train_loss"],
            label=f"Train Total (last={train_loss_last:.4f})")
    ax.plot(epochs, history["val_loss"],
            label=f"Val Total (last={val_loss_last:.4f})")

    ax.axhline(test_loss, color="black", linestyle="--",
               label=f"Test Total = {test_loss:.4f}")

    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.grid(True)
    ax.set_yscale("log")   # logarithmic y-axis
    ax.legend()

    plt.tight_layout()
    plt.savefig(outfig, dpi=150)
    plt.close()

