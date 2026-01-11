# Requirements:
# pip install xarray pandas numpy scikit-learn scikit-learn-extra

import sys
import xarray as xr
import numpy as np
import pandas as pd

from sklearn_extra.cluster import KMedoids

from utils import *
from scipy.stats import ks_2samp
from joblib import Parallel, delayed

def ks_distance_column_vs_all(col_idx, data):
    """
    Compute KS distance between column `col_idx` and all other columns.
    Returns a 1D array of distances.
    """
    x = data[:, col_idx]
    x = x[~np.isnan(x)]
    n_cols = data.shape[1]
    dists = np.zeros(n_cols)

    print(col_idx, flush=True)

    for j in range(n_cols):
        if j == col_idx:
            dists[j] = 0.0
            continue
        y = data[:, j]
        y = y[~np.isnan(y)]
        stat, _ = ks_2samp(x, y)
        dist = stat
        dists[j] = dist

    return dists

def ks_distance_matrix_parallel(df, n_jobs=-1):
    """
    Compute KS distance matrix between all columns in parallel.
    df: pandas DataFrame with shape (n_samples, n_columns)
    n_jobs: number of parallel jobs (-1 uses all CPUs)
    """
    data = df.to_numpy()
    n_cols = data.shape[1]

    results = Parallel(n_jobs=n_jobs)(
        delayed(ks_distance_column_vs_all)(i, data) for i in range(n_cols)
    )

    # Stack results into a symmetric distance matrix
    D = np.vstack(results)
    np.save("/u/dssc/mzampar/scratch/UL_Project_Tests/temp.npy", D)
    return D

def ks_distance_matrix(df):
    """
    Compute KS distance between each pair of columns in a pandas DataFrame.
    Returns a symmetric (M x M) matrix, where M = number of columns.
    """
    data = df.to_numpy()
    n_cols = data.shape[1]
    D = np.zeros((n_cols, n_cols))

    for i in range(n_cols):
        print(i, flush=True)
        x = data[:, i]
        x = x[~np.isnan(x)]

        for j in range(i + 1, n_cols):
            y = data[:, j]
            y = y[~np.isnan(y)]

            if len(x) < 2 or len(y) < 2:
                dist = 1.0  # max distance if insufficient data
            else:
                stat, _ = ks_2samp(x, y)
                dist = stat  # KS statistic is the distance

            D[i, j] = dist
            D[j, i] = dist

    return D


def cluster_gridpoints_by_mean_correlation(
    ds,
    n_clusters,
    variables=None,
    time_dim='time',
    y_dim=None,
    x_dim=None,
    min_periods=3,
    random_state=0,
    kmedoids_init='k-medoids++'  # or 'random'
):
    """
    Cluster gridpoints using the mean Pearson correlation (across variables) as similarity.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing variables with common time and spatial dims.
    n_clusters : int
        Number of clusters for K-Medoids.
    variables : list or None
        List of variable names to use. If None, use all data variables in ds.
    time_dim : str
        Name of the time dimension in ds.
    y_dim, x_dim : str or None
        Names of the two spatial dims. If None, the function will try to auto-detect
        (common patterns: ('lat','lon'), ('y','x'), ('south_north','west_east')).
    min_periods : int
        Minimum number of overlapping non-NaN time points required for a pairwise correlation.
    random_state : int
        Random seed for reproducibility.
    kmedoids_init : str
        Initialization for KMedoids ('k-medoids++' or 'random').
        
    Returns
    -------
    labels_da : xarray.DataArray
        DataArray shaped (y_dim, x_dim) with integer cluster labels [0..n_clusters-1].
    medoid_indices : ndarray
        1-D array with the index (0..N-1) of each medoid in the stacked-points ordering.
    distance_matrix : ndarray
        The final NxN distance matrix used (N = number of gridpoints).
    """
    # --------- sanity & auto-detect dims ----------
    if variables is None:
        variables = list(ds.data_vars)
    else:
        variables = list(variables)
    if len(variables) == 0:
        raise ValueError("No variables selected.")
    # stack gridpoints into single axis
    # ensure variables have these dims
    # Use stack 'points' -> length N
    sample_var = variables[0]
    if time_dim not in ds[sample_var].dims:
        raise ValueError(f"time_dim '{time_dim}' not found in variable '{sample_var}' dims.")
    stacked_coords_name = 'points'
    # Stack using xarray so we keep coords order
    # We'll stack (y_dim, x_dim) into points
    # Build a mask for points that are all-NaN across time for all variables (we'll drop them)
    # But first stack per variable as DataFrame to compute pairwise correlations
    # We'll accumulate correlations as floats in a sum matrix
    # Determine N
    # Use coords for final unstack
    # Stack on a copy of ds to avoid modifying input
    ds_stacked = ds.copy()
    ds_stacked = ds_stacked.stack(**{stacked_coords_name: (y_dim, x_dim)})

    print(ds_stacked['lon'])
    print(ds_stacked['lat'])

    # now points dimension size:
    N = ds_stacked.sizes[stacked_coords_name]
    if N == 0:
        raise ValueError("No gridpoints found after stacking.")
    # collect index labels for points to restore shape later
    point_index = ds_stacked[stacked_coords_name].values  # ( (y0,x0), (y0,x1), ... )
    # prepare accumulator for correlation sums
    corr_sum = np.zeros((N, N), dtype=float)
    var_count = 0
    # For each variable compute pairwise correlation matrix (NxN)
    for var in variables:
        print(var)
        if var not in ds_stacked:
            raise ValueError(f"Variable '{var}' not in dataset.")
        da = ds_stacked[var]  # dims: time, points
        # Ensure data shape (time, N)
        if stacked_coords_name not in da.dims:
            raise ValueError(f"Variable '{var}' has no stacked points dimension '{stacked_coords_name}'.")
        # Convert to pandas DataFrame: index=time, columns=point labels
        # Using .T will produce (points x time) so take transpose appropriately
        arr = da.values  # shape (T, N)
        if arr.ndim != 2:
            # if variable has extra dims, try to reduce (e.g., level) by taking first or error
            raise ValueError(f"Variable '{var}' must be 2D over (time, points) after stacking. Got shape {arr.shape}")
        T = arr.shape[0]
        if T < 2:
            raise ValueError("Need at least 2 time points to compute correlations.")
        # Create dataframe: rows=time, cols=point indices (0..N-1)
        df = pd.DataFrame(arr, index=ds_stacked[time_dim].values, columns=np.arange(N))
        print(df)
        # Compute pairwise Pearson corr with pairwise NaN handling
        # pandas.DataFrame.corr uses pairwise complete observations by default
        corr = df.corr(min_periods=min_periods).values  # NxN
        # corr may have NaNs for pairs lacking overlap -> set those to 0 (or small value). Here set to 0 correlation.
        corr = np.nan_to_num(corr, nan=0.0)
        #corr[corr < 0] = 0

        #corr = ks_distance_matrix(df)
        #corr = ks_distance_matrix_parallel(df)
        #corr = np.load("/u/dssc/mzampar/scratch/UL_Project_Tests/temp.npy")

        print(corr)

        # accumulate
        corr_sum += corr
        var_count += 1

    if var_count == 0:
        raise ValueError("No variables were processed.")
    mean_corr = corr_sum / float(var_count)

    # Convert correlation to a distance: here use d = 1 - corr (so corr=1 -> d=0; corr=-1 -> d=2)
    distance = 1.0 - mean_corr
    #distance = mean_corr

    # Ensure diagonal is zero
    np.fill_diagonal(distance, 0.0)

    # Apply K-Medoids with precomputed distances
    kmed = KMedoids(n_clusters=n_clusters, metric='precomputed',
                    init=kmedoids_init, random_state=random_state)
    kmed.fit(distance)
    labels = kmed.labels_  # length N
    medoid_indices = np.asarray(kmed.medoid_indices_, dtype=int)

    ny, nx = ds["lat"].values.shape
    
    # Reshape labels back to 2D using the same indexing order used by stack()
    labels_2d = labels.reshape(ny, nx)
    lat=ds_stacked['lat'].values.reshape(ny,nx)
    lon=ds_stacked['lon'].values.reshape(ny,nx)
    
    return labels_2d, lat, lon, medoid_indices, distance


def main():
    # Parse command line arguments
    files = sys.argv[1].split(" ") 
    vars_list = sys.argv[2].split(" ") 
    train_start, train_end = sys.argv[3].split()
    n_clusters = int(sys.argv[4])
    job_id = sys.argv[5]

    # Read and merge all files
    print("Loading files...")
    ds = xr.open_mfdataset(files, combine="by_coords")
    ds = ds.sel(time=slice(train_start, train_end))

    # Convert to np.datetime64
    train_start = np.datetime64(train_start)
    train_end   = np.datetime64(train_end)
    
    print("Files merged.")
    lon = ds["lon"].values
    lat = ds["lat"].values
    # Select only desired variables
    ds = ds[vars_list]
    
    # Compute mean for each (year, month)
    ds = ds.resample(time="1M").mean()
    
    labels_2d, lat, lon, medoid_indices, distance = cluster_gridpoints_by_mean_correlation(
    ds,
    n_clusters=n_clusters,
    variables=vars_list,
    time_dim='time',
    y_dim='south_north',
    x_dim='west_east',
    min_periods=3,
    random_state=0,
    kmedoids_init='k-medoids++'  # or 'random'
    )

    print("Majority map:", labels_2d.shape)

    outfig=f"cluster_map_k_medoids_{n_clusters}_{'_'.join(vars_list)}_{job_id}.png"

    plot_majority_clusters(labels_2d, lon, lat, outfig, n_clusters, title="Majority Cluster Map")
 
  
if __name__ == "__main__":
    main()



