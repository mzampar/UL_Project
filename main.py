import sys
import os
import xarray as xr
import numpy as np
import pandas as pd
import torch.nn.functional as F
from ae_2d import *
from utils import *
from time import time

def main():
    # Parse command line arguments
    files = sys.argv[1].split(" ") 
    vars_list = sys.argv[2].split(" ")
    train_start, train_end = sys.argv[3].split()
    test_start, test_end = sys.argv[4].split()
    n_clusters = int(sys.argv[5])
    job_id = int(sys.argv[6])

    # Read and merge all files
    print("Loading files...")
    ds = xr.open_mfdataset(files, combine="by_coords")

    train_start = np.datetime64(train_start)
    train_end   = np.datetime64(train_end)
    test_start  = np.datetime64(test_start)
    test_end    = np.datetime64(test_end)
    
    print("Files merged.")

    lon = ds["lon"].values
    lat = ds["lat"].values
    ds = ds[vars_list]
    
    
    # Compute mean for each (year, month)
    ds = ds.resample(time="1M").mean()
    """
    if vars_list == ['dif_precip_g']:
        ds = ds.resample(time="1M").sum()
    else:
        ds = ds.resample(time="1M").mean()
    # Apply log to precipitation after resampling
    if 'dif_precip_g' in ds:
        ds['dif_precip_g'] = np.log(ds['dif_precip_g'].clip(min=1))
    """

    # Select training and testing datasets
    ds_train = ds.sel(time=slice(train_start, train_end))
    ds_test  = ds.sel(time=slice(test_start, test_end))


    #ds_train, ds_test = preprocess_dataset(ds_train, ds_test)
    ds_train, ds_test = preprocess_dataset_monthly_minmax(ds_train, ds_test)

    print(ds)
    
    patch_size = 8  # spatial patch size
    #patch_size = 16 # spatial patch size
    num_steps = 6 # to get 6+6 steps (12 months)
    patches_test, centers_test = gen_patches(ds_test, patch_size, num_steps=num_steps)
    patches_train, centers_train = gen_patches(ds_train, patch_size, num_steps=num_steps)

    print('centers shape')
    print(len(centers_test))
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_vars = len(vars_list)
    latent_dim = 512
    kernel_size = 3
    batch_size = 1024
    batch_size = 512
    #batch_size = 256
    epochs = 400
    epochs = 800
    lr = 1e-4
    tau=0.1
    tau=0.05
    sink_eps=0.115
    sink_eps=0.05
    sink_eps=0.09
    alpha=0.5
    print("\n===== Training Parameters =====")
    print(f"num_vars    = {num_vars}")
    print(f"latent_dim  = {latent_dim}")
    print(f"kernel_size = {kernel_size}")
    print(f"epochs      = {epochs}")
    print(f"batch_size  = {batch_size}")
    print(f"lr          = {lr}")
    print(f"sink_eps    = {sink_eps}")
    print(f"tau         = {tau}")
    print(f"alpha       = {alpha}")
    print("================================\n")

    model_path = f"../scratch/UL_Project_Tests/ae_{"_".join(vars_list)}_{n_clusters}_{batch_size}_{epochs}_{lr}_{tau}_{sink_eps}_{alpha}_{job_id}.pth"
    
    model = ClusteringAutoencoder(num_vars=num_vars, latent_dim=latent_dim, kernel_size=kernel_size, num_clusters=n_clusters, num_steps=num_steps, tau=tau)
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}, loading...")
        
        # load saved weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

    else:
        print("Model not found. Training...")

        start_time = time()
        # ----- Train the autoencoder -----
        model, history = train_cae(
            data=patches_train,
            num_vars=num_vars,
            model=model,
            latent_dim=latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            alpha=alpha,
            sink_eps=sink_eps
        )
        torch.save(model.state_dict(), model_path)
        end_time = time()
        print(f'Training time: {end_time - start_time}')
    
    # Apply clustering on training set 
    patches_train_t = torch.tensor(patches_train, dtype=torch.float32).to(device)

    # Forward
    x_hat, z, f = model(patches_train_t)
    
    # Soft assignments
    s = torch.matmul(f, model.clusters.T)
    p = F.softmax(s / model.tau, dim=1)
    clusters = p.argmax(dim=1).cpu().numpy()

    cluster_maps = reconstruct_cluster_maps(
        clusters=clusters,
        patch_centers=centers_train,
        patch_size=patch_size,
        Y=lon.shape[0],
        X=lon.shape[1]
    )

    print("Cluster map:", cluster_maps.shape)
    majority_map = compute_majority_cluster(cluster_maps)

    unique_clusters = np.unique(majority_map)
    print("Unique cluster labels:", unique_clusters)

    print("Majority map:", majority_map.shape)

    outfig=f"./fig/cluster_map_cae_{"_".join(vars_list)}_{n_clusters}_{batch_size}_{epochs}_{lr}_{tau}_{sink_eps}_{alpha}_{job_id}_train.png"

    plot_majority_clusters(majority_map, lon, lat, outfig, n_clusters, title="Majority Cluster Map")


    # Apply clustering on test set 
    patches_test_t = torch.tensor(patches_test, dtype=torch.float32).to(device)
    
    # Forward
    x_hat, z, f = model(patches_test_t)
    
    # Soft assignments
    s = torch.matmul(f, model.clusters.T)
    p = F.softmax(s / model.tau, dim=1)
    clusters = p.argmax(dim=1).cpu().numpy()
    
    # Reconstruction loss
    t_recon_loss = F.mse_loss(x_hat, patches_test_t)
    
    # Cluster loss
    s = torch.matmul(f, model.clusters.T)
    Q = sinkhorn(s, epsilon=sink_eps)
    
    log_probs = F.log_softmax(s / tau, dim=1)
    t_cluster_loss = -torch.mean(torch.sum(Q * log_probs, dim=1))
    
    # Total
    loss = alpha * t_recon_loss + (1 - alpha) * t_cluster_loss

    loss = float(loss.cpu()) if hasattr(loss, "cpu") else float(loss)
    t_recon_loss = float(t_recon_loss.cpu()) if hasattr(t_recon_loss, "cpu") else float(t_recon_loss)
    t_cluster_loss = float(t_cluster_loss.cpu()) if hasattr(t_cluster_loss, "cpu") else float(t_cluster_loss)


    print(f'Test    loss: {loss}')
    print(f'Recon   loss: {t_recon_loss}')
    print(f'Cluster loss: {t_cluster_loss}')

    test_loss_dict = {
    'test_loss': loss,
    'test_recon':t_recon_loss,
    'test_cluster': t_cluster_loss
    }

    cluster_maps = reconstruct_cluster_maps(
        clusters=clusters,
        patch_centers=centers_test,
        patch_size=patch_size,
        Y=lon.shape[0],
        X=lon.shape[1]
    )
       
    print("Cluster map:", cluster_maps.shape)
    majority_map = compute_majority_cluster(cluster_maps)

    unique_clusters = np.unique(majority_map)
    print("Unique cluster labels:", unique_clusters)

    print("Majority map:", majority_map.shape)

    outfig=f"./fig/cluster_map_cae_{"_".join(vars_list)}_{n_clusters}_{batch_size}_{epochs}_{lr}_{tau}_{sink_eps}_{alpha}_{job_id}.png"

    plot_majority_clusters(majority_map, lon, lat, outfig, n_clusters, title="Majority Cluster Map")

    outfig=f"./fig/history_train_cae_{"_".join(vars_list)}_{n_clusters}_{batch_size}_{epochs}_{lr}_{tau}_{sink_eps}_{alpha}_{job_id}.png"

    plot_history(history, test_loss_dict, outfig)

    csv = f"./csv/history_train_cae_{"_".join(vars_list)}_{n_clusters}_{batch_size}_{epochs}_{lr}_{tau}_{sink_eps}_{alpha}_{job_id}.csv"
    history.to_csv(csv, index=False)

     
  
if __name__ == "__main__":
    main()



