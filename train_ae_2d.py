import sys
import numpy as np
import torch
from ae_2d import *

def main():
    if len(sys.argv) != 5:
        print("Usage: python main.py <np_array_path> <latent_dim> <vars>")
        print("Example: python main.py ./data/meteorology.npz 128 'temperature humidity wind'")
        sys.exit(1)

    np_array_path, latent_dim, kernel_size, vars_arg = sys.argv[1:]
    latent_dim = int(latent_dim)
    kernel_size = int(kernel_size)
    vars_list_requested = vars_arg.split()

    print(f"Loading data from: {np_array_path}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Requested variables: {vars_list_requested}")

    # ----- Load NumPy data -----
    try:
        loaded = np.load(np_array_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ Failed to load NumPy array: {e}")
        sys.exit(1)

    # Expect a dictionary-like npz file
    if not all(k in loaded for k in ["data", "lon", "lat", "time", "var_names"]):
        print(f"❌ Missing required keys in file. Found keys: {list(loaded.keys())}")
        sys.exit(1)

    data = loaded["data"]          # shape (time, lon, lat, num_vars)
    lon = loaded["lon"]
    lat = loaded["lat"]
    time = loaded["time"]
    all_var_names = list(loaded["var_names"])

    print(f"📊 Original data shape: {data.shape}")
    print(f"📋 Available variables: {all_var_names}")

    # ----- Extract only required variables -----
    var_indices = [all_var_names.index(v) for v in vars_list_requested if v in all_var_names]

    if len(var_indices) == 0:
        print("❌ None of the requested variables were found in var_names.")
        sys.exit(1)

    missing = [v for v in vars_list_requested if v not in all_var_names]
    if missing:
        print(f"⚠️ Warning: Missing variables not found in data: {missing}")

    # Select only chosen variable channels
    data = data[..., var_indices]  # keep only selected vars
    print(f"✅ Subset data shape after selection: {data.shape}")

    num_vars = data.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----- Train the autoencoder -----
    model = train_autoencoder_2d(
        data=data,
        num_vars=num_vars,
        latent_dim=latent_dim,
        epochs=200,
        batch_size=64,
        lr=1e-3,
        kernel_size=kernel_size,
        device=device
    )

    # ----- Save model -----
    var_suffix = "_".join(vars_list_requested)
    out_path = f"/u/dssc/mzampar/scratch/UL_Project_Tests/auto_encoders/autoencoder_{var_suffix}_latent{latent_dim}_kernel_{kernel_size}.pt"
    torch.save(model.state_dict(), out_path)
    print(f"✅ Model saved to {out_path}")
    
    # Suppose `data` has shape (time, lon, lat, num_vars)
    row_index = 100
    data_row = data[row_index]  # shape: (lon, lat, num_vars)
    
    # `vars_list` contains the names of the variables
    image = f"./results/ae/autoencoder_{var_suffix}_latent{latent_dim}_kernel_{kernel_size}_row_{row_index}.png"
    data_row = torch.tensor(data_row)
    plot_ae_reconstruction(model, data_row, vars_arg, image)

if __name__ == "__main__":
    main()
"""
# TODO

define the architecture

train on the entire period

plots of the loss in training

find the dimension in which you get less information loss by checking the final training loss of different dimensions

cluster the reduced data on entire period and on subperiods
how to cluster in the latent space? can we just simply make it a vector? 

apply the plot script

plot the image and the recostructed image

"""
