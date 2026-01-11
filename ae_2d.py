import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd


class ClusteringAutoencoder(nn.Module):
    def __init__(self, num_vars, latent_dim=256, num_clusters=10, kernel_size=3, tau=0.1, num_steps=3):
        super().__init__()
        self.tau = tau
        self.num_clusters = num_clusters

        # padding
        pad = (kernel_size - 1) // 2

        # ----- Encoder -----
        self.encoder = nn.Sequential(
            # (B, 3, 8, 8) → (B, 16, 8, 8)
            nn.Conv2d((2*num_steps)*num_vars, 24, kernel_size=kernel_size, stride=1, padding=pad),
            nn.ReLU(),
            # (B, 16, 8, 8) → (B, 32, 8, 8)
            nn.Conv2d(24, 32, kernel_size=kernel_size, stride=1, padding=pad),
            nn.ReLU(),
            nn.GroupNorm(8, 32),
            # (B, 32, 8, 8) → (B, 32, 8, 8)
            nn.Conv2d(32, 32, kernel_size=kernel_size, stride=1, padding=pad),
            nn.ReLU(),
            # (B, 32, 8, 8) → (B, 64, 4, 4)
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=2, padding=pad),
            nn.ReLU(),
            nn.GroupNorm(4, 64),
            # (B, 64, 4, 4) → (B, 64, 4, 4)
            nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=pad),
            nn.ReLU(),
            # (B, 64, 4, 4) → (B, 128, 2, 2)
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=2, padding=pad),
            nn.ReLU(),
            nn.GroupNorm(2, 128),
            # (B, 128, 2, 2) → (B, 128, 2, 2)
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=pad),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        # MLP for clustering
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
        )

        # Cluster centroids
        self.clusters = nn.Parameter(torch.randn(num_clusters, 512))
        #self.clusters = nn.Parameter(F.normalize(torch.randn(num_clusters, latent_dim), dim=1))

        # ----- Decoder -----
        self.decoder = nn.Sequential(
            # (B, 128, 2, 2) → (B, 64, 4, 4)
            nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            # (B, 64, 4, 4) → (B, 64, 4, 4)
            nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 64),
            # (B, 64, 4, 4) → (B, 32, 8, 8)
            nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            # (B, 32, 8, 8) → (B, 32, 8, 8)
            nn.Conv2d(32, 32, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 32),
            # (B, 32, 8, 8) → (B, 16, 8, 8)
            nn.Conv2d(32, 24, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 24),
            # (B, 16, 8, 8) → (B, 3, 8, 8)
            nn.Conv2d(24, (2*num_steps)*num_vars, kernel_size=kernel_size, stride=1, padding=1)
        )

    def debug_decoder(self, z):
        x = z
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if torch.isnan(x).any():
                print(f"NaN after layer {i}: {layer}")
                return
        print("No NaNs in decoder")
    

    def forward(self, x):
        z1 = self.encoder(x)
        z = self.flatten(z1)

        f = self.mlp(z)  # clustering features

        # reconstruction
        x_hat = self.decoder(z1)

        return x_hat, z, f

def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
    # out: (B, feature_dim)
    with torch.no_grad():

        Q = torch.exp(out / epsilon).t()  # K x B
        B = Q.shape[1] 
        K = Q.shape[0]
    
        sum_Q = torch.sum(Q)
        Q /= sum_Q
    
        for _ in range(sinkhorn_iterations):
            # row normalization
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K
    
            # column normalization
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
    
        Q *= B
        return Q.t()  # back to (B, K)

def train_cae(data, num_vars, model, tau=0.1,
                         latent_dim=256, epochs=10, batch_size=64, lr=1e-3, 
                         alpha=0.6, device=None, sinkhorn_fn=sinkhorn,
                         sink_eps=0.05, sink_iters=3):
    """
    Trains a 2D convolutional clustering autoencoder with reconstruction + clustering loss.

    Parameters
    ----------
    data : ndarray or tensor
        Input patches, shape (T, H, W, C)
    num_vars : int
        Number of original variables (channels)
    model : nn.Module
        Clustering autoencoder model
    alpha : float
        Weight for reconstruction loss
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Convert data to tensor
    tensor_data = torch.tensor(data, dtype=torch.float32)

    # Train/validation split
    num_samples = len(tensor_data)
    train_size = int(0.9 * num_samples)
    val_size = num_samples - train_size
    g = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
    tensor_data,
    [train_size, val_size],
    generator=g
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_data = torch.tensor(val_dataset[:], dtype=torch.float32).to(device)  # shape: (N_val, C, H, W)
    train_data = torch.tensor(train_dataset[:], dtype=torch.float32).to(device)  # shape: (N_train, C, H, W)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1 * lr )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_recon": [],
        "train_cluster": [],
        "val_loss": [],
        "val_recon": [],
        "val_cluster": []
    }

    for epoch in range(epochs):
        # ----- Training -----
        model.train()
        train_loss_total = 0.0
        train_recon_total = 0.0
        train_cluster_total = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
        
            # --- Check for NaNs in input batch ---
            if torch.isnan(batch).any():
                print("NaNs detected in batch input!")
                print("Batch indices:", torch.isnan(batch).nonzero(as_tuple=True))
                raise ValueError("NaN found in batch")
        
            # Forward
            x_hat, z, f = model(batch)

            # --- Check for NaNs in decoder output ---
            if torch.isnan(x_hat).any():
                print("NaNs detected in model output (x_hat)!")
                print("Output indices:", torch.isnan(x_hat).nonzero(as_tuple=True))
                raise ValueError("NaN found in reconstruction output")

            if torch.isnan(f).any():
                print("NaNs in clustering head features f")
                raise ValueError("NaN found in mlp(z)")
        
            # Reconstruction loss
            recon_loss = F.mse_loss(x_hat, batch)
            s = torch.matmul(f, model.clusters.T)
            Q = sinkhorn_fn(s, epsilon=sink_eps, sinkhorn_iterations=sink_iters).detach()

            log_probs = F.log_softmax(s / tau, dim=1)

            cluster_loss = -torch.mean(torch.sum(Q * log_probs, dim=1))

            # Total loss
            loss = alpha * recon_loss + (1 - alpha) * cluster_loss 
            loss.backward()
            optimizer.step()

            # accumulate
            train_loss_total += loss.item() * batch.size(0)
            train_recon_total += recon_loss.item() * batch.size(0)
            train_cluster_total += cluster_loss.item() * batch.size(0)

        train_loss_total /= len(train_loader.dataset)
        train_recon_total /= len(train_loader.dataset)
        train_cluster_total /= len(train_loader.dataset)

        with torch.no_grad():
            x_hat, z, f = model(train_data)

            # Reconstruction loss
            v_recon_loss = F.mse_loss(x_hat, train_data)

            # Clustering loss
            s = torch.matmul(f, model.clusters.T)
            Q = sinkhorn_fn(s)

            log_probs = F.log_softmax(s / tau, dim=1)
            v_cluster_loss = -torch.mean(torch.sum(Q * log_probs, dim=1))

            # Total loss
            val_loss_total = alpha * v_recon_loss + (1 - alpha) * v_cluster_loss 

        # Convert to scalar floats
        train_loss_total = val_loss_total.item()
        train_recon_total = v_recon_loss.item()
        train_cluster_total = v_cluster_loss.item()

        with torch.no_grad():
            x_hat, z, f = model(val_data)

            # Reconstruction loss
            v_recon_loss = F.mse_loss(x_hat, val_data)
        
            # Clustering loss
            s = torch.matmul(f, model.clusters.T)
            Q = sinkhorn_fn(s)
        
            log_probs = F.log_softmax(s / tau, dim=1)
            v_cluster_loss = -torch.mean(torch.sum(Q * log_probs, dim=1))
        
            # Total loss
            val_loss_total = alpha * v_recon_loss + (1 - alpha) * v_cluster_loss 
        
        # Convert to scalar floats
        val_loss_total = val_loss_total.item()
        val_recon_total = v_recon_loss.item()
        val_cluster_total = v_cluster_loss.item()

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss_total:.4f} (Recon: {train_recon_total:.4f}, Cluster: {train_cluster_total:.4f}) | "
              f"Val Loss: {val_loss_total:.4f} (Recon: {val_recon_total:.4f}, Cluster: {val_cluster_total:.4f})", flush=True)

        # Store results in history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss_total)
        history["train_recon"].append(train_recon_total)
        history["train_cluster"].append(train_cluster_total)
        history["val_loss"].append(val_loss_total)
        history["val_recon"].append(val_recon_total)
        history["val_cluster"].append(val_cluster_total)
        
    df_history = pd.DataFrame(history)

    return model, df_history


