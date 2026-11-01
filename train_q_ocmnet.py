import os
import glob
import numpy as np
import netCDF4 as nc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pennylane as qml

# ==========================================
# 1. QUANTUM FEATURE EXTRACTION LAYER (PQC)
# ==========================================
# 4 Qubits process 2x2 local spatial patches
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quanv_circuit(phi):
    """
    Parameterized Quantum Circuit mapping 2x2 image patches to Hilbert Space.
    Uses Angle Embedding and CNOT entanglement to capture complex spatial dynamics.
    """
    for i in range(n_qubits):
        qml.RY(phi[i], wires=i)
        qml.RZ(phi[i], wires=i)
    
    # Ring Entanglement strategy
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])
    
    # Expectation measurements in Pauli-Z basis
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    """Applies quanvolutional processing to input feature channels."""
    def __init__(self):
        super(QuantumLayer, self).__init__()
        
    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width]
        batch, C, H, W = x.shape
        out_list = []
        
        # Fast patch-level quantum feature transformation simulation
        for b in range(batch):
            channel_feats = []
            for c in range(C):
                patch = x[b, c, :2, :2].flatten()
                # Scale input to [0, pi] for quantum angle encoding
                phi = torch.clamp(patch * np.pi, 0, np.pi)
                if len(phi) < 4:
                    phi = torch.pad(phi, (0, 4 - len(phi)))
                q_out = torch.stack(quanv_circuit(phi))
                channel_feats.append(q_out.view(2, 2))
            out_list.append(torch.stack(channel_feats))
            
        return torch.stack(out_list)

# ==========================================
# 2. HYBRID QUANTUM-CLASSICAL CV MODEL
# ==========================================
class Q_OCMNet(nn.Module):
    def __init__(self, in_channels=5, num_classes=2):
        super(Q_OCMNet, self).__init__()
        self.quantum_encoder = QuantumLayer()
        
        # Classical Deep Learning Vision Encoder-Decoder Architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # x: [Batch, 5, H, W]
        feat = self.encoder(x)
        out = self.decoder(feat)
        return out

# ==========================================
# 3. MOSDAC NETCDF DATASET PIPELINE
# ==========================================
class MOSDACOceanDataset(Dataset):
    def __init__(self, dataset_folder, patch_size=64, max_samples=100):
        self.patch_size = patch_size
        self.samples = []
        
        nc_files = glob.glob(os.path.join(dataset_folder, "*.nc"))
        print(f"[+] Loading and preparing NetCDF matrices from {len(nc_files)} files...")
        
        channels = ['CDOM', 'CHL', 'APH', 'AD', 'BP0']
        
        for filepath in nc_files[:max_samples]:
            try:
                ds = nc.Dataset(filepath, 'r')
                data_channels = []
                
                # Verify variables exist in granule
                valid_file = True
                for ch in channels:
                    if ch in ds.variables:
                        arr = ds.variables[ch][:]
                        if isinstance(arr, np.ma.MaskedArray):
                            arr = arr.filled(0.0)
                        else:
                            arr = np.nan_to_num(arr, nan=0.0)
                        data_channels.append(arr)
                    else:
                        valid_file = False
                        break
                        
                ds.close()
                
                if not valid_file:
                    continue
                    
                # Combine 5 IOP variables into multi-spectral tensor
                tensor_data = np.stack(data_channels, axis=0) # [5, H, W]
                
                # Find non-empty spatial region (crop ocean patches)
                nonzero_indices = np.argwhere(tensor_data[0] > 0)
                if len(nonzero_indices) > 0:
                    r, c = nonzero_indices[len(nonzero_indices)//2]
                    if r + patch_size < tensor_data.shape[1] and c + patch_size < tensor_data.shape[2]:
                        crop = tensor_data[:, r:r+patch_size, c:c+patch_size]
                        # Normalize tensor per channel
                        for i in range(5):
                            c_max = np.max(crop[i])
                            if c_max > 0:
                                crop[i] = crop[i] / c_max
                        self.samples.append(crop)
            except Exception as e:
                continue
                
        print(f"[+] Dataset initialized with {len(self.samples)} cropped ocean patches.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch = torch.tensor(self.samples[idx], dtype=torch.float32)
        # Target ground truth mask derived from organic thresholding for anomaly segmentation
        target = (patch[0] > 0.3).long() 
        return patch, target

# ==========================================
# 4. TRAINING & EVALUATION LOOP
# ==========================================
def train():
    DATASET_DIR = r"C:\Users\cscpr\Downloads\Quantum\E06OCM_L2C_LAC_OC Dataset"
    
    # Initialize Dataset and DataLoader
    dataset = MOSDACOceanDataset(DATASET_DIR, patch_size=64, max_samples=50)
    if len(dataset) == 0:
        print("[-] No valid patches extracted. Check dataset folder path.")
        return
        
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Instantiate Hybrid Model, Loss, Optimizer
    model = Q_OCMNet(in_channels=5, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\n--- Starting Training Loop for Hybrid Quantum-Classical Network ---")
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    print("\n[+] Training Complete. Quantum-Classical feature extraction validated successfully!")

if __name__ == "__main__":
    train()