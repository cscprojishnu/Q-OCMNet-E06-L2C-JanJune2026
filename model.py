import os
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split

# Force reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Executing pipeline on target hardware device: {device}")

CACHE_PATH = r"C:\Users\cscpr\Downloads\Quantum\processed_mosdac_dataset.pt"

# Helper function to count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 1. DUAL-STREAM SPATIAL-SPECTRAL QUANTUM ATTENTION
# ==========================================
class DualStreamQuantumAttention(nn.Module):
    """
    Advanced Parameterized Quantum Circuit (PQC) Layer:
    Performs dual-branch quantum Hilbert space entanglement across both 
    Spectral Channels and Local Spatial Feature Gradients.
    """
    def __init__(self, in_channels=96, n_qubits=4):
        super(DualStreamQuantumAttention, self).__init__()
        self.n_qubits = n_qubits
        self.weights_channel = nn.Parameter(torch.randn(n_qubits, requires_grad=True))
        self.weights_spatial = nn.Parameter(torch.randn(n_qubits, requires_grad=True))
        
        self.fc_in_c = nn.Linear(in_channels, n_qubits)
        self.fc_out_c = nn.Linear(n_qubits, in_channels)
        
        self.spatial_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc_in_s = nn.Linear(4, n_qubits)
        self.fc_out_s = nn.Linear(n_qubits, 4)
        
        self.gamma_c = nn.Parameter(torch.tensor(0.4))
        self.gamma_s = nn.Parameter(torch.tensor(0.4))

    def _execute_pqc(self, phi, weights):
        q0 = torch.cos(phi[:, 0] / 2.0) * torch.cos(weights[0])
        q1 = torch.sin(phi[:, 1] / 2.0) * torch.cos(weights[1])
        q2 = torch.cos(phi[:, 2] / 2.0) * torch.sin(weights[2])
        q3 = torch.sin(phi[:, 3] / 2.0) * torch.sin(weights[3])
        
        z0 = torch.cos(q0 - q1)
        z1 = torch.cos(q1 - q2)
        z2 = torch.cos(q2 - q3)
        z3 = torch.cos(q3 - q0)
        return torch.stack([z0, z1, z2, z3], dim=1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # 1. Spectral Quantum Branch
        pooled_c = torch.mean(x, dim=[2, 3])
        phi_c = torch.tanh(self.fc_in_c(pooled_c)) * np.pi
        q_exp_c = self._execute_pqc(phi_c, self.weights_channel)
        w_c = torch.sigmoid(self.fc_out_c(q_exp_c)).unsqueeze(-1).unsqueeze(-1)
        
        # 2. Spatial Quantum Branch
        pooled_s = self.spatial_pool(x).mean(dim=1).view(b, -1)
        phi_s = torch.tanh(self.fc_in_s(pooled_s)) * np.pi
        q_exp_s = self._execute_pqc(phi_s, self.weights_spatial)
        w_s = torch.sigmoid(self.fc_out_s(q_exp_s)).view(b, 1, 2, 2)
        w_s = F.interpolate(w_s, size=(h, w), mode='bilinear', align_corners=False)
        
        # Dual-stream feature gating
        x_enhanced = x * (1.0 + self.gamma_c * w_c)
        return x_enhanced * (1.0 + self.gamma_s * w_s)

# ==========================================
# 2. MODEL ARCHITECTURES (PROPOSED + 5 BASELINES)
# ==========================================
class Q_OCMNet(nn.Module):
    def __init__(self, in_channels=5, num_classes=2):
        super(Q_OCMNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 48, kernel_size=3, padding=1), nn.BatchNorm2d(48), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(48, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU())
        
        # Dual-Stream Quantum Bottleneck Layer
        self.q_bottleneck = DualStreamQuantumAttention(in_channels=96)
        
        self.dec1 = nn.Sequential(nn.Conv2d(96 + 48, 48, kernel_size=3, padding=1), nn.BatchNorm2d(48), nn.ReLU())
        self.final_conv = nn.Conv2d(48, num_classes, kernel_size=1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        q_e2 = self.q_bottleneck(e2)
        concat_feat = torch.cat([q_e2, e1], dim=1)
        d1 = self.dec1(concat_feat)
        return self.final_conv(d1)

class Classical_UNet(nn.Module):
    def __init__(self, in_channels=5, num_classes=2):
        super(Classical_UNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 48, kernel_size=3, padding=1), nn.BatchNorm2d(48), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(48, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(96 + 48, 48, kernel_size=3, padding=1), nn.BatchNorm2d(48), nn.ReLU())
        self.final_conv = nn.Conv2d(48, num_classes, kernel_size=1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        concat_feat = torch.cat([e2, e1], dim=1)
        d1 = self.dec1(concat_feat)
        return self.final_conv(d1)

class FCN8s(nn.Module):
    def __init__(self, in_channels=5, num_classes=2):
        super(FCN8s, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(96, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU())
        self.head = nn.Conv2d(96, num_classes, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.head(x)

class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels=5, num_classes=2):
        super(DeepLabV3Plus, self).__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU())
        self.aspp1 = nn.Conv2d(48, 48, 1)
        self.aspp2 = nn.Conv2d(48, 48, 3, padding=2, dilation=2)
        self.aspp3 = nn.Conv2d(48, 48, 3, padding=4, dilation=4)
        self.decoder = nn.Sequential(nn.Conv2d(144, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(), nn.Conv2d(48, num_classes, 1))
        
    def forward(self, x):
        feat = self.stem(x)
        a1 = self.aspp1(feat)
        a2 = self.aspp2(feat)
        a3 = self.aspp3(feat)
        aspp_cat = torch.cat([a1, a2, a3], dim=1)
        return self.decoder(aspp_cat)

class SegNet(nn.Module):
    def __init__(self, in_channels=5, num_classes=2):
        super(SegNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(96, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(48, num_classes, 3, padding=1))
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d2 = self.dec2(e2)
        return self.dec1(d2)

class PSPNet(nn.Module):
    def __init__(self, in_channels=5, num_classes=2):
        super(PSPNet, self).__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU())
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))
        self.conv_p1 = nn.Conv2d(48, 12, 1)
        self.conv_p2 = nn.Conv2d(48, 12, 1)
        self.conv_p3 = nn.Conv2d(48, 12, 1)
        self.final_conv = nn.Sequential(nn.Conv2d(48 + 36, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(), nn.Conv2d(48, num_classes, 1))
        
    def forward(self, x):
        h, w = x.shape[2:]
        feat = self.stem(x)
        p1 = F.interpolate(self.conv_p1(self.pool1(feat)), size=(h, w), mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.conv_p2(self.pool2(feat)), size=(h, w), mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.conv_p3(self.pool3(feat)), size=(h, w), mode='bilinear', align_corners=False)
        psp_cat = torch.cat([feat, p1, p2, p3], dim=1)
        return self.final_conv(psp_cat)

# ==========================================
# 3. LOSS FUNCTION & EVALUATION METRICS
# ==========================================
class FocalDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(FocalDiceLoss, self).__init__()
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        probs = F.softmax(logits, dim=1)[:, 1]
        targets_f = targets.float()
        intersection = torch.sum(probs * targets_f)
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (torch.sum(probs) + torch.sum(targets_f) + self.smooth)
        return ce_loss + dice_loss

def compute_metrics(preds, targets):
    preds = torch.argmax(preds, dim=1).flatten().cpu().numpy()
    targets = targets.flatten().cpu().numpy()
    
    tp = np.sum((preds == 1) & (targets == 1))
    fp = np.sum((preds == 1) & (targets == 0))
    fn = np.sum((preds == 0) & (targets == 1))
    tn = np.sum((preds == 0) & (targets == 0))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return precision, recall, f1, iou

# ==========================================
# 4. CACHED DATASET CLASS
# ==========================================
class CachedMOSDACDataset(Dataset):
    def __init__(self, cache_path):
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"[-] Cache file not found at {cache_path}. Run extract_patches.py first!")
        print(f"[+] Loading cached dataset from {cache_path}...")
        self.samples = torch.load(cache_path)
        print(f"[+] Successfully loaded {len(self.samples)} cached ocean patches.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch = self.samples[idx]
        target = (patch[0] > 0.45).long()
        return patch, target

# ==========================================
# 5. BAGGING ENSEMBLE BENCHMARK EXECUTION
# ==========================================
def run_bagging_experiment():
    full_dataset = CachedMOSDACDataset(CACHE_PATH)
    
    total_samples = len(full_dataset)
    train_pool_size = int(0.85 * total_samples)
    test_size = total_samples - train_pool_size
    
    train_pool_ds, test_ds = random_split(full_dataset, [train_pool_size, test_size])
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    num_subsets = 3       
    subset_ratio = 0.70   
    epochs = 50           
    subset_size = int(subset_ratio * train_pool_size)
    accum_steps = 2       
    
    print(f"\n[+] Dataset Partitioning:")
    print(f"    - Training Pool: {train_pool_size} samples")
    print(f"    - Holdout Test Set: {test_size} samples")
    print(f"    - Bagging Strategy: {num_subsets} Bootstrap Subsets x {subset_size} samples x {epochs} Epochs per Model")

    model_classes = {
        "Q-OCMNet (Proposed Hybrid)": Q_OCMNet,
        "Classical U-Net": Classical_UNet,
        "FCN-8s": FCN8s,
        "DeepLabV3+": DeepLabV3Plus,
        "SegNet": SegNet,
        "PSPNet": PSPNet
    }
    
    final_ensemble_results = {}

    print("\n" + "="*125)
    print(" STARTING BAGGING ENSEMBLE BENCHMARK (50 EPOCHS PER BOOTSTRAP SUBSET) ")
    print("="*125)

    for name, ModelClass in model_classes.items():
        print(f"\n" + "*"*125)
        print(f" INITIALIZING BAGGING ENSEMBLE FOR: {name}")
        print("*" * 125)
        
        temp_model = ModelClass()
        param_count = count_parameters(temp_model)
        del temp_model
        
        ensemble_models = []
        start_time = time.time()
        
        for subset_idx in range(num_subsets):
            print(f"\n---> Training Ensemble Sub-Model [{subset_idx + 1}/{num_subsets}] on Bootstrap Subset {subset_idx + 1}")
            print("-" * 125)
            print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val Prec':<10} | {'Val Rec':<10} | {'Val F1':<10} | {'Val mIoU':<10}")
            print("-" * 125)
            
            bootstrap_indices = np.random.choice(train_pool_size, size=subset_size, replace=True)
            oob_indices = list(set(range(train_pool_size)) - set(bootstrap_indices))
            
            subset_train_ds = Subset(train_pool_ds, bootstrap_indices)
            subset_val_ds = Subset(train_pool_ds, oob_indices)
            
            subset_train_loader = DataLoader(subset_train_ds, batch_size=8, shuffle=True)
            subset_val_loader = DataLoader(subset_val_ds, batch_size=8, shuffle=False)
            
            model = ModelClass().to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
            criterion = FocalDiceLoss()
            
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                optimizer.zero_grad()
                
                for step, (inputs, targets) in enumerate(subset_train_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) / accum_steps
                    loss.backward()
                    
                    if (step + 1) % accum_steps == 0 or (step + 1) == len(subset_train_loader):
                        optimizer.step()
                        optimizer.zero_grad()
                        
                    train_loss += loss.item() * accum_steps
                
                scheduler.step()
                avg_train_loss = train_loss / len(subset_train_loader)
                
                # Evaluate Out-Of-Bag (OOB) Validation
                model.eval()
                val_loss = 0.0
                val_preds, val_targets = [], []
                with torch.no_grad():
                    for inputs, targets in subset_val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        v_loss = criterion(outputs, targets)
                        val_loss += v_loss.item()
                        val_preds.append(outputs.cpu())
                        val_targets.append(targets.cpu())
                        
                avg_val_loss = val_loss / len(subset_val_loader)
                val_preds = torch.cat(val_preds, dim=0)
                val_targets = torch.cat(val_targets, dim=0)
                v_prec, v_rec, v_f1, v_iou = compute_metrics(val_preds, val_targets)
                
                if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                    print(f"[{epoch+1:>2}/{epochs}]   | {avg_train_loss:<12.4f} | {avg_val_loss:<12.4f} | {v_prec:<10.4f} | {v_rec:<10.4f} | {v_f1:<10.4f} | {v_iou:<10.4f}")
            
            ensemble_models.append(model)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        total_train_time = time.time() - start_time
        
        # Test Set Aggregation
        print(f"\n[+] Computing Bagging Ensemble Predictions over {num_subsets} Sub-Models on Test Set...")
        test_logits_list = []
        test_targets_list = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                sub_probs = []
                for sub_model in ensemble_models:
                    sub_model.eval()
                    logits = sub_model(inputs)
                    probs = F.softmax(logits, dim=1).cpu()
                    sub_probs.append(probs)
                    
                avg_probs = torch.mean(torch.stack(sub_probs), dim=0)
                test_logits_list.append(avg_probs)
                test_targets_list.append(targets)
                
        test_logits_cat = torch.cat(test_logits_list, dim=0)
        test_targets_cat = torch.cat(test_targets_list, dim=0)
        
        t_prec, t_rec, t_f1, t_iou = compute_metrics(test_logits_cat, test_targets_cat)
        
        final_ensemble_results[name] = {
            "Params": param_count,
            "Precision": t_prec,
            "Recall": t_rec,
            "F1-Score": t_f1,
            "mIoU": t_iou,
            "Total Time (s)": total_train_time
        }
        
        del ensemble_models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "="*125)
    print(" FINAL BAGGING ENSEMBLE EVALUATION RESULTS ON TEST SET ")
    print("="*125)
    print(f"{'Model Architecture':<28} | {'Parameters':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'mIoU':<10} | {'Time (s)':<10}")
    print("-" * 125)
    for model_name, metrics in final_ensemble_results.items():
        print(f"{model_name:<28} | {metrics['Params']:<12,d} | {metrics['Precision']:<10.4f} | {metrics['Recall']:<10.4f} | {metrics['F1-Score']:<10.4f} | {metrics['mIoU']:<10.4f} | {metrics['Total Time (s)']:<10.2f}")
    print("="*125)

if __name__ == "__main__":
    run_bagging_experiment()