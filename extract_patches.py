import os
import glob
import gc
import numpy as np
import netCDF4 as nc
import torch

DATASET_DIR = r"C:\Users\cscpr\Downloads\Quantum\E06OCM_L2C_LAC_OC Dataset"
OUTPUT_CACHE_PATH = r"C:\Users\cscpr\Downloads\Quantum\processed_mosdac_dataset.pt"

def extract_and_cache_dataset():
    nc_files = glob.glob(os.path.join(DATASET_DIR, "*.nc"))
    print(f"[+] Found {len(nc_files)} NetCDF granules. Extracting patches...")
    
    channels = ['CDOM', 'CHL', 'APH', 'AD', 'BP0']
    patch_size = 64
    samples = []
    
    for idx, filepath in enumerate(nc_files):
        if idx % 50 == 0:
            print(f"    Processing file {idx}/{len(nc_files)}...")
            gc.collect()  # Force C-level and Python heap cleanup
            
        try:
            ds = nc.Dataset(filepath, 'r')
            data_channels = []
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
            del ds
            
            if not valid_file:
                continue
                
            tensor_data = np.stack(data_channels, axis=0)
            nonzero = np.argwhere(tensor_data[0] > 0)
            
            if len(nonzero) > 100:
                for k in range(min(3, len(nonzero) // 1000 + 1)):
                    loc_idx = (k * len(nonzero)) // 3
                    r, c = nonzero[loc_idx]
                    if r + patch_size < tensor_data.shape[1] and c + patch_size < tensor_data.shape[2]:
                        crop = tensor_data[:, r:r+patch_size, c:c+patch_size]
                        for i in range(5):
                            c_max = np.max(crop[i])
                            if c_max > 0:
                                crop[i] = crop[i] / c_max
                        samples.append(torch.tensor(crop, dtype=torch.float32))
        except Exception:
            continue

    print(f"[+] Successfully extracted {len(samples)} high-quality ocean patches.")
    print(f"[+] Saving cached dataset to {OUTPUT_CACHE_PATH}...")
    torch.save(samples, OUTPUT_CACHE_PATH)
    print("[+] Cache saved successfully! You can now run model training without NetCDF errors.")

if __name__ == "__main__":
    extract_and_cache_dataset()