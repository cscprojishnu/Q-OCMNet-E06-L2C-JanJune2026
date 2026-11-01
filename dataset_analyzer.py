import os
import glob
from pathlib import Path
import numpy as np
from PIL import Image

# Import netCDF4 or h5py for reading .nc scientific files
try:
    import netCDF4 as nc
    HAS_NETCDF = True
except ImportError:
    try:
        import h5py
        HAS_H5PY = True
        HAS_NETCDF = False
    except ImportError:
        HAS_NETCDF = False
        HAS_H5PY = False

def print_separator(title=""):
    print("\n" + "=" * 80)
    if title:
        print(f" {title.upper()} ".center(80, "="))
        print("=" * 80)

def analyze_netcdf_file(filepath):
    """Inspects a single NetCDF file and prints its structure, dimensions, and statistical summary."""
    print(f"\n--- Analyzing NetCDF Granule: {os.path.basename(filepath)} ---")
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"File Size: {file_size_mb:.2f} MB")
    
    if HAS_NETCDF:
        try:
            ds = nc.Dataset(filepath, 'r')
            print(f"Global Attributes ({len(ds.ncattrs())}):")
            for attr in list(ds.ncattrs())[:5]:  # Print top 5 global attributes
                print(f"  - {attr}: {getattr(ds, attr)}")
            
            print(f"\nDimensions ({len(ds.dimensions)}):")
            for dim_name, dim in ds.dimensions.items():
                print(f"  - {dim_name}: size {len(dim)}")
                
            print(f"\nVariables / Feature Channels ({len(ds.variables)}):")
            for var_name, var in ds.variables.items():
                shape = var.shape
                dtype = var.dtype
                print(f"\n  > Variable: '{var_name}' | Shape: {shape} | Type: {dtype}")
                
                # Perform numerical statistics on data arrays
                try:
                    data = var[:]
                    if isinstance(data, np.ma.MaskedArray):
                        valid_data = data.compressed()
                        fill_pct = ((data.size - valid_data.size) / data.size) * 100
                    else:
                        valid_mask = ~np.isnan(data)
                        valid_data = data[valid_mask]
                        fill_pct = ((data.size - valid_data.size) / data.size) * 100
                        
                    print(f"    - Valid Pixels: {100 - fill_pct:.2f}% | Missing/Cloud/Fill Pixels: {fill_pct:.2f}%")
                    if len(valid_data) > 0:
                        print(f"    - Min: {np.min(valid_data):.6f} | Max: {np.max(valid_data):.6f} | Mean: {np.mean(valid_data):.6f} | Std: {np.std(valid_data):.6f}")
                except Exception as e:
                    print(f"    - Could not compute numerical stats: {e}")
            ds.close()
        except Exception as e:
            print(f"Error reading NetCDF with netCDF4: {e}")
            
    elif HAS_H5PY:
        try:
            with h5py.File(filepath, 'r') as f:
                print("Keys in HDF5/NetCDF file:")
                f.visititems(lambda name, obj: print(f"  - {name}") if isinstance(obj, h5py.Dataset) else None)
        except Exception as e:
            print(f"Error reading with h5py: {e}")
    else:
        print("[-] Neither 'netCDF4' nor 'h5py' installed. Install via: pip install netCDF4 h5py")

def analyze_browse_images(image_folder):
    """Analyzes the browse JPEG/PNG images for dimensions, color profiles, and validity."""
    print_separator("BROWSE IMAGES ANALYSIS")
    image_files = glob.glob(os.path.join(image_folder, "*.[jJ][pP][gG]")) + \
                  glob.glob(os.path.join(image_folder, "*.[pP][nN][gG]"))
                  
    print(f"Total Browse Images Found: {len(image_files)}")
    if not image_files:
        print("No browse images found in directory.")
        return

    resolutions = []
    modes = []
    file_sizes = []

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                resolutions.append(img.size)
                modes.append(img.mode)
                file_sizes.append(os.path.getsize(img_path) / 1024)  # KB
        except Exception as e:
            print(f"Corrupt or unreadable image: {os.path.basename(img_path)} - {e}")

    unique_resolutions = set(resolutions)
    unique_modes = set(modes)
    avg_size = np.mean(file_sizes) if file_sizes else 0

    print(f"\nImage Resolutions Found (Width x Height): {unique_resolutions}")
    print(f"Color Modes: {unique_modes}")
    print(f"Average Browse Image File Size: {avg_size:.2f} KB")
    
    # Inspect first image in detail
    sample_img_path = image_files[0]
    print(f"\nSample Image Detail ({os.path.basename(sample_img_path)}):")
    with Image.open(sample_img_path) as img:
        img_arr = np.array(img)
        print(f"  - Array Shape: {img_arr.shape}")
        print(f"  - Min Pixel Value: {np.min(img_arr)}, Max Pixel Value: {np.max(img_arr)}")

def run_dataset_summary(root_directory):
    """Recursively scans the root directory and generates a text summary report."""
    print_separator("DATASET DIRECTORY & STRUCTURE OVERVIEW")
    print(f"Root Folder: {os.path.abspath(root_directory)}")
    
    all_files = []
    nc_files = []
    img_files = []
    
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)
            if file.endswith('.nc'):
                nc_files.append(full_path)
            elif file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_files.append(full_path)

    total_size_gb = sum(os.path.getsize(f) for f in all_files) / (1024 ** 3)

    print(f"\nTotal Files Count: {len(all_files)}")
    print(f"Total Dataset Size: {total_size_gb:.3f} GB")
    print(f"NetCDF (.nc) Science Files Count: {len(nc_files)}")
    print(f"Browse Image Files Count: {len(img_files)}")

    # 1. Analyze NetCDF Scientific Data
    print_separator("NETCDF (.NC) MATRIX ANALYSIS")
    if nc_files:
        print(f"Analyzing representative sample file (1 of {len(nc_files)})...")
        analyze_netcdf_file(nc_files[0])
    else:
        print("No .nc files found.")

    # 2. Analyze Browse Images
    img_dir = os.path.join(root_directory, "images")
    if os.path.exists(img_dir):
        analyze_browse_images(img_dir)
    elif img_files:
        analyze_browse_images(root_directory)

if __name__ == "__main__":
    # Point this path to your extracted dataset directory
    DATASET_DIR = r"./E06OCM_L2C_LAC_OC Dataset" 
    
    if os.path.exists(DATASET_DIR):
        run_dataset_summary(DATASET_DIR)
    else:
        print(f"Directory '{DATASET_DIR}' not found. Please verify the folder path in DATASET_DIR.")