import os
import glob
import numpy as np
import netCDF4 as nc

def analyze_dataset_granules(dataset_folder):
    nc_files = glob.glob(os.path.join(dataset_folder, "*.nc"))
    print(f"Total NetCDF granules detected: {len(nc_files)}\n")
    
    valid_ocean_granules = []
    
    for idx, filepath in enumerate(nc_files):
        filename = os.path.basename(filepath)
        try:
            ds = nc.Dataset(filepath, 'r')
            
            # Extract primary IOP variable (e.g., CDOM, CHL, or APH)
            var_name = None
            for key in ['CDOM', 'CHL', 'APH', 'AD', 'BP0']:
                if key in ds.variables:
                    var_name = key
                    break
            
            if var_name is None:
                ds.close()
                continue
                
            var_data = ds.variables[var_name]
            raw_array = var_data[:]
            
            # Check for fill values / masks
            if isinstance(raw_array, np.ma.MaskedArray):
                valid_mask = ~raw_array.mask
                valid_pixels = np.sum(valid_mask)
            else:
                fill_val = getattr(var_data, '_FillValue', -999.0)
                valid_mask = (raw_array != fill_val) & (~np.isnan(raw_array))
                valid_pixels = np.sum(valid_mask)
                
            total_pixels = raw_array.size
            valid_pct = (valid_pixels / total_pixels) * 100
            
            if valid_pct > 0.5:  # At least 0.5% ocean coverage in the granule
                lat_min, lat_max = np.min(ds.variables['latitude']), np.max(ds.variables['latitude'])
                lon_min, lon_max = np.min(ds.variables['longitude']), np.max(ds.variables['longitude'])
                
                valid_ocean_granules.append({
                    'filename': filename,
                    'path': filepath,
                    'valid_pct': valid_pct,
                    'variable': var_name,
                    'lat_range': (lat_min, lat_max),
                    'lon_range': (lon_min, lon_max)
                })
                print(f"[FOUND OCEAN SCENE] {filename} | {var_name} Valid: {valid_pct:.2f}% | Lat: {lat_min:.1f}-{lat_max:.1f}, Lon: {lon_min:.1f}-{lon_max:.1f}")
            
            ds.close()
            
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            
    print("\n" + "="*80)
    print(f"TOTAL OCEAN GRANULES WITH VALID DATA: {len(valid_ocean_granules)} / {len(nc_files)}")
    print("="*80)
    
    return valid_ocean_granules

if __name__ == "__main__":
    DATASET_DIR = r"C:\Users\cscpr\Downloads\Quantum\E06OCM_L2C_LAC_OC Dataset"
    ocean_granules = analyze_dataset_granules(DATASET_DIR)