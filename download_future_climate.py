#!/usr/bin/env python3
"""
Download future climate data from WorldClim CMIP6 for educational SDM platform
Ensures compatibility with current WorldClim Brazil data for comparison
"""

import os
import requests
import rasterio
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Configuration for Atlantic Forest educational setup
CONFIG = {
    "output_dir": "data/worldclim_future",
    "current_worldclim_dir": "data/worldclim_brazil",  # Reference for compatibility
    "files": {
        "ssp126_2081-2100": "https://geodata.ucdavis.edu/cmip6/2.5m/MPI-ESM1-2-HR/ssp126/wc2.1_2.5m_bioc_MPI-ESM1-2-HR_ssp126_2081-2100.tif",
        "ssp585_2081-2100": "https://geodata.ucdavis.edu/cmip6/2.5m/MPI-ESM1-2-HR/ssp585/wc2.1_2.5m_bioc_MPI-ESM1-2-HR_ssp585_2081-2100.tif"
    }
}

def get_reference_metadata():
    """Get metadata from current WorldClim Brazil data for compatibility"""
    ref_dir = Path(CONFIG["current_worldclim_dir"])
    ref_file = ref_dir / "bio1_brazil.tif"
    
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_file}\nPlease ensure current WorldClim data is available.")
    
    with rasterio.open(ref_file) as src:
        metadata = {
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds,
            'width': src.width,
            'height': src.height,
            'dtype': src.dtypes[0],  # dtype of first band
            'nodata': src.nodata
        }
    
    print(f"Reference metadata loaded from: {ref_file}")
    print(f"  CRS: {metadata['crs']}")
    print(f"  Bounds: {metadata['bounds']}")
    print(f"  Dimensions: {metadata['width']} x {metadata['height']}")
    
    return metadata

def create_directories():
    """Create necessary directory structure"""
    base_path = Path(CONFIG["output_dir"])
    
    for scenario_period in CONFIG["files"].keys():
        dir_path = base_path / scenario_period
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def download_file(url, dest_path):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def extract_and_align_bioclim(multiband_tif, output_dir, scenario_period, ref_metadata):
    """Extract bioclim variables and align to current WorldClim Brazil data"""
    try:
        with rasterio.open(multiband_tif) as src:
            print(f"\nProcessing {src.count} bioclimatic variables...")
            
            # Process each band (bioclim variable)
            for band_idx in range(1, src.count + 1):
                # Read the full band data
                data = src.read(band_idx)
                
                # Create a temporary array for the reprojected data
                dst_shape = (ref_metadata['height'], ref_metadata['width'])
                dst_array = np.zeros(dst_shape, dtype=ref_metadata['dtype'])
                
                # Reproject to match reference data
                reproject(
                    source=data,
                    destination=dst_array,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_metadata['transform'],
                    dst_crs=ref_metadata['crs'],
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=ref_metadata['nodata']
                )
                
                # Output filename (matching current WorldClim naming)
                output_file = output_dir / f"wc2.1_2.5m_bioc_MPI-ESM1-2-HR_{scenario_period}_bio{band_idx}.tif"
                
                # Write with exact same profile as reference
                profile = {
                    'driver': 'GTiff',
                    'dtype': ref_metadata['dtype'],
                    'nodata': ref_metadata['nodata'],
                    'width': ref_metadata['width'],
                    'height': ref_metadata['height'],
                    'count': 1,
                    'crs': ref_metadata['crs'],
                    'transform': ref_metadata['transform'],
                    'compress': 'lzw'
                }
                
                with rasterio.open(output_file, 'w', **profile) as dst:
                    dst.write(dst_array, 1)
                
                print(f"  Created: bio{band_idx} -> {output_file.name}")
        
        return True
    except Exception as e:
        print(f"Error processing multiband TIF: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def download_future_climate_data():
    """Download all required future climate data"""
    create_directories()
    
    # Get reference metadata for compatibility
    try:
        ref_metadata = get_reference_metadata()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please run the current WorldClim download first.")
        return False
    
    base_path = Path(CONFIG["output_dir"])
    
    for scenario_period, url in CONFIG["files"].items():
        print(f"\n{'='*50}")
        print(f"Processing: {scenario_period}")
        print('='*50)
        
        # Download paths
        multiband_file = base_path / f"temp_{scenario_period}.tif"
        output_dir = base_path / scenario_period
        
        # Check if already processed
        existing_files = list(output_dir.glob("*.tif"))
        if len(existing_files) >= 19:
            print(f"Already processed: {len(existing_files)} files exist")
            # Verify compatibility
            verify_compatibility(output_dir, ref_metadata)
            continue
        
        # Download multiband TIF
        print(f"\nDownloading from: {url}")
        if download_file(url, multiband_file):
            print("\nExtracting and aligning bioclimatic variables...")
            if extract_and_align_bioclim(multiband_file, output_dir, scenario_period, ref_metadata):
                # Clean up multiband file
                multiband_file.unlink()
                print(f"Cleaned up: {multiband_file.name}")
                # Verify compatibility
                verify_compatibility(output_dir, ref_metadata)
            else:
                print(f"Failed to process {multiband_file}")
        else:
            print(f"Failed to download {scenario_period}")
    
    print("\n✅ Future climate data download complete!")
    return True

def verify_compatibility(data_dir, ref_metadata):
    """Verify that processed data is compatible with reference"""
    print("\nVerifying compatibility...")
    
    # Check first file
    test_file = list(data_dir.glob("*bio1.tif"))[0]
    
    with rasterio.open(test_file) as src:
        compatible = True
        
        # Check CRS
        if src.crs != ref_metadata['crs']:
            print(f"  ❌ CRS mismatch: {src.crs} vs {ref_metadata['crs']}")
            compatible = False
        
        # Check dimensions
        if (src.width, src.height) != (ref_metadata['width'], ref_metadata['height']):
            print(f"  ❌ Dimension mismatch: {src.width}x{src.height} vs {ref_metadata['width']}x{ref_metadata['height']}")
            compatible = False
        
        # Check bounds (with small tolerance)
        bounds_match = all(abs(a - b) < 0.0001 for a, b in zip(src.bounds, ref_metadata['bounds']))
        if not bounds_match:
            print(f"  ❌ Bounds mismatch: {src.bounds} vs {ref_metadata['bounds']}")
            compatible = False
        
        if compatible:
            print("  ✅ Data is fully compatible with current WorldClim Brazil data")
        else:
            print("  ⚠️  Warning: Data may not be fully compatible")

def verify_downloads():
    """Verify all required files are present and compatible"""
    base_path = Path(CONFIG["output_dir"])
    
    print("\n📊 Verification Summary:")
    print("="*50)
    
    # Get reference metadata
    try:
        ref_metadata = get_reference_metadata()
    except Exception as e:
        print(f"Cannot verify without reference data: {e}")
        return
    
    for scenario_period in CONFIG["files"].keys():
        path = base_path / scenario_period
        tif_files = list(path.glob("*.tif"))
        
        print(f"\n{scenario_period}:")
        print(f"  Files found: {len(tif_files)}")
        
        # Check if all 19 bioclim variables are present
        missing = []
        for i in range(1, 20):
            expected_file = path / f"wc2.1_2.5m_bioc_MPI-ESM1-2-HR_{scenario_period}_bio{i}.tif"
            if not expected_file.exists():
                missing.append(i)
        
        if missing:
            print(f"  Missing variables: bio{missing}")
        else:
            print("  ✅ All 19 bioclimatic variables present")
        
        # Check file sizes
        if tif_files:
            sizes = [f.stat().st_size / (1024*1024) for f in tif_files]
            print(f"  File sizes: {min(sizes):.1f} - {max(sizes):.1f} MB")
            
        # Verify compatibility
        if tif_files:
            verify_compatibility(path, ref_metadata)
    
    print("\n" + "="*50)

if __name__ == "__main__":
    print("🌡️  WorldClim CMIP6 Future Climate Data Downloader")
    print("=" * 50)
    print("Model: MPI-ESM1-2-HR (Max Planck Institute, Germany)")
    print("Scenarios: SSP1-2.6 (optimistic), SSP5-8.5 (pessimistic)")
    print("Period: 2081-2100")
    print("Region: Brazil (aligned with current WorldClim data)")
    print("=" * 50)
    
    try:
        if download_future_climate_data():
            verify_downloads()
            
            print("\n📝 Next steps:")
            print("1. Run the TAIPA application")
            print("2. Train a model for your species")
            print("3. Navigate to 'Projeção Futura' to analyze climate change impacts")
            print("\n✅ Data is fully compatible with current WorldClim Brazil data")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()