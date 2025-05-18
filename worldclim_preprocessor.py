"""
WorldClim Data Preprocessor
Este módulo deve ser executado localmente para baixar e processar os dados WorldClim.
Os dados recortados para o Brasil serão salvos para uso no aplicativo web.
"""

import os
import zipfile
import requests
import rasterio
from rasterio.mask import mask
import numpy as np
from pathlib import Path
import geopandas as gpd
import geobr
from tqdm import tqdm

class WorldClimPreprocessor:
    def __init__(self, output_dir='data/worldclim_brazil'):
        self.output_dir = Path(output_dir)
        self.temp_dir = Path('temp_worldclim')
        self.worldclim_url = 'https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_bio.zip'
        self.brazil_gdf = None
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def download_worldclim(self):
        """Download WorldClim bioclimatic data"""
        print("Baixando dados WorldClim (pode demorar)...")
        
        zip_path = self.temp_dir / 'worldclim_bio.zip'
        
        if not zip_path.exists():
            response = requests.get(self.worldclim_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc='Download') as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        else:
            print("Arquivo já baixado, pulando download...")
        
        # Extract files
        print("Extraindo arquivos...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(self.temp_dir)
        
        print("Download e extração concluídos!")
        
    def get_brazil_boundary(self):
        """Get Brazil boundary from geobr"""
        print("Carregando limites do Brasil...")
        
        try:
            # Get Brazil boundary
            self.brazil_gdf = geobr.read_country(year=2020)
            # Ensure it's in the correct CRS
            self.brazil_gdf = self.brazil_gdf.to_crs('EPSG:4326')
            print("Limites do Brasil carregados com sucesso!")
        except Exception as e:
            print(f"Erro ao carregar limites do Brasil: {e}")
            raise
    
    def crop_to_brazil(self, input_path, output_path):
        """Crop a raster file to Brazil boundaries"""
        with rasterio.open(input_path) as src:
            # Get Brazil geometry
            brazil_geom = self.brazil_gdf.geometry.values[0]
            
            # Crop raster
            out_image, out_transform = mask(src, [brazil_geom], crop=True, nodata=-9999)
            
            # Update metadata
            out_meta = src.meta
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw",
                "nodata": -9999
            })
            
            # Write cropped raster
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
    
    def process_all_layers(self):
        """Process all bioclimatic layers"""
        print("Processando layers bioclimáticos...")
        
        # Get all bio tif files
        bio_files = list(self.temp_dir.glob('wc2.1_2.5m_bio_*.tif'))
        
        if not bio_files:
            print("Erro: Nenhum arquivo bio encontrado!")
            return
        
        # Process each file
        for bio_file in tqdm(bio_files, desc='Recortando layers'):
            # Extract bio number from filename
            bio_num = bio_file.stem.split('_')[-1]
            output_file = self.output_dir / f'bio{bio_num}_brazil.tif'
            
            try:
                self.crop_to_brazil(bio_file, output_file)
            except Exception as e:
                print(f"Erro ao processar {bio_file}: {e}")
        
        print("Processamento concluído!")
    
    def create_metadata_file(self):
        """Create metadata file with layer information"""
        from datetime import datetime
        
        metadata = {
            'layers': {
                'bio1': {'name': 'Annual Mean Temperature', 'unit': '°C * 10'},
                'bio2': {'name': 'Mean Diurnal Range', 'unit': '°C * 10'},
                'bio3': {'name': 'Isothermality', 'unit': '%'},
                'bio4': {'name': 'Temperature Seasonality', 'unit': 'standard deviation * 100'},
                'bio5': {'name': 'Max Temperature of Warmest Month', 'unit': '°C * 10'},
                'bio6': {'name': 'Min Temperature of Coldest Month', 'unit': '°C * 10'},
                'bio7': {'name': 'Temperature Annual Range', 'unit': '°C * 10'},
                'bio8': {'name': 'Mean Temperature of Wettest Quarter', 'unit': '°C * 10'},
                'bio9': {'name': 'Mean Temperature of Driest Quarter', 'unit': '°C * 10'},
                'bio10': {'name': 'Mean Temperature of Warmest Quarter', 'unit': '°C * 10'},
                'bio11': {'name': 'Mean Temperature of Coldest Quarter', 'unit': '°C * 10'},
                'bio12': {'name': 'Annual Precipitation', 'unit': 'mm'},
                'bio13': {'name': 'Precipitation of Wettest Month', 'unit': 'mm'},
                'bio14': {'name': 'Precipitation of Driest Month', 'unit': 'mm'},
                'bio15': {'name': 'Precipitation Seasonality', 'unit': 'coefficient of variation'},
                'bio16': {'name': 'Precipitation of Wettest Quarter', 'unit': 'mm'},
                'bio17': {'name': 'Precipitation of Driest Quarter', 'unit': 'mm'},
                'bio18': {'name': 'Precipitation of Warmest Quarter', 'unit': 'mm'},
                'bio19': {'name': 'Precipitation of Coldest Quarter', 'unit': 'mm'}
            },
            'resolution': '2.5 minutes',
            'extent': 'Brazil',
            'source': 'WorldClim v2.1',
            'preprocessed_date': datetime.now().isoformat()
        }
        
        import json
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def cleanup(self):
        """Remove temporary files"""
        print("Limpando arquivos temporários...")
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        print("Limpeza concluída!")
    
    def run_full_pipeline(self):
        """Run the complete preprocessing pipeline"""
        print("=== WorldClim Preprocessor para Brasil ===")
        
        try:
            # 1. Download WorldClim data
            self.download_worldclim()
            
            # 2. Get Brazil boundary
            self.get_brazil_boundary()
            
            # 3. Process all layers
            self.process_all_layers()
            
            # 4. Create metadata
            self.create_metadata_file()
            
            # 5. Cleanup
            self.cleanup()
            
            print("\n✅ Preprocessamento concluído com sucesso!")
            print(f"Dados salvos em: {self.output_dir}")
            
        except Exception as e:
            print(f"\n❌ Erro durante o processamento: {e}")
            raise

if __name__ == '__main__':
    # Run preprocessing
    preprocessor = WorldClimPreprocessor()
    preprocessor.run_full_pipeline()