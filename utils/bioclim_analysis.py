"""
Bioclimatic Analysis Module
Funções para análise de variáveis bioclimáticas
"""

import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import json

class BioclimAnalyzer:
    def __init__(self, data_dir='data/worldclim_brazil'):
        self.data_dir = Path(data_dir)
        self.metadata = self._load_metadata()
        self.available_layers = self._get_available_layers()
    
    def _load_metadata(self):
        """Load metadata about bioclimatic layers"""
        metadata_path = self.data_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            # Default metadata if file doesn't exist
            return {
                'layers': {
                    f'bio{i}': {
                        'name': f'Bioclimatic Variable {i}',
                        'unit': 'units'
                    } for i in range(1, 20)
                }
            }
    
    def _get_available_layers(self):
        """Get list of available bioclimatic layers"""
        layers = {}
        for bio_file in self.data_dir.glob('bio*_brazil.tif'):
            # Extract bio number from filename like 'bio1_brazil.tif'
            bio_num = bio_file.stem.split('_')[0][3:]  # Remove 'bio' prefix
            layers[f'bio{bio_num}'] = bio_file
        return layers
    
    def extract_values_at_points(self, points: List[Tuple[float, float]], 
                                layers: List[str]) -> pd.DataFrame:
        """
        Extract bioclimatic values at specific points
        
        Args:
            points: List of (lat, lon) tuples
            layers: List of layer names (e.g., ['bio1', 'bio12'])
        
        Returns:
            DataFrame with extracted values
        """
        data = {'latitude': [], 'longitude': []}
        
        # Initialize columns for each layer
        for layer in layers:
            data[layer] = []
        
        # Extract values for each point
        for lat, lon in points:
            data['latitude'].append(lat)
            data['longitude'].append(lon)
            
            for layer in layers:
                if layer not in self.available_layers:
                    data[layer].append(np.nan)
                    continue
                
                # Read raster value at point
                with rasterio.open(self.available_layers[layer]) as src:
                    # Convert lat/lon to pixel coordinates
                    row, col = src.index(lon, lat)
                    
                    # Handle out of bounds
                    if row < 0 or col < 0 or row >= src.height or col >= src.width:
                        data[layer].append(np.nan)
                    else:
                        # Read value
                        value = src.read(1)[row, col]
                        # Handle nodata
                        if value == src.nodata:
                            data[layer].append(np.nan)
                        else:
                            # Convert temperature values (stored as °C * 10)
                            if 'bio' in layer and int(layer.replace('bio', '')) in [1,2,5,6,7,8,9,10,11]:
                                value = value / 10.0
                            data[layer].append(value)
        
        return pd.DataFrame(data)
    
    def calculate_correlation_matrix(self, df: pd.DataFrame, 
                                   method='pearson') -> pd.DataFrame:
        """Calculate correlation matrix for bioclimatic variables"""
        # Select only bioclimatic columns
        bio_cols = [col for col in df.columns if col.startswith('bio')]
        return df[bio_cols].corr(method=method)
    
    def calculate_vif(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Variance Inflation Factor for each variable"""
        from statsmodels.tools.tools import add_constant
        
        # Select only bioclimatic columns
        bio_cols = [col for col in df.columns if col.startswith('bio')]
        bio_data = df[bio_cols].dropna()
        
        # Add constant for proper VIF calculation
        X = add_constant(bio_data)
        
        # Calculate VIF
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(X.shape[1])]
        
        # Remove the constant from results
        vif_data = vif_data[vif_data["Variable"] != "const"]
        
        # Add metadata
        vif_data["Description"] = [self.metadata['layers'].get(var, {}).get('name', var) 
                                  for var in vif_data["Variable"]]
        
        return vif_data.sort_values('VIF', ascending=False)
    
    def select_variables(self, df: pd.DataFrame, 
                        vif_threshold: float = 5.0,
                        correlation_threshold: float = 0.7,
                        return_steps: bool = False) -> List[str]:
        """
        Automatically select variables using stepwise VIF elimination
        
        Args:
            df: DataFrame with bioclimatic data
            vif_threshold: Maximum acceptable VIF value
            correlation_threshold: Maximum acceptable correlation
            return_steps: If True, return all steps of the elimination process
        
        Returns:
            List of selected variable names (or tuple with steps if return_steps=True)
        """
        from statsmodels.tools.tools import add_constant
        
        # Select only bioclimatic columns
        bio_cols = [col for col in df.columns if col.startswith('bio')]
        bio_data = df[bio_cols].dropna()
        
        variables = list(bio_cols)
        steps = []
        iteration = 1
        
        # Stepwise elimination
        while len(variables) > 1:
            # Add constant and calculate VIF
            X_temp = add_constant(bio_data[variables])
            vif_df = pd.DataFrame()
            vif_df["Variable"] = X_temp.columns
            vif_df["VIF"] = [variance_inflation_factor(X_temp.values, i) 
                           for i in range(X_temp.shape[1])]
            
            # Exclude constant from removal consideration
            vif_df_no_const = vif_df[vif_df["Variable"] != "const"]
            
            step_info = {
                'iteration': iteration,
                'variables': variables.copy(),
                'vif_values': vif_df_no_const.copy()
            }
            
            # Check if all variables are below threshold
            max_vif = vif_df_no_const["VIF"].max()
            if max_vif < vif_threshold:
                step_info['action'] = 'completed'
                steps.append(step_info)
                break
            
            # Remove variable with highest VIF
            max_var = vif_df_no_const.sort_values("VIF", ascending=False)["Variable"].iloc[0]
            max_vif_value = vif_df_no_const.sort_values("VIF", ascending=False)["VIF"].iloc[0]
            
            step_info['removed_variable'] = max_var
            step_info['removed_vif'] = max_vif_value
            step_info['action'] = 'removed'
            steps.append(step_info)
            
            variables.remove(max_var)
            iteration += 1
        
        # Final check for correlations among selected variables
        if len(variables) > 1:
            corr_matrix = self.calculate_correlation_matrix(bio_data[variables])
            
            # Remove highly correlated variables
            to_remove = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                        # Remove the variable with higher VIF
                        var1, var2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        # Get final VIF values
                        X_final = add_constant(bio_data[variables])
                        vif_final = pd.DataFrame()
                        vif_final["Variable"] = X_final.columns
                        vif_final["VIF"] = [variance_inflation_factor(X_final.values, i) 
                                          for i in range(X_final.shape[1])]
                        vif_final = vif_final[vif_final["Variable"] != "const"]
                        
                        vif1 = vif_final[vif_final['Variable'] == var1]['VIF'].values[0]
                        vif2 = vif_final[vif_final['Variable'] == var2]['VIF'].values[0]
                        to_remove.add(var1 if vif1 > vif2 else var2)
            
            if to_remove:
                variables = [var for var in variables if var not in to_remove]
                steps.append({
                    'iteration': iteration,
                    'action': 'correlation_removal',
                    'removed_variables': list(to_remove),
                    'final_variables': variables
                })
        
        if return_steps:
            return variables, steps
        return variables
    
    def plot_correlation_matrix(self, corr_matrix: pd.DataFrame) -> plt.Figure:
        """Plot correlation matrix heatmap"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix), k=1)
        
        # Draw heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5, ax=ax)
        
        ax.set_title('Correlation Matrix - Bioclimatic Variables', fontsize=16, pad=20)
        plt.tight_layout()
        
        return fig
    
    def plot_vif_bars(self, vif_data: pd.DataFrame, threshold: float = 5.0) -> plt.Figure:
        """Plot VIF values as horizontal bars"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sort by VIF value
        vif_data = vif_data.sort_values('VIF', ascending=True)
        
        # Create bar colors based on threshold
        colors = ['red' if vif > threshold else 'green' for vif in vif_data['VIF']]
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(vif_data)), vif_data['VIF'], color=colors)
        
        # Add variable names
        ax.set_yticks(range(len(vif_data)))
        ax.set_yticklabels([f"{row['Variable']}: {row['Description']}" 
                           for _, row in vif_data.iterrows()])
        
        # Add threshold line
        ax.axvline(x=threshold, color='black', linestyle='--', 
                  label=f'Threshold ({threshold})')
        
        # Labels and title
        ax.set_xlabel('VIF Value', fontsize=12)
        ax.set_title('Variance Inflation Factor by Variable', fontsize=16, pad=20)
        ax.legend()
        
        # Add value labels on bars
        for bar, vif in zip(bars, vif_data['VIF']):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                   f'{vif:.2f}', ha='left', va='center')
        
        plt.tight_layout()
        return fig
    
    def visualize_layer(self, layer_name: str, cmap: str = 'viridis') -> plt.Figure:
        """Visualize a bioclimatic layer"""
        if layer_name not in self.available_layers:
            raise ValueError(f"Layer {layer_name} not available")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Read raster
        with rasterio.open(self.available_layers[layer_name]) as src:
            data = src.read(1)
            
            # Mask nodata values
            data = np.ma.masked_where(data == src.nodata, data)
            
            # Convert temperature values if needed
            if int(layer_name.replace('bio', '')) in [1,2,5,6,7,8,9,10,11]:
                data = data / 10.0
            
            # Plot
            im = ax.imshow(data, cmap=cmap, extent=[
                src.bounds.left, src.bounds.right,
                src.bounds.bottom, src.bounds.top
            ])
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Get metadata
            layer_info = self.metadata['layers'].get(layer_name, {})
            layer_desc = layer_info.get('name', layer_name)
            layer_unit = layer_info.get('unit', '')
            
            # Labels
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            ax.set_title(f'{layer_desc} ({layer_name})', fontsize=16, pad=20)
            cbar.set_label(layer_unit, fontsize=12)
            
            # Brazil boundaries
            ax.set_xlim(-74, -34)
            ax.set_ylim(-34, 6)
            
        plt.tight_layout()
        return fig