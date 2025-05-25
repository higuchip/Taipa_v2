import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from typing import List, Tuple, Dict, Union
from statsmodels.stats.outliers_influence import variance_inflation_factor
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Only suppress specific warnings that are known and harmless
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')
warnings.filterwarnings('ignore', message='The frame.append method is deprecated')


class BioclimAnalyzer:
    def __init__(self, data_path: str = "data/worldclim_brazil"):
        self.data_path = Path(data_path)
        self.available_layers = self._scan_available_layers()
        self.metadata = self._load_metadata()
        
    def _scan_available_layers(self) -> Dict[str, Path]:
        """Scan directory for available bioclimatic layers"""
        layers = {}
        
        # Pattern for bioclimatic files
        pattern = "bio*_brazil.tif"
        
        for file in self.data_path.glob(pattern):
            # Extract layer name (e.g., bio1 from bio1_brazil.tif)
            layer_name = file.stem.replace('_brazil', '')
            layers[layer_name] = file
        
        return layers
    
    def _load_metadata(self) -> Dict:
        """Load or create metadata for bioclimatic variables"""
        # Pre-defined metadata for WorldClim bioclimatic variables
        metadata = {
            'layers': {
                'bio1': {'name': 'Annual Mean Temperature', 'unit': '°C'},
                'bio2': {'name': 'Mean Diurnal Range', 'unit': '°C'},
                'bio3': {'name': 'Isothermality', 'unit': '%'},
                'bio4': {'name': 'Temperature Seasonality', 'unit': 'CV'},
                'bio5': {'name': 'Max Temperature of Warmest Month', 'unit': '°C'},
                'bio6': {'name': 'Min Temperature of Coldest Month', 'unit': '°C'},
                'bio7': {'name': 'Temperature Annual Range', 'unit': '°C'},
                'bio8': {'name': 'Mean Temperature of Wettest Quarter', 'unit': '°C'},
                'bio9': {'name': 'Mean Temperature of Driest Quarter', 'unit': '°C'},
                'bio10': {'name': 'Mean Temperature of Warmest Quarter', 'unit': '°C'},
                'bio11': {'name': 'Mean Temperature of Coldest Quarter', 'unit': '°C'},
                'bio12': {'name': 'Annual Precipitation', 'unit': 'mm'},
                'bio13': {'name': 'Precipitation of Wettest Month', 'unit': 'mm'},
                'bio14': {'name': 'Precipitation of Driest Month', 'unit': 'mm'},
                'bio15': {'name': 'Precipitation Seasonality', 'unit': 'CV'},
                'bio16': {'name': 'Precipitation of Wettest Quarter', 'unit': 'mm'},
                'bio17': {'name': 'Precipitation of Driest Quarter', 'unit': 'mm'},
                'bio18': {'name': 'Precipitation of Warmest Quarter', 'unit': 'mm'},
                'bio19': {'name': 'Precipitation of Coldest Quarter', 'unit': 'mm'},
            }
        }
        
        return metadata
    
    def extract_values_at_points_optimized(self, points: List[Tuple[float, float]], 
                                         layers: List[str],
                                         progress_callback=None) -> pd.DataFrame:
        """
        Optimized extraction of raster values at given points.
        Opens each raster once and extracts values for all points.
        
        Args:
            points: List of (lat, lon) tuples
            layers: List of layer names to extract
        
        Returns:
            DataFrame with values for each point and layer
        """
        # Initialize result array
        n_points = len(points)
        
        # Convert points to arrays for vectorized operations
        lats = np.array([p[0] for p in points])
        lons = np.array([p[1] for p in points])
        
        # Initialize result dictionary
        data = {
            'latitude': lats,
            'longitude': lons
        }
        
        # Process each layer
        for layer_idx, layer in enumerate(layers):
            if progress_callback:
                progress_callback(layer_idx / len(layers), f"Processando {layer}...")
            
            if layer not in self.available_layers:
                data[layer] = np.full(n_points, np.nan)
                continue
            
            # Read the entire raster once
            with rasterio.open(self.available_layers[layer]) as src:
                # Read all data at once
                raster_data = src.read(1)
                
                # Convert all lat/lon to pixel coordinates at once
                rows, cols = rasterio.transform.rowcol(src.transform, lons, lats)
                
                # Initialize values array
                values = np.full(n_points, np.nan)
                
                # Extract values for all valid points
                for i in range(n_points):
                    row, col = rows[i], cols[i]
                    
                    # Check bounds
                    if 0 <= row < src.height and 0 <= col < src.width:
                        value = raster_data[row, col]
                        
                        # Handle nodata
                        if value != src.nodata:
                            # Convert temperature values (stored as °C * 10)
                            if 'bio' in layer and int(layer.replace('bio', '')) in [1,2,5,6,7,8,9,10,11]:
                                value = value / 10.0
                            values[i] = value
                
                data[layer] = values
        
        return pd.DataFrame(data)
    
    def extract_values_parallel(self, points: List[Tuple[float, float]], 
                              layers: List[str], max_workers: int = 4) -> pd.DataFrame:
        """
        Parallel extraction using multiple threads.
        
        Args:
            points: List of (lat, lon) tuples
            layers: List of layer names to extract
            max_workers: Number of parallel workers
        
        Returns:
            DataFrame with values for each point and layer
        """
        n_points = len(points)
        
        # Convert points to arrays
        lats = np.array([p[0] for p in points])
        lons = np.array([p[1] for p in points])
        
        # Initialize result dictionary
        data = {
            'latitude': lats,
            'longitude': lons
        }
        
        def extract_layer(layer):
            """Extract values for a single layer"""
            if layer not in self.available_layers:
                return layer, np.full(n_points, np.nan)
            
            with rasterio.open(self.available_layers[layer]) as src:
                raster_data = src.read(1)
                rows, cols = rasterio.transform.rowcol(src.transform, lons, lats)
                
                values = np.full(n_points, np.nan)
                
                for i in range(n_points):
                    row, col = rows[i], cols[i]
                    
                    if 0 <= row < src.height and 0 <= col < src.width:
                        value = raster_data[row, col]
                        
                        if value != src.nodata:
                            if 'bio' in layer and int(layer.replace('bio', '')) in [1,2,5,6,7,8,9,10,11]:
                                value = value / 10.0
                            values[i] = value
                
                return layer, values
        
        # Process layers in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(extract_layer, layer): layer for layer in layers}
            
            for future in as_completed(futures):
                layer, values = future.result()
                data[layer] = values
        
        return pd.DataFrame(data)
    
    def extract_values_cached(self, points: List[Tuple[float, float]], 
                            layers: List[str]) -> pd.DataFrame:
        """
        Extract values with memory caching of raster data.
        Useful when doing multiple extractions on the same layers.
        
        Args:
            points: List of (lat, lon) tuples
            layers: List of layer names to extract
        
        Returns:
            DataFrame with values for each point and layer
        """
        # Cache raster data in memory
        if not hasattr(self, '_raster_cache'):
            self._raster_cache = {}
        
        n_points = len(points)
        lats = np.array([p[0] for p in points])
        lons = np.array([p[1] for p in points])
        
        data = {
            'latitude': lats,
            'longitude': lons
        }
        
        for layer in layers:
            if layer not in self.available_layers:
                data[layer] = np.full(n_points, np.nan)
                continue
            
            # Check cache
            if layer not in self._raster_cache:
                # Load and cache the raster
                with rasterio.open(self.available_layers[layer]) as src:
                    self._raster_cache[layer] = {
                        'data': src.read(1),
                        'transform': src.transform,
                        'nodata': src.nodata,
                        'height': src.height,
                        'width': src.width
                    }
            
            # Use cached data
            cache = self._raster_cache[layer]
            rows, cols = rasterio.transform.rowcol(cache['transform'], lons, lats)
            
            values = np.full(n_points, np.nan)
            
            for i in range(n_points):
                row, col = rows[i], cols[i]
                
                if 0 <= row < cache['height'] and 0 <= col < cache['width']:
                    value = cache['data'][row, col]
                    
                    if value != cache['nodata']:
                        if 'bio' in layer and int(layer.replace('bio', '')) in [1,2,5,6,7,8,9,10,11]:
                            value = value / 10.0
                        values[i] = value
            
            data[layer] = values
        
        return pd.DataFrame(data)
    
    def clear_cache(self):
        """Clear the raster cache to free memory"""
        if hasattr(self, '_raster_cache'):
            self._raster_cache.clear()
    
    # Keep all other methods the same
    def calculate_correlation_matrix(self, df: pd.DataFrame, 
                                   method='pearson') -> pd.DataFrame:
        """Calculate correlation matrix for bioclimatic variables"""
        bio_cols = [col for col in df.columns if col.startswith('bio')]
        return df[bio_cols].corr(method=method)
    
    def calculate_vif(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Variance Inflation Factor for each variable"""
        from statsmodels.tools.tools import add_constant
        
        bio_cols = [col for col in df.columns if col.startswith('bio')]
        bio_data = df[bio_cols].dropna()
        
        X = add_constant(bio_data)
        
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(X.shape[1])]
        
        vif_data = vif_data[vif_data["Variable"] != "const"]
        
        vif_data["Description"] = [self.metadata['layers'].get(var, {}).get('name', var) 
                                  for var in vif_data["Variable"]]
        
        return vif_data.sort_values('VIF', ascending=False)
    
    def select_variables(self, df: pd.DataFrame, 
                        vif_threshold: float = 5.0,
                        correlation_threshold: float = 0.7,
                        method: str = 'vif',
                        return_steps: bool = False) -> Union[List[str], Tuple[List[str], List[Dict]]]:
        """
        Select variables based on multicollinearity analysis
        
        Args:
            df: DataFrame with bioclimatic data
            vif_threshold: Maximum VIF value allowed
            correlation_threshold: Maximum correlation allowed
            method: Selection method ('vif', 'correlation', or 'stepwise')
        
        Returns:
            List of selected variables
        """
        bio_cols = [col for col in df.columns if col.startswith('bio')]
        
        if method == 'vif':
            # Remove variables with high VIF iteratively
            selected_vars = bio_cols.copy()
            steps = []
            iteration = 0
            
            while True:
                iteration += 1
                vif_df = self.calculate_vif(df[selected_vars])
                max_vif = vif_df['VIF'].max()
                
                if max_vif <= vif_threshold:
                    break
                
                # Remove variable with highest VIF
                var_to_remove = vif_df.iloc[0]['Variable']
                selected_vars.remove(var_to_remove)
                
                if return_steps:
                    steps.append({
                        'iteration': iteration,
                        'action': 'vif_removal',
                        'removed_variable': var_to_remove,
                        'vif_value': max_vif,
                        'remaining_variables': selected_vars.copy()
                    })
                
                if len(selected_vars) < 2:
                    break
            
            if return_steps:
                return selected_vars, steps
            return selected_vars
        
        elif method == 'correlation':
            # Remove highly correlated variables
            corr_matrix = self.calculate_correlation_matrix(df)
            steps = []
            
            # Find pairs with high correlation
            high_corr_pairs = []
            for i in range(len(bio_cols)):
                for j in range(i + 1, len(bio_cols)):
                    if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                        high_corr_pairs.append((bio_cols[i], bio_cols[j], 
                                              abs(corr_matrix.iloc[i, j])))
            
            # Sort by correlation value
            high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Remove variables with highest correlations
            vars_to_remove = set()
            for var1, var2, corr in high_corr_pairs:
                if var1 not in vars_to_remove and var2 not in vars_to_remove:
                    # Remove the one with higher average correlation
                    avg_corr1 = corr_matrix[var1].abs().mean()
                    avg_corr2 = corr_matrix[var2].abs().mean()
                    
                    if avg_corr1 > avg_corr2:
                        vars_to_remove.add(var1)
                        if return_steps:
                            steps.append({
                                'action': 'correlation_removal',
                                'removed_variable': var1,
                                'correlation_with': var2,
                                'correlation_value': corr,
                                'reason': f'Higher average correlation ({avg_corr1:.3f} vs {avg_corr2:.3f})'
                            })
                    else:
                        vars_to_remove.add(var2)
                        if return_steps:
                            steps.append({
                                'action': 'correlation_removal',
                                'removed_variable': var2,
                                'correlation_with': var1,
                                'correlation_value': corr,
                                'reason': f'Higher average correlation ({avg_corr2:.3f} vs {avg_corr1:.3f})'
                            })
            
            selected_vars = [var for var in bio_cols if var not in vars_to_remove]
            
            if return_steps:
                return selected_vars, steps
            return selected_vars
        
        else:  # stepwise
            if return_steps:
                return self.stepwise_selection_detailed(df, vif_threshold, correlation_threshold, return_steps=True)
            return self._stepwise_selection(df)
    
    def _stepwise_selection(self, df: pd.DataFrame) -> List[str]:
        """Simplified stepwise selection based on VIF"""
        bio_cols = [col for col in df.columns if col.startswith('bio')]
        
        # Start with variable with lowest average correlation
        corr_matrix = self.calculate_correlation_matrix(df)
        avg_corr = corr_matrix.abs().mean()
        first_var = avg_corr.idxmin()
        
        selected = [first_var]
        remaining = [v for v in bio_cols if v != first_var]
        
        # Add variables one by one
        while remaining:
            best_var = None
            best_vif = float('inf')
            
            for var in remaining:
                temp_vars = selected + [var]
                vif_df = self.calculate_vif(df[temp_vars])
                max_vif = vif_df['VIF'].max()
                
                if max_vif < best_vif:
                    best_vif = max_vif
                    best_var = var
            
            if best_vif <= 5.0 and best_var:
                selected.append(best_var)
                remaining.remove(best_var)
            else:
                break
        
        return selected
    
    def plot_correlation_matrix(self, corr_matrix: pd.DataFrame) -> plt.Figure:
        """Plot correlation matrix heatmap"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
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
        import matplotlib.pyplot as plt
        
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
        import matplotlib.pyplot as plt
        
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
            metadata = self.metadata['layers'].get(layer_name, {})
            unit = metadata.get('unit', '')
            name = metadata.get('name', layer_name)
            
            cbar.set_label(f'{unit}', rotation=270, labelpad=20)
            ax.set_title(f'{layer_name}: {name}', fontsize=16, pad=20)
            
            # Labels
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            
            plt.tight_layout()
        
        return fig
    
    def select_variables_robust(self, df: pd.DataFrame,
                              correlation_threshold: float = 0.7,
                              vif_threshold: float = 10.0,
                              min_variables: int = 3) -> Tuple[List[str], pd.DataFrame]:
        """
        Robust variable selection method that avoids common pitfalls
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with bioclimatic variables
        correlation_threshold : float
            Maximum correlation allowed between variables
        vif_threshold : float
            Maximum VIF allowed
        min_variables : int
            Minimum number of variables to retain
            
        Returns:
        --------
        selected_vars : list
            List of selected variable names
        selection_info : pd.DataFrame
            DataFrame with selection information
        """
        # Get bioclimatic columns
        bio_cols = [col for col in df.columns if col.startswith('bio')]
        
        # Step 1: Remove variables with too many missing values
        missing_pct = df[bio_cols].isnull().sum() / len(df) * 100
        valid_vars = missing_pct[missing_pct < 10].index.tolist()
        
        # Step 2: Calculate correlation matrix
        corr_matrix = df[valid_vars].corr().abs()
        
        # Step 3: Find highly correlated pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Step 4: For each highly correlated pair, keep the one with lower mean correlation
        to_drop = set()
        correlation_pairs = []
        
        for column in upper_tri.columns:
            correlated_features = list(
                upper_tri.index[upper_tri[column] > correlation_threshold]
            )
            for feature in correlated_features:
                if column not in to_drop and feature not in to_drop:
                    # Calculate mean correlation for both
                    mean_corr_col = corr_matrix[column].mean()
                    mean_corr_feat = corr_matrix[feature].mean()
                    
                    # Keep the one with lower mean correlation
                    if mean_corr_col > mean_corr_feat:
                        to_drop.add(column)
                        kept = feature
                    else:
                        to_drop.add(feature)
                        kept = column
                    
                    correlation_pairs.append({
                        'var1': column,
                        'var2': feature,
                        'correlation': corr_matrix.loc[column, feature],
                        'removed': column if column in to_drop else feature,
                        'kept': kept
                    })
        
        # Remove highly correlated variables
        selected_vars = [v for v in valid_vars if v not in to_drop]
        
        # Step 5: Check VIF if we have enough variables
        if len(selected_vars) > 2:
            vif_data = pd.DataFrame()
            vif_data["Variable"] = selected_vars
            X = df[selected_vars].dropna()
            
            vif_values = []
            for i in range(len(selected_vars)):
                try:
                    vif = variance_inflation_factor(X.values, i)
                    vif_values.append(vif)
                except:
                    vif_values.append(np.inf)
            
            vif_data["VIF"] = vif_values
            
            # Remove variables with high VIF iteratively
            while (vif_data["VIF"] > vif_threshold).any() and len(selected_vars) > min_variables:
                # Remove variable with highest VIF
                max_vif_idx = vif_data["VIF"].idxmax()
                removed_var = vif_data.loc[max_vif_idx, "Variable"]
                selected_vars.remove(removed_var)
                
                # Recalculate VIF
                if len(selected_vars) > 2:
                    vif_data = pd.DataFrame()
                    vif_data["Variable"] = selected_vars
                    X = df[selected_vars].dropna()
                    
                    vif_values = []
                    for i in range(len(selected_vars)):
                        try:
                            vif = variance_inflation_factor(X.values, i)
                            vif_values.append(vif)
                        except:
                            vif_values.append(np.inf)
                    
                    vif_data["VIF"] = vif_values
                else:
                    break
        
        # Create selection info DataFrame
        selection_info = pd.DataFrame({
            'variable': bio_cols,
            'selected': [v in selected_vars for v in bio_cols],
            'reason_removed': 'kept'
        })
        
        # Add removal reasons
        for var in bio_cols:
            if var not in valid_vars:
                selection_info.loc[selection_info['variable'] == var, 'reason_removed'] = 'too many missing values'
            elif var in to_drop:
                selection_info.loc[selection_info['variable'] == var, 'reason_removed'] = 'high correlation'
            elif var in valid_vars and var not in selected_vars:
                selection_info.loc[selection_info['variable'] == var, 'reason_removed'] = 'high VIF'
        
        return selected_vars, selection_info
    
    def stepwise_selection_detailed(self, df: pd.DataFrame,
                                  vif_threshold: float = 5.0,
                                  correlation_threshold: float = 0.7,
                                  return_steps: bool = True) -> Union[List[str], Tuple[List[str], List[Dict]]]:
        """
        Stepwise variable selection with detailed steps
        
        This method uses an improved algorithm that:
        1. First removes variables with VIF > threshold
        2. Then removes correlated variables, keeping the one with lower VIF
        3. Uses proper VIF calculation for multicollinearity detection
        """
        bio_cols = [col for col in df.columns if col.startswith('bio')]
        variables = bio_cols.copy()
        steps = []
        
        iteration = 0
        while True:
            iteration += 1
            
            # Check VIF
            vif_data = self.calculate_vif(df[variables])
            max_vif = vif_data['VIF'].max()
            
            # Remove high VIF
            if max_vif > vif_threshold:
                var_to_remove = vif_data.iloc[0]['Variable']
                variables.remove(var_to_remove)
                
                if return_steps:
                    steps.append({
                        'iteration': iteration,
                        'action': 'vif_removal',
                        'removed_variable': var_to_remove,
                        'vif_value': max_vif,
                        'remaining_variables': variables.copy()
                    })
            
            # Check correlations
            corr_matrix = self.calculate_correlation_matrix(df[variables])
            upper_triangle = np.triu(corr_matrix.values, k=1)
            high_corr_indices = np.where(np.abs(upper_triangle) > correlation_threshold)
            
            if len(high_corr_indices[0]) > 0:
                to_remove = set()
                
                for i, j in zip(high_corr_indices[0], high_corr_indices[1]):
                    var1 = variables[i]
                    var2 = variables[j]
                    
                    if var1 not in to_remove and var2 not in to_remove:
                        # Calculate VIF for both variables in the current set
                        # to decide which one to remove
                        current_vars = [v for v in variables if v not in to_remove]
                        
                        # Get VIF for each variable
                        if len(current_vars) > 2:
                            X_current = df[current_vars].dropna()
                            vif_dict = {}
                            
                            for idx, var in enumerate(current_vars):
                                try:
                                    vif_dict[var] = variance_inflation_factor(X_current.values, idx)
                                except:
                                    vif_dict[var] = np.inf
                            
                            # Remove the variable with higher VIF
                            vif1 = vif_dict.get(var1, np.inf)
                            vif2 = vif_dict.get(var2, np.inf)
                            
                            # If both have high VIF, remove the one with higher value
                            # If VIFs are similar, remove based on correlation strength
                            if abs(vif1 - vif2) < 0.1:  # Similar VIF values
                                # Remove the one with higher average correlation
                                corr1 = np.mean(np.abs(corr_matrix.loc[var1, [v for v in variables if v != var1]]))
                                corr2 = np.mean(np.abs(corr_matrix.loc[var2, [v for v in variables if v != var2]]))
                                to_remove.add(var1 if corr1 > corr2 else var2)
                            else:
                                to_remove.add(var1 if vif1 > vif2 else var2)
                        else:
                            # If only 2 variables left, remove one arbitrarily
                            to_remove.add(var2)
                
                if to_remove:
                    variables = [var for var in variables if var not in to_remove]
                    if return_steps:
                        steps.append({
                            'iteration': iteration,
                            'action': 'correlation_removal',
                            'removed_variables': list(to_remove),
                            'remaining_variables': variables.copy()
                        })
                else:
                    break
            else:
                break
            
            if len(variables) < 2:
                break
        
        if return_steps:
            return variables, steps
        return variables