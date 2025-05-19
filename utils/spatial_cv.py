import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.cluster import KMeans
from typing import Iterator, Tuple, Optional
import geopandas as gpd
from shapely.geometry import Point
import logging

class SpatialKFold(BaseCrossValidator):
    """Spatial K-Fold cross-validation for SDM"""
    
    def __init__(self, n_splits: int = 5, buffer_distance: float = None):
        self.n_splits = n_splits
        self.buffer_distance = buffer_distance
        self.logger = logging.getLogger(__name__)
    
    def split(self, X: np.ndarray, y: np.ndarray = None, 
              coordinates: np.ndarray = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate spatial train/test splits"""
        
        if coordinates is None:
            raise ValueError("Coordinates must be provided for spatial CV")
        
        n_samples = X.shape[0]
        
        # Cluster spatial locations
        kmeans = KMeans(n_clusters=self.n_splits, random_state=42)
        cluster_labels = kmeans.fit_predict(coordinates)
        
        # Generate folds based on spatial clusters
        for fold in range(self.n_splits):
            test_mask = cluster_labels == fold
            train_mask = ~test_mask
            
            # Apply buffer if specified
            if self.buffer_distance:
                train_mask = self._apply_buffer(
                    train_mask, test_mask, coordinates
                )
            
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            yield train_idx, test_idx
    
    def _apply_buffer(self, train_mask: np.ndarray, test_mask: np.ndarray,
                     coordinates: np.ndarray) -> np.ndarray:
        """Apply spatial buffer to avoid spatial autocorrelation"""
        
        # Convert to GeoDataFrame
        points = [Point(coord) for coord in coordinates]
        gdf = gpd.GeoDataFrame(geometry=points)
        
        # Get test points
        test_points = gdf[test_mask]
        
        # Buffer around test points
        buffer_union = test_points.buffer(self.buffer_distance).unary_union
        
        # Remove training points within buffer
        train_points = gdf[train_mask]
        within_buffer = train_points.within(buffer_union)
        
        # Update training mask
        train_mask[train_mask] = ~within_buffer.values
        
        return train_mask
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits


class BlockCV:
    """Block cross-validation for spatial data"""
    
    def __init__(self, n_blocks: int = 10, random_state: int = 42):
        self.n_blocks = n_blocks
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
    
    def create_blocks(self, coordinates: np.ndarray) -> np.ndarray:
        """Create spatial blocks based on coordinates"""
        
        # Get extent
        min_x, min_y = coordinates.min(axis=0)
        max_x, max_y = coordinates.max(axis=0)
        
        # Calculate block dimensions
        n_blocks_x = int(np.sqrt(self.n_blocks))
        n_blocks_y = int(np.ceil(self.n_blocks / n_blocks_x))
        
        block_width = (max_x - min_x) / n_blocks_x
        block_height = (max_y - min_y) / n_blocks_y
        
        # Assign points to blocks
        block_labels = np.zeros(len(coordinates), dtype=int)
        
        for i, (x, y) in enumerate(coordinates):
            block_x = min(int((x - min_x) / block_width), n_blocks_x - 1)
            block_y = min(int((y - min_y) / block_height), n_blocks_y - 1)
            block_labels[i] = block_y * n_blocks_x + block_x
        
        return block_labels
    
    def split(self, X: np.ndarray, y: np.ndarray = None,
              coordinates: np.ndarray = None,
              n_splits: int = 5) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate block-based train/test splits"""
        
        if coordinates is None:
            raise ValueError("Coordinates must be provided for block CV")
        
        # Create blocks
        block_labels = self.create_blocks(coordinates)
        unique_blocks = np.unique(block_labels)
        
        # Randomly assign blocks to folds
        np.random.seed(self.random_state)
        np.random.shuffle(unique_blocks)
        
        fold_assignment = np.array_split(unique_blocks, n_splits)
        
        # Generate folds
        for fold_blocks in fold_assignment:
            test_mask = np.isin(block_labels, fold_blocks)
            train_mask = ~test_mask
            
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            yield train_idx, test_idx


def spatial_train_test_split(X: np.ndarray, y: np.ndarray,
                           coordinates: np.ndarray,
                           test_size: float = 0.2,
                           method: str = 'random',
                           buffer_distance: Optional[float] = None) -> Tuple:
    """
    Spatial train-test split
    
    Parameters
    ----------
    X : array-like
        Features
    y : array-like
        Labels
    coordinates : array-like
        Spatial coordinates (lon, lat)
    test_size : float
        Proportion of data for testing
    method : str
        Split method: 'random', 'block', 'cluster'
    buffer_distance : float, optional
        Buffer distance to avoid spatial autocorrelation
    
    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
    """
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    if method == 'random':
        # Random spatial split
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        
    elif method == 'block':
        # Block-based split
        block_cv = BlockCV(n_blocks=int(1/test_size))
        splits = list(block_cv.split(X, y, coordinates, n_splits=1))
        train_idx, test_idx = splits[0]
        
    elif method == 'cluster':
        # Cluster-based split
        n_clusters = int(1/test_size)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(coordinates)
        
        # Select one cluster for testing
        test_cluster = np.random.choice(n_clusters)
        test_idx = np.where(cluster_labels == test_cluster)[0]
        train_idx = np.where(cluster_labels != test_cluster)[0]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply buffer if specified
    if buffer_distance and method != 'block':
        train_mask = np.zeros(n_samples, dtype=bool)
        train_mask[train_idx] = True
        test_mask = np.zeros(n_samples, dtype=bool)
        test_mask[test_idx] = True
        
        # Remove training points within buffer of test points
        points = [Point(coord) for coord in coordinates]
        gdf = gpd.GeoDataFrame(geometry=points)
        test_points = gdf[test_mask]
        
        buffer_union = test_points.buffer(buffer_distance).unary_union
        train_points = gdf[train_mask]
        within_buffer = train_points.within(buffer_union)
        
        train_mask[train_idx] = ~within_buffer.values
        train_idx = np.where(train_mask)[0]
    
    return (X[train_idx], X[test_idx], 
            y[train_idx], y[test_idx])