import folium
from folium.plugins import Draw, HeatMap
from typing import List, Dict, Tuple, Union
import rasterio
import numpy as np
import os
import math

# Constants for geographic calculations
KM_PER_DEGREE_LAT = 111.0  # Approximately constant
WORLDCLIM_TEMP_SCALE = 10.0  # WorldClim stores temperature * 10

def create_occurrence_map(occurrences: List[Dict], center: Tuple[float, float] = None) -> folium.Map:
    """
    Create a Folium map with occurrence points and drawing tools
    
    Args:
        occurrences: List of occurrence dictionaries with lat/lon
        center: Optional center coordinates for the map
    
    Returns:
        Folium map object with drawing capabilities
    """
    if not center and occurrences:
        # Calculate center from occurrences
        lats = [occ["lat"] for occ in occurrences if occ.get("lat")]
        lons = [occ["lon"] for occ in occurrences if occ.get("lon")]
        if lats and lons:
            center = (sum(lats) / len(lats), sum(lons) / len(lons))
        else:
            center = (0, 0)
    elif not center:
        center = (0, 0)
    
    # Create base map
    m = folium.Map(location=center, zoom_start=5)
    
    # Add drawing tools
    draw = Draw(
        export=False,
        position='topleft',
        draw_options={
            'polygon': {
                'allowIntersection': False,
                'shapeOptions': {
                    'color': 'red',
                    'fillColor': 'red', 
                    'fillOpacity': 0.2
                }
            },
            'rectangle': {
                'shapeOptions': {
                    'color': 'red',
                    'fillColor': 'red',
                    'fillOpacity': 0.2
                }
            },
            'polyline': False,
            'circle': False,
            'marker': False,
            'circlemarker': False,
        }
    )
    draw.add_to(m)
    
    # Add occurrence markers as simple blue circles
    for occ in occurrences:
        if occ.get("lat") and occ.get("lon"):
            popup_text = f"""
            <b>{occ.get('scientific_name', 'Unknown')}</b><br>
            Country: {occ.get('country', 'Unknown')}<br>
            State: {occ.get('state', 'Unknown')}<br>
            Year: {occ.get('year', 'Unknown')}<br>
            Institution: {occ.get('institution', 'Unknown')}<br>
            Basis: {occ.get('basis_of_record', 'Unknown')}
            """
            
            # Use CircleMarker for a simple blue circle
            folium.CircleMarker(
                location=[occ["lat"], occ["lon"]],
                radius=5,
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"{occ.get('scientific_name', 'Unknown')} - {occ.get('year', 'Unknown')}",
                color='#0000FF',  # Blue border
                weight=2,
                fill=True,
                fillColor='#0000FF',  # Blue fill
                fillOpacity=0.7
            ).add_to(m)
    
    return m

def extract_raster_values(points: Union[List[Tuple[float, float]], np.ndarray], 
                         raster_paths: Union[str, List[str]]) -> Union[List[float], Dict[str, List[float]]]:
    """
    Extract raster values at given points
    
    NOTE: This function is provided for compatibility but is NOT used in the main
    workflow. The BioclimAnalyzer class uses its own optimized extraction methods.
    Temperature scaling (bio1-bio11) is properly handled here.
    
    Args:
        points: List of (lat, lon) tuples or numpy array
        raster_paths: Path to raster file or list of paths
    
    Returns:
        List of extracted values or dict of values per layer
    """
    # Convert to list of tuples if numpy array
    if isinstance(points, np.ndarray):
        if points.shape[1] == 2:
            # Assume columns are lon, lat
            points = [(row[1], row[0]) for row in points]
        else:
            raise ValueError("Array must have 2 columns (longitude, latitude)")
    
    # Handle single raster or multiple rasters
    if isinstance(raster_paths, str):
        raster_paths = [raster_paths]
        return_dict = False
    else:
        return_dict = True
    
    results = {}
    
    for raster_path in raster_paths:
        values = []
        # Get layer name for temperature scaling check
        layer_name = os.path.basename(raster_path).replace('.tif', '')
        
        # Check if raster file exists
        if os.path.exists(raster_path):
            try:
                with rasterio.open(raster_path) as src:
                    # Extract values for each point
                    for lat, lon in points:
                        # Sample the raster at the point location
                        # rasterio expects (lon, lat) order
                        sample = list(src.sample([(lon, lat)]))[0]
                        # Get first band value
                        value = float(sample[0])
                        
                        # Check for nodata values
                        if src.nodata is not None and value == src.nodata:
                            value = np.nan
                        else:
                            # WorldClim temperature data is stored as integers multiplied by 10
                            # Check if this is a temperature variable (bio1-bio11)
                            if any(temp_var in layer_name for temp_var in 
                                   ['bio1', 'bio2', 'bio3', 'bio4', 'bio5', 'bio6', 
                                    'bio7', 'bio8', 'bio9', 'bio10', 'bio11']):
                                value = value / WORLDCLIM_TEMP_SCALE
                            
                        values.append(value)
                        
            except (rasterio.errors.RasterioIOError, ValueError) as e:
                print(f"Error reading {raster_path}: {str(e)}")
                # Fill with NaN if error
                values = [np.nan] * len(points)
        else:
            print(f"Warning: Raster file not found: {raster_path}")
            # Fill with NaN if file not found
            values = [np.nan] * len(points)
        
        # Store results using layer name as key
        results[layer_name] = values
    
    return results if return_dict else results[list(results.keys())[0]]

def calculate_buffer_area(point: Tuple[float, float], buffer_km: float) -> Dict:
    """
    Calculate a buffer area around a point
    
    Args:
        point: (lat, lon) tuple
        buffer_km: Buffer distance in kilometers
    
    Returns:
        Dictionary with bounding box coordinates
    """
    lat, lon = point
    
    # Latitude: 1 degree ~ KM_PER_DEGREE_LAT km (approximately constant)
    buffer_lat_deg = buffer_km / KM_PER_DEGREE_LAT
    
    # Longitude: varies with latitude
    # At the equator: 1 degree ~ KM_PER_DEGREE_LAT km
    # At latitude φ: 1 degree ~ KM_PER_DEGREE_LAT * cos(φ) km
    lat_rad = math.radians(lat)
    km_per_lon_degree = KM_PER_DEGREE_LAT * math.cos(lat_rad)
    
    # Avoid division by zero near poles
    if km_per_lon_degree > 0.1:
        buffer_lon_deg = buffer_km / km_per_lon_degree
    else:
        buffer_lon_deg = buffer_lat_deg  # Fallback for polar regions
    
    return {
        "min_lat": lat - buffer_lat_deg,
        "max_lat": lat + buffer_lat_deg,
        "min_lon": lon - buffer_lon_deg,
        "max_lon": lon + buffer_lon_deg,
        "center": point,
        "buffer_km": buffer_km
    }

def create_heatmap(occurrences: List[Dict]) -> folium.Map:
    """
    Create a heatmap from occurrence points
    
    Args:
        occurrences: List of occurrence dictionaries
    
    Returns:
        Folium map with heatmap layer
    """
    # Extract coordinates
    heat_data = []
    for occ in occurrences:
        if occ.get("lat") and occ.get("lon"):
            heat_data.append([occ["lat"], occ["lon"]])
    
    if not heat_data:
        return create_occurrence_map(occurrences)
    
    # Calculate center
    lats = [point[0] for point in heat_data]
    lons = [point[1] for point in heat_data]
    center = (sum(lats) / len(lats), sum(lons) / len(lons))
    
    # Create map
    m = folium.Map(location=center, zoom_start=5)
    
    # Add heatmap
    HeatMap(heat_data).add_to(m)
    
    return m

def get_worldclim_layers() -> Dict[str, str]:
    """
    Get list of WorldClim bioclimatic variables
    
    Returns:
        Dictionary with variable codes as keys and descriptions as values
    """
    variables = {
        "bio1": "Annual Mean Temperature (°C)",
        "bio2": "Mean Diurnal Range (°C)",
        "bio3": "Isothermality (%)",
        "bio4": "Temperature Seasonality (CV)",
        "bio5": "Max Temperature of Warmest Month (°C)",
        "bio6": "Min Temperature of Coldest Month (°C)",
        "bio7": "Temperature Annual Range (°C)",
        "bio8": "Mean Temperature of Wettest Quarter (°C)",
        "bio9": "Mean Temperature of Driest Quarter (°C)",
        "bio10": "Mean Temperature of Warmest Quarter (°C)",
        "bio11": "Mean Temperature of Coldest Quarter (°C)",
        "bio12": "Annual Precipitation (mm)",
        "bio13": "Precipitation of Wettest Month (mm)",
        "bio14": "Precipitation of Driest Month (mm)",
        "bio15": "Precipitation Seasonality (CV)",
        "bio16": "Precipitation of Wettest Quarter (mm)",
        "bio17": "Precipitation of Driest Quarter (mm)",
        "bio18": "Precipitation of Warmest Quarter (mm)",
        "bio19": "Precipitation of Coldest Quarter (mm)"
    }
    return variables