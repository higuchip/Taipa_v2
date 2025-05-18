import folium
import pandas as pd
from typing import List, Dict, Tuple
import requests
from io import BytesIO
import rasterio
from rasterio.features import geometry_window
from shapely.geometry import Point, box
import numpy as np

def create_occurrence_map(occurrences: List[Dict], center: Tuple[float, float] = None) -> folium.Map:
    """
    Create a Folium map with occurrence points
    
    Args:
        occurrences: List of occurrence dictionaries with lat/lon
        center: Optional center coordinates for the map
    
    Returns:
        Folium map object
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

def extract_raster_values(points: List[Tuple[float, float]], raster_path: str) -> List[float]:
    """
    Extract raster values at given points (placeholder for MVP)
    
    Args:
        points: List of (lat, lon) tuples
        raster_path: Path to raster file
    
    Returns:
        List of extracted values
    """
    # Placeholder implementation for MVP
    # In production, this would use rasterio to extract actual values
    values = []
    for point in points:
        # Simulate extracted values
        values.append(np.random.uniform(-5, 35))  # Temperature-like values
    return values

def calculate_buffer_area(point: Tuple[float, float], buffer_km: float) -> Dict:
    """
    Calculate a buffer area around a point
    
    Args:
        point: (lat, lon) tuple
        buffer_km: Buffer distance in kilometers
    
    Returns:
        Dictionary with bounding box coordinates
    """
    # Rough approximation: 1 degree ~ 111 km
    buffer_deg = buffer_km / 111.0
    
    lat, lon = point
    return {
        "min_lat": lat - buffer_deg,
        "max_lat": lat + buffer_deg,
        "min_lon": lon - buffer_deg,
        "max_lon": lon + buffer_deg,
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
    from folium.plugins import HeatMap
    
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