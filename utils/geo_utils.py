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
    
    # Add occurrence markers
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
            
            folium.Marker(
                location=[occ["lat"], occ["lon"]],
                popup=popup_text,
                tooltip=f"{occ.get('scientific_name', 'Unknown')} - {occ.get('year', 'Unknown')}"
            ).add_to(m)
    
    # Add marker cluster for better visualization with many points
    if len(occurrences) > 50:
        from folium.plugins import MarkerCluster
        marker_cluster = MarkerCluster().add_to(m)
        
        for occ in occurrences:
            if occ.get("lat") and occ.get("lon"):
                folium.Marker(
                    location=[occ["lat"], occ["lon"]],
                    popup=f"{occ.get('scientific_name', 'Unknown')} - {occ.get('year', 'Unknown')}"
                ).add_to(marker_cluster)
    
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

def get_worldclim_layers() -> List[Dict[str, str]]:
    """
    Get list of WorldClim bioclimatic variables
    
    Returns:
        List of dictionaries with variable info
    """
    variables = [
        {"code": "bio1", "name": "Annual Mean Temperature", "unit": "°C"},
        {"code": "bio2", "name": "Mean Diurnal Range", "unit": "°C"},
        {"code": "bio3", "name": "Isothermality", "unit": "%"},
        {"code": "bio4", "name": "Temperature Seasonality", "unit": "CV"},
        {"code": "bio5", "name": "Max Temperature of Warmest Month", "unit": "°C"},
        {"code": "bio6", "name": "Min Temperature of Coldest Month", "unit": "°C"},
        {"code": "bio7", "name": "Temperature Annual Range", "unit": "°C"},
        {"code": "bio8", "name": "Mean Temperature of Wettest Quarter", "unit": "°C"},
        {"code": "bio9", "name": "Mean Temperature of Driest Quarter", "unit": "°C"},
        {"code": "bio10", "name": "Mean Temperature of Warmest Quarter", "unit": "°C"},
        {"code": "bio11", "name": "Mean Temperature of Coldest Quarter", "unit": "°C"},
        {"code": "bio12", "name": "Annual Precipitation", "unit": "mm"},
        {"code": "bio13", "name": "Precipitation of Wettest Month", "unit": "mm"},
        {"code": "bio14", "name": "Precipitation of Driest Month", "unit": "mm"},
        {"code": "bio15", "name": "Precipitation Seasonality", "unit": "CV"},
        {"code": "bio16", "name": "Precipitation of Wettest Quarter", "unit": "mm"},
        {"code": "bio17", "name": "Precipitation of Driest Quarter", "unit": "mm"},
        {"code": "bio18", "name": "Precipitation of Warmest Quarter", "unit": "mm"},
        {"code": "bio19", "name": "Precipitation of Coldest Quarter", "unit": "mm"}
    ]
    return variables