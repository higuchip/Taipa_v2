import geopandas as gpd
from shapely.geometry import Point
import geobr
import streamlit as st

@st.cache_data
def get_brazil_boundary():
    """
    Get Brazil boundary from geobr (official IBGE data)
    Returns a Polygon/MultiPolygon object representing Brazil
    """
    try:
        # Download Brazil boundary from geobr
        # year=2020 for most recent data
        brazil = geobr.read_country(year=2020)
        
        # Simplify geometry for better performance
        # tolerance in degrees (0.01 = ~1km)
        brazil_simplified = brazil.geometry.simplify(0.01)
        
        return brazil_simplified.iloc[0]
        
    except Exception as e:
        st.warning(f"Erro ao carregar dados do geobr: {e}. Usando limite simplificado.")
        return get_simplified_brazil_boundary()

def get_simplified_brazil_boundary():
    """
    Fallback: Get a simplified boundary of Brazil
    Used if geobr is not available
    """
    # Simplified Brazil boundary coordinates (fallback)
    from shapely.geometry import Polygon
    
    brazil_coords = [
        # Sul
        (-57.64, -30.22), (-53.37, -33.77), (-48.55, -28.84),
        (-48.64, -26.23), (-48.51, -25.43), (-54.26, -24.05),
        # Sudeste  
        (-48.02, -25.01), (-44.99, -23.03), (-42.02, -22.91),
        (-40.96, -21.26), (-39.27, -17.89),
        # Nordeste
        (-38.50, -12.97), (-37.05, -11.35), (-35.74, -9.65),
        (-34.87, -8.05), (-34.86, -7.12), (-35.21, -5.80),
        (-38.54, -3.72), (-41.77, -2.90), (-44.28, -2.53),
        # Norte
        (-48.50, -1.46), (-51.07, -0.03), (-50.73, 2.04),
        (-60.03, 3.13), (-67.08, 0.19), (-69.90, -4.38),
        (-73.63, -7.37), (-70.55, -9.33), (-65.29, -10.58),
        (-60.04, -13.45),
        # Centro-Oeste
        (-57.66, -16.04), (-55.63, -22.22), (-54.62, -25.45),
        # Volta ao início
        (-57.64, -30.22)
    ]
    
    return Polygon(brazil_coords)

@st.cache_data
def get_brazil_gdf():
    """
    Get Brazil boundary as a GeoDataFrame
    Uses geobr for official data
    """
    try:
        # Get official Brazil boundary from geobr
        brazil = geobr.read_country(year=2020)
        return brazil
    except:
        # Fallback to simplified boundary
        brazil_polygon = get_simplified_brazil_boundary()
        gdf = gpd.GeoDataFrame([1], geometry=[brazil_polygon], crs='EPSG:4326')
        return gdf

def is_point_in_brazil(lat, lon):
    """
    Check if a point is within Brazil's boundaries
    
    Args:
        lat: Latitude
        lon: Longitude
    
    Returns:
        Boolean indicating if point is in Brazil
    """
    brazil_polygon = get_brazil_boundary()
    point = Point(lon, lat)
    return brazil_polygon.contains(point)

@st.cache_data
def get_brazil_states():
    """
    Get all Brazilian states boundaries from geobr
    Useful for more detailed analysis
    """
    try:
        states = geobr.read_state(year=2020)
        return states
    except:
        return None

def get_state_for_point(lat, lon):
    """
    Get the Brazilian state for a given point
    
    Args:
        lat: Latitude
        lon: Longitude
    
    Returns:
        State name or None if not in Brazil
    """
    states = get_brazil_states()
    if states is None:
        return None
    
    point = Point(lon, lat)
    
    for idx, state in states.iterrows():
        if state.geometry.contains(point):
            return state['name_state']
    
    return None