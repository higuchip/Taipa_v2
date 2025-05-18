import requests
import json
from typing import Dict, List, Optional

def search_species(scientific_name: str, country: Optional[str] = None, limit: int = 100) -> Dict:
    """
    Search for species occurrences in GBIF
    
    Args:
        scientific_name: Scientific name of the species
        country: Optional country code (e.g., 'BR' for Brazil)
        limit: Maximum number of results to return
    
    Returns:
        Dictionary with occurrence data
    """
    base_url = "https://api.gbif.org/v1/occurrence/search"
    
    params = {
        "scientificName": scientific_name,
        "limit": limit,
        "hasCoordinate": True,
        "hasGeospatialIssue": False
    }
    
    if country:
        params["country"] = country
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "results": []}

def get_species_key(scientific_name: str) -> Optional[int]:
    """
    Get GBIF species key from scientific name
    
    Args:
        scientific_name: Scientific name of the species
    
    Returns:
        GBIF species key or None if not found
    """
    base_url = "https://api.gbif.org/v1/species/match"
    
    params = {
        "name": scientific_name,
        "strict": False
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("matchType") != "NONE":
            return data.get("usageKey")
        return None
    except requests.exceptions.RequestException:
        return None

def get_countries() -> List[Dict[str, str]]:
    """
    Get list of countries from GBIF
    
    Returns:
        List of dictionaries with country code and name
    """
    # Hardcoded list of common countries for MVP
    # In production, this could fetch from GBIF enumeration API
    countries = [
        {"code": "BR", "name": "Brazil"},
        {"code": "US", "name": "United States"},
        {"code": "MX", "name": "Mexico"},
        {"code": "CA", "name": "Canada"},
        {"code": "AR", "name": "Argentina"},
        {"code": "CO", "name": "Colombia"},
        {"code": "PE", "name": "Peru"},
        {"code": "CL", "name": "Chile"},
        {"code": "VE", "name": "Venezuela"},
        {"code": "EC", "name": "Ecuador"},
        {"code": "BO", "name": "Bolivia"},
        {"code": "PY", "name": "Paraguay"},
        {"code": "UY", "name": "Uruguay"},
        {"code": "GY", "name": "Guyana"},
        {"code": "SR", "name": "Suriname"},
        {"code": "GF", "name": "French Guiana"}
    ]
    return countries

def format_occurrence_for_map(occurrence: Dict) -> Dict:
    """
    Format GBIF occurrence data for map display
    
    Args:
        occurrence: GBIF occurrence record
    
    Returns:
        Formatted dictionary for map markers
    """
    return {
        "lat": occurrence.get("decimalLatitude"),
        "lon": occurrence.get("decimalLongitude"),
        "scientific_name": occurrence.get("scientificName", "Unknown"),
        "country": occurrence.get("country", "Unknown"),
        "state": occurrence.get("stateProvince", "Unknown"),
        "year": occurrence.get("year", "Unknown"),
        "institution": occurrence.get("institutionCode", "Unknown"),
        "basis_of_record": occurrence.get("basisOfRecord", "Unknown"),
        "key": occurrence.get("key")
    }