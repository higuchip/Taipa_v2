import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Point
import random
from utils.brazil_boundary import get_brazil_gdf, is_point_in_brazil

def generate_pseudo_absences(presence_points, n_absences, buffer_dist, exclusion_radius):
    """
    Generate pseudo-absence points using buffer strategy
    
    Args:
        presence_points: List of (lat, lon) tuples
        n_absences: Number of pseudo-absences to generate
        buffer_dist: Buffer distance around presence points (km)
        exclusion_radius: Minimum distance from presence points (km)
    
    Returns:
        List of pseudo-absence points as (lat, lon) tuples
    """
    # Get Brazil boundary
    brazil_gdf = get_brazil_gdf()
    brazil_gdf_projected = brazil_gdf.to_crs('EPSG:32723')
    brazil_boundary = brazil_gdf_projected.geometry[0]
    
    # Convert to GeoDataFrame
    presence_gdf = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lat, lon in presence_points],
        crs='EPSG:4326'
    )
    
    # Project to metric system for distance calculations
    presence_gdf_projected = presence_gdf.to_crs('EPSG:32723')  # UTM 23S (Brazil)
    
    # Create buffer around presence points
    buffer_dist_m = buffer_dist * 1000  # Convert km to meters
    buffer_area = presence_gdf_projected.buffer(buffer_dist_m).unary_union
    
    # Create exclusion zones around presence points
    exclusion_dist_m = exclusion_radius * 1000
    exclusion_zones = presence_gdf_projected.buffer(exclusion_dist_m).unary_union
    
    # Calculate sampling area (buffer minus exclusion, intersected with Brazil)
    sampling_area = buffer_area.difference(exclusion_zones)
    sampling_area = sampling_area.intersection(brazil_boundary)
    
    # Generate random points within sampling area
    pseudo_absences = []
    attempts = 0
    max_attempts = n_absences * 20  # Increase attempts due to Brazil constraint
    
    while len(pseudo_absences) < n_absences and attempts < max_attempts:
        # Get bounds of sampling area
        minx, miny, maxx, maxy = sampling_area.bounds
        
        # Generate random point
        random_point = Point(
            random.uniform(minx, maxx),
            random.uniform(miny, maxy)
        )
        
        # Check if point is within sampling area
        if sampling_area.contains(random_point):
            # Convert back to lat/lon
            point_latlon = gpd.GeoSeries([random_point], crs='EPSG:32723').to_crs('EPSG:4326')
            lat, lon = point_latlon.iloc[0].y, point_latlon.iloc[0].x
            
            # Double check with Brazil validation
            if is_point_in_brazil(lat, lon):
                pseudo_absences.append((lat, lon))
        
        attempts += 1
    
    # If we couldn't generate enough points, show warning
    if len(pseudo_absences) < n_absences:
        st.warning(f"Apenas {len(pseudo_absences)} pseudo-ausências foram geradas dentro do Brasil. Tente aumentar a distância do buffer.")
    
    return pseudo_absences

def render_page():
    st.title("Geração de Pseudo-ausências")
    st.markdown("Gere pontos de pseudo-ausência para modelagem SDM usando estratégia de buffer")
    
    # Check for presence data in session state
    if 'occurrence_data' not in st.session_state:
        st.warning("⚠️ Nenhum dado de presença disponível. Por favor, busque dados de ocorrência primeiro.")
        
        # Mostrar status do progresso
        st.error("""
        ❌ **Pré-requisito não atendido**
        
        Para gerar pseudo-ausências, você precisa primeiro:
        1. Buscar uma espécie no GBIF
        2. Confirmar os dados de ocorrência
        
        **Status atual:**
        - Espécie selecionada: {}
        - Ocorrências disponíveis: {}
        """.format(
            st.session_state.get('species_name', 'Nenhuma'),
            st.session_state.get('n_occurrences', 0)
        ))
        
        if st.button("Ir para Busca de Espécies"):
            st.switch_page("pages/pagina_busca_api.py")
        return
    
    # Get species name if available
    species_name = st.session_state.get('species_name', 'Espécie')
    st.info(f"Gerando pseudo-ausências para: **{species_name}**")
    
    # Configuration section
    st.subheader("Configurações de Geração")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_absences = st.number_input(
            "Número de pseudo-ausências",
            min_value=10,
            max_value=5000,
            value=st.session_state.get('n_occurrences', 100),
            step=10,
            help="Recomenda-se usar o mesmo número de presenças"
        )
    
    with col2:
        buffer_dist = st.slider(
            "Distância do buffer (km)",
            min_value=50,
            max_value=1000,
            value=200,
            step=10,
            help="Raio máximo ao redor dos pontos de presença"
        )
    
    with col3:
        exclusion_radius = st.slider(
            "Raio de exclusão (km)",
            min_value=5,
            max_value=100,
            value=10,
            step=5,
            help="Distância mínima dos pontos de presença"
        )
    
    # Generate button
    if st.button("Gerar Pseudo-ausências", type="primary", use_container_width=True):
        with st.spinner("Gerando pseudo-ausências..."):
            # Get presence data
            presence_data = st.session_state['occurrence_data']
            presence_points = [(occ['lat'], occ['lon']) for occ in presence_data 
                              if occ.get('lat') and occ.get('lon')]
            
            # Generate pseudo-absences
            pseudo_absences = generate_pseudo_absences(
                presence_points,
                n_absences,
                buffer_dist,
                exclusion_radius
            )
            
            # Store in session state
            st.session_state['pseudo_absences'] = pseudo_absences
            st.session_state['presence_points'] = presence_points
            st.session_state['map_generated'] = True
            
            # Also store as DataFrame for easier processing
            pseudo_absence_df = pd.DataFrame(pseudo_absences, columns=['decimalLatitude', 'decimalLongitude'])
            pseudo_absence_df['type'] = 'absence'
            st.session_state['pseudo_absence_data'] = pseudo_absence_df
            
            # Display results
            st.success(f"✅ {len(pseudo_absences)} pseudo-ausências geradas!")
            
            # Indicador de conclusão
            st.info("""
            ✅ **Etapa Concluída!**
            
            Você gerou {} pseudo-ausências para a espécie {}.
            
            **Próximo passo:** Vá para o Módulo 3 - Análise Bioclimática
            """.format(len(pseudo_absences), species_name))
    
    # Display results if available
    if 'pseudo_absences' in st.session_state and st.session_state.get('map_generated', False):
        st.subheader("Visualização dos Resultados")
        
        # Get data from session state
        pseudo_absences = st.session_state['pseudo_absences']
        presence_points = st.session_state['presence_points']
        
        # Create container for map
        map_container = st.container()
        
        with map_container:
            # Calculate center
            all_lats = [p[0] for p in presence_points + pseudo_absences]
            all_lons = [p[1] for p in presence_points + pseudo_absences]
            center_lat = np.mean(all_lats)
            center_lon = np.mean(all_lons)
            
            # Create map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
            
            # Add presence points (blue)
            for lat, lon in presence_points:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    popup=f"Presença: ({lat:.3f}, {lon:.3f})",
                    color='blue',
                    fill=True,
                    fillColor='blue',
                    fillOpacity=0.7
                ).add_to(m)
            
            # Add pseudo-absence points (red)
            for lat, lon in pseudo_absences:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    popup=f"Pseudo-ausência: ({lat:.3f}, {lon:.3f})",
                    color='red',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.7
                ).add_to(m)
            
            # Display map
            st_folium(m, width=700, height=500, returned_objects=["bounds"])
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Presenças", len(presence_points))
        
        with col2:
            st.metric("Pseudo-ausências", len(pseudo_absences))
        
        with col3:
            st.metric("Total de pontos", len(presence_points) + len(pseudo_absences))
        
        # Download options
        st.subheader("Download dos Dados")
        
        # Prepare data for download
        presence_df = pd.DataFrame(presence_points, columns=['latitude', 'longitude'])
        presence_df['type'] = 'presence'
        presence_df['value'] = 1
        
        absence_df = pd.DataFrame(pseudo_absences, columns=['latitude', 'longitude'])
        absence_df['type'] = 'absence'
        absence_df['value'] = 0
        
        combined_df = pd.concat([presence_df, absence_df], ignore_index=True)
        
        # Add species name to the dataframe
        combined_df['species'] = species_name
        
        # Download combined data
        csv_combined = combined_df.to_csv(index=False)
        st.download_button(
            label="Download Dados Completos (Presenças + Pseudo-ausências)",
            data=csv_combined,
            file_name=f"{species_name.replace(' ', '_')}_sdm_data.csv",
            mime="text/csv"
        )
        
        # Indicador de progresso
        if st.session_state.get('pseudo_absences'):
            st.success("""
            ✅ **Pseudo-ausências prontas!**
            
            Você pode prosseguir para a Análise Bioclimática.
            """)
    
    # Information section
    with st.expander("ℹ Sobre Pseudo-ausências em SDM"):
        st.markdown("""
        ### Geração de Pseudo-ausências - Estratégia de Buffer
        
        Esta página utiliza a estratégia de buffer para gerar pseudo-ausências:
        
        1. **Buffer**: Cria uma área ao redor dos pontos de presença
        2. **Exclusão**: Remove área muito próxima às presenças
        3. **Amostragem**: Gera pontos aleatórios na área resultante
        
        **Parâmetros:**
        - **Distância do buffer**: Define o raio máximo de busca
        - **Raio de exclusão**: Evita pseudo-ausências muito próximas
        - **Número**: Geralmente igual ao número de presenças
        
        **Validação automática:**
        - Pontos restritos ao território brasileiro
        - Evita sobreposição com áreas de presença
        
        **Uso dos dados:**
        - Download do arquivo CSV combinado
        - Dados prontos para modelagem (Random Forest, MaxEnt, etc.)
        - Coluna 'value': 1 para presença, 0 para ausência
        """)

if __name__ == "__main__":
    render_page()