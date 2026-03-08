import streamlit as st
import pandas as pd
from utils.gbif_api import search_species, get_countries, format_occurrence_for_map
from utils.geo_utils import create_occurrence_map, create_heatmap
from streamlit_folium import st_folium

@st.cache_data(show_spinner=False)
def cached_search_species(scientific_name: str, country: str, limit: int) -> dict:
    """
    Cached wrapper for GBIF species search to speed up repeated queries.
    """
    return search_species(scientific_name, country, limit)

def render_page():
    st.title("Busca de Espécies - GBIF")
    st.markdown("Pesquise ocorrências de espécies no Brasil na base de dados do GBIF")
    
    # Search form
    col1, col2 = st.columns([3, 1])
    
    with col1:
        species_name = st.text_input(
            "Nome científico da espécie",
            placeholder="Ex: Araucaria angustifolia",
            help="Digite o nome científico da espécie",
            value=st.session_state.get('species_name', '')
        )
    
    with col2:
        limit = st.number_input(
            "Limite de resultados",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Máximo de 500 ocorrências para manter o desempenho"
        )
    
    map_type = st.radio(
        "Tipo de visualização",
        ["Pontos", "Mapa de calor"],
        horizontal=True
    )
    
    # Search button - only search when button is clicked
    if st.button("Buscar", type="primary", use_container_width=True):
        if species_name:
            with st.spinner("Buscando ocorrências no Brasil..."):
                # Always search in Brazil (country code: BR)
                country_code = "BR"
                
                # Search species (cached)
                results = cached_search_species(species_name, country_code, limit)
                
                if "error" in results:
                    st.error(f"Erro na busca: {results['error']}")
                elif results.get("results"):
                    occurrences = results["results"]
                    st.success(f"Encontradas {len(occurrences)} ocorrências no Brasil")
                    
                    # Format data for map
                    map_data = [format_occurrence_for_map(occ) for occ in occurrences]
                    
                    # Store in session state for other modules
                    st.session_state['occurrence_data'] = map_data
                    st.session_state['n_occurrences'] = len(map_data)
                    st.session_state['species_name'] = species_name
                    st.session_state['original_occurrences'] = occurrences  # Keep original for reset
                else:
                    st.warning("Nenhuma ocorrência encontrada no Brasil")
        else:
            st.warning("Por favor, insira o nome científico da espécie")
    
    # Display results if we have data
    if st.session_state.get('occurrence_data'):
        map_data = st.session_state['occurrence_data']
        
        # Display metrics
        st.divider()
        
        # Display species name if available
        if 'species_name' in st.session_state:
            st.info(f"🌿 Espécie: **{st.session_state['species_name']}**")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Total de ocorrências", len(st.session_state.get('original_occurrences', [])))
        with metric_col2:
            st.metric("Ocorrências no Brasil", len(map_data))
        with metric_col3:
            current_count = len(map_data)
            original_count = len(st.session_state.get('original_occurrences', []))
            filtered_count = original_count - current_count
            st.metric("Ocorrências filtradas", filtered_count)
        
        # Add instruction for polygon drawing
        st.info("💡 **Dica**: Use as ferramentas no mapa para desenhar polígonos e remover ocorrências duvidosas.")
        
        # Create and display map
        if map_type == "Pontos":
            m = create_occurrence_map(map_data)
        else:
            m = create_heatmap(map_data)

        # Render map and capture drawn polygons for filtering
        folium_output = st_folium(
            m, 
            width=700, 
            height=500, 
            key="species_map",
            returned_objects=["all_draws", "last_active_drawing"],
            return_on_hover=False,
            use_container_width=False
        )
        
        # Process drawn polygons
        drawn = folium_output.get("all_draws", []) if folium_output else []
        
        # Alternative: check for last active drawing
        last_drawing = folium_output.get("last_active_drawing") if folium_output else None
        
        
        # Use either drawn polygons or last active drawing
        shapes_to_process = drawn if drawn else ([last_drawing] if last_drawing else [])
        
        if shapes_to_process:
            from shapely.geometry import shape, Point

            # Count points to be removed
            points_to_remove = set()
            for drawing in shapes_to_process:
                if drawing and drawing.get("geometry", {}).get("type") in ["Polygon", "Rectangle"]:
                    try:
                        polygon = shape(drawing["geometry"])
                        for i, occ in enumerate(map_data):
                            if polygon.contains(Point(occ["lon"], occ["lat"])):
                                points_to_remove.add(i)
                    except Exception as e:
                        st.error(f"Erro ao processar polígono: {e}")
            
            if points_to_remove:
                # Filter out the points
                filtered_data = [occ for i, occ in enumerate(map_data) 
                               if i not in points_to_remove]
                removed_count = len(points_to_remove)
                
                # Show removal confirmation
                st.divider()
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.warning(f"🔍 {removed_count} pontos estão dentro do(s) polígono(s) desenhado(s)")
                with col2:
                    if st.button("Remover pontos", type="primary", key="remove_points"):
                        st.session_state['occurrence_data'] = filtered_data
                        st.session_state['n_occurrences'] = len(filtered_data)
                        st.session_state['points_removed'] = removed_count
                        st.rerun()
                with col3:
                    if st.button("Limpar filtros", key="clear_filters"):
                        # Reset to original occurrences
                        original_occurrences = st.session_state.get('original_occurrences', [])
                        original_data = [format_occurrence_for_map(occ) for occ in original_occurrences]
                        st.session_state['occurrence_data'] = original_data
                        st.session_state['n_occurrences'] = len(original_data)
                        st.session_state.pop('points_removed', None)
                        st.rerun()
        
        # Show success message if points were just removed
        if st.session_state.get('points_removed'):
            st.success(f"✅ {st.session_state['points_removed']} pontos removidos com sucesso!")
            st.session_state.pop('points_removed')
        
        # Build DataFrame from occurrences and update session state
        df_data = []
        for occ in map_data:
            df_data.append({
                "scientificName": occ.get("scientific_name"),
                "country": "Brasil",
                "stateProvince": occ.get("state"),
                "year": occ.get("year"),
                "decimalLatitude": occ.get("lat"),
                "decimalLongitude": occ.get("lon"),
                "institutionCode": occ.get("institution"),
                "basisOfRecord": occ.get("basis_of_record")
            })
        df = pd.DataFrame(df_data)
        st.session_state['gbif_data'] = df
        
        # Use species name already stored from search, don't overwrite with GBIF record name
        species_for_session = st.session_state.get('species_name', 'Unknown Species')

        # Display data table
        with st.expander("Dados das ocorrências", expanded=False):
            display_df = df.rename(columns={
                "scientificName": "Nome Científico",
                "country": "País",
                "stateProvince": "Estado",
                "year": "Ano",
                "decimalLatitude": "Latitude",
                "decimalLongitude": "Longitude",
                "institutionCode": "Instituição",
                "basisOfRecord": "Base de Registro"
            })
            st.dataframe(display_df, use_container_width=True)

            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{species_for_session.replace(' ', '_')}_occurrences_BR.csv",
                mime="text/csv"
            )
            
        # Indicador de conclusão da etapa
        if st.session_state.get('species_name') and st.session_state.get('occurrence_data'):
            st.success("""
            ✅ **Etapa Concluída!**
            
            Você tem {} ocorrências da espécie {} prontas para análise.
            
            **Próximo passo:** Vá para o Módulo 2 - Pseudo-ausências
            """.format(
                st.session_state.get('n_occurrences', 0),
                st.session_state.get('species_name', '')
            ))
    
    # Information section
    with st.expander("ℹ Sobre esta página"):
        st.markdown("""
        ### Busca de Espécies no Brasil - GBIF
        
        Esta página permite buscar ocorrências de espécies no Brasil na base de dados do GBIF 
        (Global Biodiversity Information Facility).
        
        **Como usar:**
        1. Digite o nome científico da espécie
        2. Escolha o limite de resultados
        3. Selecione o tipo de visualização (pontos ou mapa de calor)
        4. Clique em "Buscar"
        5. Use as ferramentas de desenho no mapa para filtrar ocorrências duvidosas
        
        **Recursos:**
        - Busca automática restrita ao Brasil
        - Busca otimizada com paginação para grandes conjuntos de dados
        - Visualização em mapa interativo
        - **NOVO**: Filtragem de pontos por polígono
        - **NOVO**: Métricas de busca em tempo real
        - Tabela com dados detalhados
        - Download dos resultados em CSV
        - Dois tipos de visualização
        
        **Dicas para remoção de pontos:**
        - Clique no ícone de polígono no mapa
        - Desenhe ao redor dos pontos duvidosos
        - Clique em "Remover pontos" para confirmar
        - Use "Limpar filtros" para restaurar todos os pontos
        """)

if __name__ == "__main__":
    render_page()