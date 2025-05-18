import streamlit as st
import pandas as pd
from utils.gbif_api import search_species, get_countries, format_occurrence_for_map
from utils.geo_utils import create_occurrence_map, create_heatmap
from streamlit_folium import st_folium

def render_page():
    st.title("Busca de Espécies - GBIF")
    st.markdown("Pesquise ocorrências de espécies no Brasil na base de dados do GBIF")
    
    # Search form
    col1, col2 = st.columns([3, 1])
    
    with col1:
        species_name = st.text_input(
            "Nome científico da espécie",
            placeholder="Ex: Araucaria angustifolia",
            help="Digite o nome científico da espécie"
        )
    
    with col2:
        limit = st.number_input(
            "Limite de resultados",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )
    
    map_type = st.radio(
        "Tipo de visualização",
        ["Pontos", "Mapa de calor"],
        horizontal=True
    )
    
    # Search button
    if st.button("Buscar", type="primary", use_container_width=True):
        if species_name:
            with st.spinner("Buscando ocorrências no Brasil..."):
                # Always search in Brazil (country code: BR)
                country_code = "BR"
                
                # Search species
                results = search_species(species_name, country_code, limit)
                
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
                    
                    # Create and display map
                    if map_type == "Pontos":
                        m = create_occurrence_map(map_data)
                    else:
                        m = create_heatmap(map_data)
                    
                    st_folium(m, width=700, height=500, returned_objects=["all_draws"])
                    
                    # Display data table
                    with st.expander("Dados das ocorrências", expanded=False):
                        df_data = []
                        for occ in map_data:
                            df_data.append({
                                "Nome Científico": occ.get("scientific_name"),
                                "Estado": occ.get("state"),
                                "Ano": occ.get("year"),
                                "Latitude": occ.get("lat"),
                                "Longitude": occ.get("lon"),
                                "Instituição": occ.get("institution"),
                                "Base": occ.get("basis_of_record")
                            })
                        
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Download option
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"{species_name.replace(' ', '_')}_occurrences_BR.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("Nenhuma ocorrência encontrada no Brasil")
        else:
            st.warning("Por favor, insira o nome científico da espécie")
    
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
        
        **Recursos:**
        - Busca automática restrita ao Brasil
        - Visualização em mapa interativo
        - Tabela com dados detalhados
        - Download dos resultados em CSV
        - Dois tipos de visualização
        """)

if __name__ == "__main__":
    render_page()