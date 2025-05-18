import streamlit as st
import pandas as pd
from utils.gbif_api import search_species, get_countries, format_occurrence_for_map
from utils.geo_utils import create_occurrence_map, create_heatmap
from streamlit_folium import st_folium

def render_page():
    st.title("Busca de Esp�cies - GBIF")
    st.markdown("Pesquise ocorr�ncias de esp�cies na base de dados do GBIF")
    
    # Search form
    col1, col2 = st.columns([2, 1])
    
    with col1:
        species_name = st.text_input(
            "Nome cient�fico da esp�cie",
            placeholder="Ex: Panthera onca",
            help="Digite o nome cient�fico da esp�cie"
        )
    
    with col2:
        countries = get_countries()
        country_options = ["Todos"] + [f"{c['name']} ({c['code']})" for c in countries]
        selected_country = st.selectbox("Pa�s", country_options)
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        limit = st.number_input(
            "Limite de resultados",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )
    
    with col4:
        map_type = st.radio(
            "Tipo de visualiza��o",
            ["Pontos", "Mapa de calor"],
            horizontal=True
        )
    
    # Search button
    if st.button("Buscar", type="primary", use_container_width=True):
        if species_name:
            with st.spinner("Buscando ocorr�ncias..."):
                # Get country code if selected
                country_code = None
                if selected_country != "Todos":
                    for country in countries:
                        if country['name'] in selected_country:
                            country_code = country['code']
                            break
                
                # Search species
                results = search_species(species_name, country_code, limit)
                
                if "error" in results:
                    st.error(f"Erro na busca: {results['error']}")
                elif results.get("results"):
                    occurrences = results["results"]
                    st.success(f"Encontradas {len(occurrences)} ocorr�ncias")
                    
                    # Format data for map
                    map_data = [format_occurrence_for_map(occ) for occ in occurrences]
                    
                    # Create and display map
                    if map_type == "Pontos":
                        m = create_occurrence_map(map_data)
                    else:
                        m = create_heatmap(map_data)
                    
                    st_folium(m, width=700, height=500, returned_objects=["all_draws"])
                    
                    # Display data table
                    with st.expander("Dados das ocorr�ncias", expanded=False):
                        df_data = []
                        for occ in map_data:
                            df_data.append({
                                "Nome Cient�fico": occ.get("scientific_name"),
                                "Pa�s": occ.get("country"),
                                "Estado": occ.get("state"),
                                "Ano": occ.get("year"),
                                "Latitude": occ.get("lat"),
                                "Longitude": occ.get("lon"),
                                "Institui��o": occ.get("institution"),
                                "Base": occ.get("basis_of_record")
                            })
                        
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Download option
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"{species_name.replace(' ', '_')}_occurrences.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("Nenhuma ocorr�ncia encontrada")
        else:
            st.warning("Por favor, insira o nome cient�fico da esp�cie")
    
    # Information section
    with st.expander("9 Sobre esta p�gina"):
        st.markdown("""
        ### Busca de Esp�cies no GBIF
        
        Esta p�gina permite buscar ocorr�ncias de esp�cies na base de dados do GBIF 
        (Global Biodiversity Information Facility).
        
        **Como usar:**
        1. Digite o nome cient�fico da esp�cie
        2. Opcionalmente, selecione um pa�s para filtrar
        3. Escolha o limite de resultados
        4. Selecione o tipo de visualiza��o (pontos ou mapa de calor)
        5. Clique em "Buscar"
        
        **Recursos:**
        - Visualiza��o em mapa interativo
        - Tabela com dados detalhados
        - Download dos resultados em CSV
        - Filtro por pa�s
        - Dois tipos de visualiza��o
        """)

if __name__ == "__main__":
    render_page()