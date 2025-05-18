import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from utils.geo_utils import get_worldclim_layers, extract_raster_values
import warnings
warnings.filterwarnings('ignore')

def calculate_vif(df):
    """Calculate VIF for each variable in the dataframe"""
    vif_data = pd.DataFrame()
    vif_data["Variable"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    return vif_data

def render_page():
    st.title("Análise de Variáveis Ambientais")
    st.markdown("Análise de correlação e seleção de variáveis bioclimáticas")
    
    # Get bioclimatic variables
    variables = get_worldclim_layers()
    
    # Variable selection
    st.subheader("1. Seleção de Variáveis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_vars = st.multiselect(
            "Selecione as variáveis bioclimáticas",
            options=[f"{v['code']} - {v['name']}" for v in variables],
            default=[f"{v['code']} - {v['name']}" for v in variables[:5]]
        )
    
    with col2:
        analysis_type = st.radio(
            "Tipo de análise",
            ["Correlação", "VIF", "Ambos"]
        )
    
    # Data source
    st.subheader("2. Fonte de Dados")
    
    data_source = st.radio(
        "Escolha a fonte de dados",
        ["Dados simulados (MVP)", "Upload de arquivo CSV", "Extrair de rasters"]
    )
    
    df = None
    
    if data_source == "Dados simulados (MVP)":
        if st.button("Gerar dados simulados", type="primary"):
            with st.spinner("Gerando dados..."):
                # Generate simulated data
                n_points = 100
                selected_codes = [v.split(" - ")[0] for v in selected_vars]
                
                data = {}
                for code in selected_codes:
                    if "bio1" in code:  # Temperature-based
                        data[code] = np.random.normal(20, 5, n_points)
                    elif "bio12" in code:  # Precipitation-based
                        data[code] = np.random.normal(1200, 300, n_points)
                    else:
                        data[code] = np.random.normal(50, 20, n_points)
                
                # Add some correlations
                if len(selected_codes) > 1:
                    for i in range(1, len(selected_codes)):
                        correlation = np.random.uniform(0.3, 0.8)
                        data[selected_codes[i]] += correlation * data[selected_codes[0]]
                
                df = pd.DataFrame(data)
                st.session_state['env_data'] = df
    
    elif data_source == "Upload de arquivo CSV":
        uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state['env_data'] = df
    
    elif data_source == "Extrair de rasters":
        st.info("Funcionalidade de extração de rasters será implementada na próxima fase")
        
        # Placeholder for raster extraction
        col1, col2 = st.columns(2)
        with col1:
            points_file = st.file_uploader("Upload de pontos (CSV)", type="csv")
        with col2:
            raster_folder = st.text_input("Caminho para rasters", placeholder="/path/to/rasters")
        
        if st.button("Extrair valores"):
            st.warning("Esta funcionalidade está em desenvolvimento")
    
    # Analysis section
    if 'env_data' in st.session_state and st.session_state['env_data'] is not None:
        df = st.session_state['env_data']
        
        st.subheader("3. Análise de Correlação")
        
        # Filter dataframe for selected variables
        selected_codes = [v.split(" - ")[0] for v in selected_vars]
        df_filtered = df[[col for col in df.columns if col in selected_codes]]
        
        if len(df_filtered.columns) > 1:
            # Correlation analysis
            if analysis_type in ["Correlação", "Ambos"]:
                st.markdown("#### Matriz de Correlação")
                
                corr = df_filtered.corr()
                
                # Correlation heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                           square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
                ax.set_title("Matriz de Correlação das Variáveis Bioclimáticas")
                plt.tight_layout()
                st.pyplot(fig)
                
                # Correlation table
                with st.expander("Tabela de Correlação"):
                    st.dataframe(corr, use_container_width=True)
            
            # VIF analysis
            if analysis_type in ["VIF", "Ambos"]:
                st.markdown("#### Análise de VIF (Variance Inflation Factor)")
                
                try:
                    vif_data = calculate_vif(df_filtered)
                    
                    # VIF bar plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(vif_data['Variable'], vif_data['VIF'])
                    
                    # Color bars based on VIF value
                    for i, bar in enumerate(bars):
                        if vif_data['VIF'].iloc[i] > 10:
                            bar.set_color('red')
                        elif vif_data['VIF'].iloc[i] > 5:
                            bar.set_color('orange')
                        else:
                            bar.set_color('green')
                    
                    ax.axhline(y=5, color='orange', linestyle='--', label='VIF = 5')
                    ax.axhline(y=10, color='red', linestyle='--', label='VIF = 10')
                    ax.set_xlabel('Variáveis')
                    ax.set_ylabel('VIF')
                    ax.set_title('Variance Inflation Factor por Variável')
                    ax.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # VIF table
                    with st.expander("Tabela de VIF"):
                        st.dataframe(vif_data, use_container_width=True)
                    
                    # Recommendations
                    st.markdown("#### Recomendações")
                    high_vif = vif_data[vif_data['VIF'] > 10]
                    if not high_vif.empty:
                        st.warning(f"  Variáveis com VIF > 10 (alta multicolinearidade): {', '.join(high_vif['Variable'].tolist())}")
                    
                    medium_vif = vif_data[(vif_data['VIF'] > 5) & (vif_data['VIF'] <= 10)]
                    if not medium_vif.empty:
                        st.info(f"9 Variáveis com VIF entre 5-10 (multicolinearidade moderada): {', '.join(medium_vif['Variable'].tolist())}")
                    
                    low_vif = vif_data[vif_data['VIF'] <= 5]
                    if not low_vif.empty:
                        st.success(f" Variáveis com VIF < 5 (baixa multicolinearidade): {', '.join(low_vif['Variable'].tolist())}")
                
                except Exception as e:
                    st.error(f"Erro ao calcular VIF: {str(e)}")
        
        else:
            st.warning("Selecione pelo menos 2 variáveis para análise de correlação")
    
    # Information section
    with st.expander("9 Sobre esta página"):
        st.markdown("""
        ### Análise de Variáveis Ambientais
        
        Esta página permite analisar a correlação entre variáveis bioclimáticas e 
        identificar problemas de multicolinearidade.
        
        **Funcionalidades:**
        - Seleção de 19 variáveis bioclimáticas WorldClim
        - Análise de correlação com matriz e heatmap
        - Cálculo de VIF (Variance Inflation Factor)
        - Recomendações para seleção de variáveis
        
        **Interpretação do VIF:**
        - VIF < 5: Baixa multicolinearidade (OK)
        - VIF 5-10: Multicolinearidade moderada (cautela)
        - VIF > 10: Alta multicolinearidade (considerar remover)
        
        **Próximos passos:**
        - Integração com rasters WorldClim reais
        - Extração automática de valores
        - Análise espacial avançada
        """)

if __name__ == "__main__":
    render_page()