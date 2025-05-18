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
    
    # Configuration section
    st.subheader("1. Configuração da Análise")
    
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_type = st.selectbox(
            "Tipo de análise",
            ["Análise com dados de exemplo", "Upload de arquivo de ocorrências"]
        )
    
    with col2:
        vif_threshold = st.number_input(
            "Threshold VIF",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="Variáveis com VIF > threshold serão marcadas como colineares"
        )
    
    # WorldClim layers info
    layers_info = get_worldclim_layers()
    
    # Variable selection
    st.subheader("2. Seleção de Variáveis")
    
    selected_vars = st.multiselect(
        "Selecione as variáveis bioclimáticas",
        options=list(layers_info.keys()),
        default=list(layers_info.keys())[:10],
        format_func=lambda x: f"{x}: {layers_info[x]}"
    )
    
    if st.button("Analisar Variáveis", type="primary", use_container_width=True):
        if not selected_vars:
            st.warning("Por favor, selecione pelo menos uma variável")
        else:
            with st.spinner("Processando análise..."):
                # Generate sample data
                np.random.seed(42)
                n_samples = 1000
                
                # Create correlated variables
                data = {}
                base = np.random.randn(n_samples)
                
                for i, var in enumerate(selected_vars):
                    if i == 0:
                        data[var] = base + np.random.randn(n_samples) * 0.5
                    elif i < 3:
                        data[var] = base * 0.8 + np.random.randn(n_samples) * 0.6
                    else:
                        data[var] = np.random.randn(n_samples)
                
                df = pd.DataFrame(data)
                
                # Correlation Analysis
                st.subheader("3. Análise de Correlação")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    correlation_matrix = df.corr()
                    mask = np.triu(np.ones_like(correlation_matrix), k=1)
                    sns.heatmap(
                        correlation_matrix,
                        mask=mask,
                        annot=True,
                        fmt='.2f',
                        cmap='coolwarm',
                        center=0,
                        vmin=-1,
                        vmax=1,
                        ax=ax
                    )
                    plt.title("Matriz de Correlação")
                    st.pyplot(fig)
                
                with col2:
                    st.markdown("### Correlações Altas")
                    threshold = 0.7
                    high_corr = []
                    
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(i+1, len(correlation_matrix.columns)):
                            if abs(correlation_matrix.iloc[i, j]) > threshold:
                                high_corr.append({
                                    'Par': f"{correlation_matrix.columns[i]} - {correlation_matrix.columns[j]}",
                                    'Correlação': correlation_matrix.iloc[i, j]
                                })
                    
                    if high_corr:
                        high_corr_df = pd.DataFrame(high_corr)
                        st.dataframe(high_corr_df, hide_index=True)
                    else:
                        st.info("Nenhuma correlação alta encontrada")
                
                # VIF Analysis
                st.subheader("4. Análise VIF (Variance Inflation Factor)")
                
                vif_results = calculate_vif(df)
                vif_results['Status'] = vif_results['VIF'].apply(
                    lambda x: '❌ Colinear' if x > vif_threshold else '✅ OK'
                )
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['red' if vif > vif_threshold else 'green' for vif in vif_results['VIF']]
                    bars = ax.barh(vif_results['Variable'], vif_results['VIF'], color=colors)
                    ax.axvline(x=vif_threshold, color='black', linestyle='--', label=f'Threshold ({vif_threshold})')
                    ax.set_xlabel('VIF Value')
                    ax.set_title('VIF por Variável')
                    ax.legend()
                    
                    # Add value labels on bars
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width, bar.get_y() + bar.get_height()/2, 
                               f'{width:.2f}', 
                               ha='left' if width < vif_threshold else 'right',
                               va='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    st.dataframe(vif_results, hide_index=True)
                
                # Summary and Recommendations
                st.subheader("5. Resumo e Recomendações")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Estatísticas")
                    n_vars = len(selected_vars)
                    n_colinear = len(vif_results[vif_results['VIF'] > vif_threshold])
                    n_high_corr = len(high_corr)
                    
                    st.metric("Total de variáveis", n_vars)
                    st.metric("Variáveis colineares (VIF)", n_colinear)
                    st.metric("Pares com alta correlação", n_high_corr)
                
                with col2:
                    st.markdown("### Variáveis Recomendadas")
                    recommended_vars = vif_results[vif_results['VIF'] <= vif_threshold]['Variable'].tolist()
                    
                    if recommended_vars:
                        for var in recommended_vars:
                            st.write(f"✅ {var}")
                    else:
                        st.warning("Todas as variáveis apresentam alta colinearidade")
                
                # Download results
                st.subheader("6. Download dos Resultados")
                
                results_data = {
                    'correlation_matrix': correlation_matrix,
                    'vif_results': vif_results,
                    'recommended_variables': recommended_vars
                }
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv_corr = correlation_matrix.to_csv(index=True)
                    st.download_button(
                        label="Download Matriz de Correlação",
                        data=csv_corr,
                        file_name="correlation_matrix.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    csv_vif = vif_results.to_csv(index=False)
                    st.download_button(
                        label="Download Análise VIF",
                        data=csv_vif,
                        file_name="vif_analysis.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    summary_text = f"""
                    Análise de Variáveis Ambientais - Resumo
                    
                    Total de variáveis analisadas: {n_vars}
                    Variáveis com VIF > {vif_threshold}: {n_colinear}
                    Pares com correlação > 0.7: {n_high_corr}
                    
                    Variáveis recomendadas:
                    {', '.join(recommended_vars)}
                    """
                    
                    st.download_button(
                        label="Download Resumo",
                        data=summary_text,
                        file_name="analysis_summary.txt",
                        mime="text/plain"
                    )
    
    # Information section
    with st.expander("ℹ Sobre esta análise"):
        st.markdown("""
        ### Análise de Variáveis Ambientais
        
        Esta página permite analisar a correlação e colinearidade entre variáveis 
        bioclimáticas do WorldClim.
        
        **Métricas utilizadas:**
        
        1. **Correlação de Pearson**: Mede a relação linear entre duas variáveis
           - Valores próximos a 1 ou -1 indicam forte correlação
           - Valores próximos a 0 indicam ausência de correlação linear
        
        2. **VIF (Variance Inflation Factor)**: Detecta multicolinearidade
           - VIF = 1: Sem correlação com outras variáveis
           - VIF > 5: Indica potencial problema de colinearidade
           - VIF > 10: Forte indicação de multicolinearidade
        
        **Recomendações:**
        - Evite usar variáveis com alta correlação (>0.7) no mesmo modelo
        - Remova variáveis com VIF > 5 para evitar problemas de colinearidade
        - Mantenha variáveis que sejam ecologicamente relevantes para sua análise
        """)

if __name__ == "__main__":
    render_page()