import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils.bioclim_analysis import BioclimAnalyzer
import warnings
warnings.filterwarnings('ignore')

def render_page():
    st.title("Análise de Variáveis Bioclimáticas")
    st.markdown("Análise de correlação, VIF e seleção de variáveis WorldClim")
    
    # Check if we have data directory
    data_dir = Path('data/worldclim_brazil')
    if not data_dir.exists():
        st.error("⚠️ Dados WorldClim não encontrados!")
        st.info("""
        Para usar este módulo, execute primeiro o preprocessamento:
        ```bash
        python worldclim_preprocessor.py
        ```
        Isso baixará e processará os dados WorldClim para o Brasil.
        """)
        return
    
    # Initialize analyzer
    analyzer = BioclimAnalyzer(data_dir)
    
    if not analyzer.available_layers:
        st.error("Nenhum layer bioclimático encontrado no diretório de dados.")
        st.write(f"Diretório verificado: {data_dir}")
        st.write(f"Diretório existe? {data_dir.exists()}")
        if data_dir.exists():
            st.write("Arquivos .tif encontrados:")
            for f in data_dir.glob('*.tif'):
                st.write(f"  - {f.name}")
        return
    
    # Check for occurrence data
    use_occurrence_data = False
    if 'occurrence_data' in st.session_state:
        use_occurrence_data = st.checkbox(
            "Usar dados de ocorrência da busca GBIF",
            value=True,
            help="Analisa as variáveis bioclimáticas nos pontos de ocorrência"
        )
    else:
        st.info("💡 Faça uma busca de espécies no módulo GBIF primeiro para analisar variáveis bioclimáticas em pontos reais de ocorrência.")
    
    # Variable selection
    st.subheader("1. Seleção de Variáveis")
    
    # Display available layers
    available_vars = list(analyzer.available_layers.keys())
    available_vars.sort(key=lambda x: int(x.replace('bio', '')))
    
    # Debug info
    # st.write(f"Variáveis disponíveis: {available_vars}")
    
    selected_vars = st.multiselect(
        "Selecione as variáveis bioclimáticas para análise",
        options=available_vars,
        default=available_vars[:10] if len(available_vars) >= 10 else available_vars,
        format_func=lambda x: f"{x}: {analyzer.metadata['layers'].get(x, {}).get('name', x)}"
    )
    
    if not selected_vars:
        st.warning("Por favor, selecione pelo menos uma variável.")
        return
    
    # Analysis parameters
    st.subheader("2. Parâmetros de Análise")
    
    col1, col2 = st.columns(2)
    
    with col1:
        vif_threshold = st.number_input(
            "Threshold VIF",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="Variáveis com VIF > threshold são consideradas colineares"
        )
    
    with col2:
        corr_threshold = st.number_input(
            "Threshold de Correlação",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Pares com correlação > threshold são considerados redundantes"
        )
    
    # Data source for analysis
    st.subheader("3. Fonte de Dados")
    
    df_analysis = None
    
    if use_occurrence_data and 'occurrence_data' in st.session_state:
        # Extract values at occurrence points
        occurrence_data = st.session_state['occurrence_data']
        points = [(occ['lat'], occ['lon']) for occ in occurrence_data 
                  if occ.get('lat') and occ.get('lon')]
        
        st.success(f"✅ {len(points)} pontos de ocorrência disponíveis da busca GBIF")
        
        if st.button("Extrair Valores nos Pontos de Ocorrência", type="primary"):
            with st.spinner("Extraindo valores bioclimáticos..."):
                df_analysis = analyzer.extract_values_at_points(points, selected_vars)
                st.session_state['bioclim_data'] = df_analysis
                st.success(f"Valores extraídos para {len(points)} pontos!")
    else:
        # Generate sample data for demonstration
        st.warning("⚠️ Nenhum dado de ocorrência disponível. Use dados de exemplo para demonstração.")
        
        if st.button("Gerar Dados de Exemplo", type="primary"):
            with st.spinner("Gerando dados de exemplo..."):
                # Create synthetic correlated data
                n_samples = 1000
                data = {}
                
                # Generate correlated variables
                base = np.random.randn(n_samples)
                for i, var in enumerate(selected_vars):
                    if i == 0:
                        data[var] = base + np.random.randn(n_samples) * 0.5
                    elif i < 3:
                        data[var] = base * 0.8 + np.random.randn(n_samples) * 0.6
                    else:
                        data[var] = np.random.randn(n_samples)
                
                # Add coordinates
                data['latitude'] = np.random.uniform(-33, 5, n_samples)
                data['longitude'] = np.random.uniform(-73, -34, n_samples)
                
                df_analysis = pd.DataFrame(data)
                st.session_state['bioclim_data'] = df_analysis
                st.info("Dados de exemplo gerados para demonstração.")
    
    # Perform analysis if data is available
    if 'bioclim_data' in st.session_state:
        df_analysis = st.session_state['bioclim_data']
        
        # Display data preview
        with st.expander("Visualizar Dados", expanded=False):
            st.dataframe(df_analysis.head(10))
            st.write(f"Shape: {df_analysis.shape}")
        
        # Correlation Analysis
        st.subheader("4. Análise de Correlação")
        
        corr_matrix = analyzer.calculate_correlation_matrix(df_analysis)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_corr = analyzer.plot_correlation_matrix(corr_matrix)
            st.pyplot(fig_corr)
        
        with col2:
            st.markdown("### Correlações Altas")
            high_corr = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                        high_corr.append({
                            'Par': f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}",
                            'Correlação': f"{corr_matrix.iloc[i, j]:.3f}"
                        })
            
            if high_corr:
                high_corr_df = pd.DataFrame(high_corr)
                st.dataframe(high_corr_df, hide_index=True)
            else:
                st.info("Nenhuma correlação alta encontrada")
        
        # VIF Analysis
        st.subheader("5. Análise VIF (Variance Inflation Factor)")
        
        vif_results = analyzer.calculate_vif(df_analysis)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_vif = analyzer.plot_vif_bars(vif_results, vif_threshold)
            st.pyplot(fig_vif)
        
        with col2:
            # Add status column
            vif_results['Status'] = vif_results['VIF'].apply(
                lambda x: '❌ Colinear' if x > vif_threshold else '✅ OK'
            )
            
            # Display table
            display_cols = ['Variable', 'VIF', 'Status']
            st.dataframe(vif_results[display_cols], hide_index=True)
        
        # Variable Selection
        st.subheader("6. Seleção Automática de Variáveis")
        
        # Get steps of the selection process
        selected_final, steps = analyzer.select_variables(
            df_analysis, 
            vif_threshold=vif_threshold,
            correlation_threshold=corr_threshold,
            return_steps=True
        )
        
        # Show stepwise elimination process
        with st.expander("Ver processo de eliminação stepwise", expanded=True):
            for step in steps:
                if step['action'] == 'removed':
                    st.write(f"**Iteração {step['iteration']}**")
                    st.dataframe(step['vif_values'])
                    st.write(f"Removendo: **{step['removed_variable']}** (VIF = {step['removed_vif']:.2f})")
                    st.write("---")
                elif step['action'] == 'completed':
                    st.write(f"**Processo concluído na iteração {step['iteration']}**")
                    st.write("Todas as variáveis têm VIF abaixo do threshold")
                    st.dataframe(step['vif_values'])
                elif step['action'] == 'correlation_removal':
                    st.write(f"**Remoção por correlação alta**")
                    st.write(f"Removidas: {', '.join(step['removed_variables'])}")
                    st.write(f"Variáveis finais: {', '.join(step['final_variables'])}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Variáveis Selecionadas")
            if selected_final:
                for var in selected_final:
                    var_info = analyzer.metadata['layers'].get(var, {})
                    st.write(f"✅ **{var}**: {var_info.get('name', var)}")
            else:
                st.warning("Nenhuma variável passou pelos critérios de seleção!")
            
            # Store selected variables
            st.session_state['selected_bioclim_vars'] = selected_final
        
        with col2:
            st.markdown("### Estatísticas")
            st.metric("Total de variáveis", len(selected_vars))
            st.metric("Variáveis selecionadas", len(selected_final))
            st.metric("Variáveis removidas", len(selected_vars) - len(selected_final))
        
        # Visualization of layers
        st.subheader("7. Visualização de Layers")
        
        selected_layer = st.selectbox(
            "Selecione um layer para visualizar",
            options=selected_vars,
            format_func=lambda x: f"{x}: {analyzer.metadata['layers'].get(x, {}).get('name', x)}"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            cmap = st.selectbox(
                "Colormap",
                options=['viridis', 'plasma', 'coolwarm', 'RdYlBu_r', 'terrain'],
                index=2
            )
        
        with col2:
            if st.button("Visualizar Layer"):
                with st.spinner("Gerando visualização..."):
                    try:
                        fig_layer = analyzer.visualize_layer(selected_layer, cmap=cmap)
                        st.pyplot(fig_layer)
                    except Exception as e:
                        st.error(f"Erro ao visualizar layer: {e}")
        
        # Download results
        st.subheader("8. Download dos Resultados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_corr = corr_matrix.to_csv(index=True)
            st.download_button(
                label="Download Matriz de Correlação",
                data=csv_corr,
                file_name="correlation_matrix_bioclim.csv",
                mime="text/csv"
            )
        
        with col2:
            csv_vif = vif_results.to_csv(index=False)
            st.download_button(
                label="Download Análise VIF",
                data=csv_vif,
                file_name="vif_analysis_bioclim.csv",
                mime="text/csv"
            )
        
        with col3:
            if 'bioclim_data' in st.session_state:
                # Save only selected variables
                selected_data = df_analysis[['latitude', 'longitude'] + selected_final]
                csv_selected = selected_data.to_csv(index=False)
                st.download_button(
                    label="Download Dados Selecionados",
                    data=csv_selected,
                    file_name="selected_bioclim_data.csv",
                    mime="text/csv"
                )
    
    # Information section
    with st.expander("ℹ Sobre as Variáveis Bioclimáticas"):
        st.markdown("""
        ### Variáveis Bioclimáticas WorldClim
        
        As 19 variáveis bioclimáticas representam diferentes aspectos do clima:
        
        **Temperatura:**
        - BIO1-BIO11: Variáveis relacionadas à temperatura
        - Valores armazenados como °C × 10 (divididos por 10 durante o processamento)
        
        **Precipitação:**
        - BIO12-BIO19: Variáveis relacionadas à precipitação
        - Valores em milímetros (mm)
        
        **Importância para SDM:**
        - Essas variáveis são fundamentais para modelagem de distribuição de espécies
        - A seleção adequada evita multicolinearidade e melhora os modelos
        - Variables altamente correlacionadas podem causar problemas de convergência
        
        **Critérios de Seleção:**
        - VIF < 5: Evita multicolinearidade severa
        - |r| < 0.7: Remove variáveis altamente correlacionadas
        - Relevância ecológica para a espécie em estudo
        """)

if __name__ == "__main__":
    render_page()