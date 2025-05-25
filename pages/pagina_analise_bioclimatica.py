import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils.bioclim_analysis_optimized import BioclimAnalyzer
import warnings
# Suppress specific matplotlib warnings about font fallback
warnings.filterwarnings('ignore', message='Glyph.*missing from current font')

@st.cache_resource
def get_analyzer():
    """Cache the BioclimAnalyzer instance"""
    return BioclimAnalyzer()

@st.cache_data
def extract_bioclim_values(points_tuple, layers, analyzer_id):
    """Cache the extraction results"""
    # Convert tuple back to list for processing
    points = list(points_tuple)
    analyzer = get_analyzer()
    return analyzer.extract_values_at_points_optimized(points, layers)

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
    analyzer = get_analyzer()
    
    if not analyzer.available_layers:
        st.error("Nenhum layer bioclimático encontrado no diretório de dados.")
        st.write(f"Diretório verificado: {data_dir}")
        st.write(f"Diretório existe? {data_dir.exists()}")
        if data_dir.exists():
            st.write("Arquivos .tif encontrados:")
            for f in data_dir.glob('*.tif'):
                st.write(f"  - {f.name}")
        return
    
    # Check for available data
    has_occurrence_data = 'occurrence_data' in st.session_state
    has_pseudo_absence_data = 'pseudo_absence_data' in st.session_state
    
    if has_occurrence_data and has_pseudo_absence_data:
        st.markdown("### Dados Disponíveis")
        n_occurrences = len(st.session_state['occurrence_data'])
        n_pseudo_absences = len(st.session_state['pseudo_absence_data'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ocorrências GBIF", n_occurrences)
        with col2:
            st.metric("Pseudo-ausências", n_pseudo_absences)
        with col3:
            st.metric("Total de pontos", n_occurrences + n_pseudo_absences)
    elif has_occurrence_data:
        st.warning("⚠️ Apenas dados de ocorrência disponíveis. Gere pseudo-ausências no módulo anterior para uma análise completa.")
    elif has_pseudo_absence_data:
        st.warning("⚠️ Apenas dados de pseudo-ausências disponíveis. Faça uma busca GBIF para obter dados de ocorrência.")
    else:
        st.info("💡 Faça uma busca de espécies no módulo GBIF e gere pseudo-ausências para analisar variáveis bioclimáticas.")
    
    # Variable selection
    st.subheader("1. Seleção de Variáveis")
    
    # Display available layers
    available_vars = list(analyzer.available_layers.keys())
    available_vars.sort(key=lambda x: int(x.replace('bio', '')))
    
    st.info(f"📊 {len(available_vars)} variáveis bioclimáticas disponíveis. Todas estão selecionadas por padrão para análise completa.")
    
    # Debug info
    # st.write(f"Variáveis disponíveis: {available_vars}")
    
    selected_vars = st.multiselect(
        "Selecione as variáveis bioclimáticas para análise",
        options=available_vars,
        default=available_vars,  # Include all variables by default
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
    
    if (has_occurrence_data and has_pseudo_absence_data) and selected_vars:
        # Combine both datasets
        points = []
        point_types = []  # To track point type (presence/absence)
        
        # Add occurrence points
        occurrence_data = st.session_state['occurrence_data']
        occurrence_points = [(occ['lat'], occ['lon']) for occ in occurrence_data 
                  if occ.get('lat') and occ.get('lon')]
        points.extend(occurrence_points)
        point_types.extend(['presence'] * len(occurrence_points))
        
        # Add pseudo-absence points
        pseudo_absence_data = st.session_state['pseudo_absence_data']
        absence_points = [(row['decimalLatitude'], row['decimalLongitude']) 
                  for _, row in pseudo_absence_data.iterrows()]
        points.extend(absence_points)
        point_types.extend(['absence'] * len(absence_points))
        
        if st.button("Extrair Valores Bioclimáticos", type="primary"):
            import time
            start_time = time.time()
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Preparando extração...")
                
                # Use cached extraction
                # Convert points to tuple for caching
                points_tuple = tuple(points)
                df_analysis = extract_bioclim_values(points_tuple, selected_vars, id(analyzer))
                
                # Add point type column
                df_analysis['point_type'] = point_types
                st.session_state['bioclim_data'] = df_analysis
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                progress_bar.progress(1.0)
                status_text.text("Extração concluída!")
                
                st.success(f"✅ Valores extraídos para {len(points)} pontos em {elapsed_time:.2f} segundos!")
                
                # Show performance metrics
                points_per_second = len(points) / elapsed_time
                if points_per_second > 100:
                    st.info(f"💡 Performance: {points_per_second:.0f} pontos/segundo - Extração otimizada!")
                
                # Clear progress elements
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Erro na extração: {str(e)}")
    elif (has_occurrence_data or has_pseudo_absence_data) and selected_vars:
        st.warning("⚠️ São necessários tanto dados de ocorrência quanto pseudo-ausências para realizar a análise completa.")
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
            display_df = df_analysis.copy()
            # Move point_type column to the beginning if it exists
            if 'point_type' in display_df.columns:
                cols = ['point_type'] + [col for col in display_df.columns if col != 'point_type']
                display_df = display_df[cols]
            st.dataframe(display_df.head(10))
            st.write(f"Shape: {df_analysis.shape}")
            
            # Show distribution if we have both types
            if 'point_type' in df_analysis.columns and df_analysis['point_type'].nunique() > 1:
                st.write("Distribuição dos pontos:")
                point_counts = df_analysis['point_type'].value_counts()
                for ptype, count in point_counts.items():
                    st.write(f"- {ptype}: {count} pontos")
        
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
        
        # Selection method choice
        selection_method = st.radio(
            "Método de seleção:",
            ["Método Robusto (Recomendado)", "Método Stepwise Original"],
            index=0,
            help="O método robusto evita problemas comuns de multicolinearidade"
        )
        
        if selection_method == "Método Robusto (Recomendado)":
            # Use the new robust method
            selected_final, selection_info = analyzer.select_variables_robust(
                df_analysis,
                correlation_threshold=corr_threshold,
                vif_threshold=vif_threshold
            )
            
            # Show selection details
            with st.expander("Ver detalhes da seleção", expanded=True):
                # Summary by removal reason
                removal_summary = selection_info.groupby('reason_removed').size()
                st.write("**Resumo da seleção:**")
                for reason, count in removal_summary.items():
                    if reason != 'kept':
                        st.write(f"- Removidas por {reason}: {count} variáveis")
                    else:
                        st.write(f"- Mantidas: {count} variáveis")
                
                st.write("\n**Detalhes por variável:**")
                # Show detailed table
                display_info = selection_info[['variable', 'selected', 'reason_removed']].copy()
                display_info['selected'] = display_info['selected'].apply(lambda x: '✅' if x else '❌')
                st.dataframe(display_info, hide_index=True)
            
            steps = []  # For compatibility
        else:
            # Use original stepwise method
            selected_final, steps = analyzer.select_variables(
                df_analysis, 
                vif_threshold=vif_threshold,
                correlation_threshold=corr_threshold,
                return_steps=True
            )
        
        # Show stepwise elimination process only for original method
        if selection_method != "Método Robusto (Recomendado)" and steps:
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
            overlay_points = st.checkbox(
                "Sobrepor pontos",
                value=True,
                help="Mostrar pontos de ocorrência e pseudo-ausências sobre o layer"
            )
        
        # Additional visualization options
        if overlay_points:
            col1b, col2b, col3b = st.columns(3)
            with col1b:
                point_size = st.slider("Tamanho dos pontos", 10, 50, 20)
            with col2b:
                point_alpha = st.slider("Transparência", 0.1, 1.0, 0.7)
            with col3b:
                legend_position = st.selectbox("Posição da legenda", 
                                             ['upper right', 'upper left', 'lower right', 'lower left'],
                                             index=0)
        
        if st.button("Visualizar Layer", type="primary"):
                with st.spinner("Gerando visualização..."):
                    try:
                        fig_layer = analyzer.visualize_layer(selected_layer, cmap=cmap)
                        
                        # Add occurrence and pseudo-absence points if requested
                        if overlay_points and has_occurrence_data and has_pseudo_absence_data:
                            # Get current axes
                            ax = fig_layer.gca()
                            
                            # Get visualization options with defaults
                            ps = point_size if 'point_size' in locals() else 20
                            pa = point_alpha if 'point_alpha' in locals() else 0.7
                            lp = legend_position if 'legend_position' in locals() else 'upper right'
                            
                            # Plot occurrence points
                            occurrence_data = st.session_state['occurrence_data']
                            occ_lons = [occ['lon'] for occ in occurrence_data if occ.get('lon')]
                            occ_lats = [occ['lat'] for occ in occurrence_data if occ.get('lat')]
                            ax.scatter(occ_lons, occ_lats, c='blue', s=ps, alpha=pa, 
                                     edgecolors='white', linewidth=0.5, label='Ocorrências')
                            
                            # Plot pseudo-absence points
                            pseudo_absence_data = st.session_state['pseudo_absence_data']
                            abs_lons = pseudo_absence_data['decimalLongitude'].tolist()
                            abs_lats = pseudo_absence_data['decimalLatitude'].tolist()
                            ax.scatter(abs_lons, abs_lats, c='red', s=ps, alpha=pa,
                                     edgecolors='white', linewidth=0.5, label='Pseudo-ausências')
                            
                            # Add legend
                            ax.legend(loc=lp, framealpha=0.9, fontsize=8)
                            
                            # Update the plot
                            fig_layer.tight_layout()
                        
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
                cols_to_save = ['latitude', 'longitude']
                if 'point_type' in df_analysis.columns:
                    cols_to_save.append('point_type')
                cols_to_save.extend(selected_final)
                selected_data = df_analysis[cols_to_save]
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