import streamlit as st
import pandas as pd
import numpy as np
import rasterio
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import tempfile
import os
from datetime import datetime

def render_page():
    st.title("🌡️ Projeção Futura - Mudanças Climáticas")
    st.markdown("Analise como as mudanças climáticas podem afetar a distribuição da espécie")
    
    # Display species info
    if 'species_name' in st.session_state:
        st.info(f"🌿 Espécie: **{st.session_state['species_name']}**")
    else:
        st.warning("⚠️ Nenhuma espécie selecionada. Por favor, comece pela busca de espécies.")
        return
    
    # Check if model is trained
    if not st.session_state.get('model_trained'):
        st.warning("⚠️ Treine um modelo na aba de Modelagem primeiro.")
        return
    
    model = st.session_state['trained_model']
    selected_vars = st.session_state['selected_vars']
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Configurações de Projeção Futura")
        
        st.info("""
        📊 **Configuração pedagógica otimizada para Mata Atlântica:**
        
        • **Modelo climático**: MPI-ESM1-2-HR
        • **Cenários**: SSP1-2.6 (otimista) e SSP5-8.5 (pessimista)
        • **Período**: 2081-2100
        """)
        
        st.warning("""
        ⚠️ **Nota Importante sobre Ensemble de Modelos**
        
        A recomendação científica é trabalhar com um ensemble (conjunto) de 
        múltiplos modelos climáticos para capturar a incerteza nas projeções. 
        
        No TAIPA, para fins didáticos, utilizamos apenas um modelo (MPI-ESM1-2-HR) 
        para simplificar o processo de aprendizagem.
        
        Em aplicações científicas reais, sempre use múltiplos GCMs.
        """)
        
        # Scenario selection
        scenario = st.selectbox(
            "Cenário climático",
            ["SSP1-2.6 (Otimista)", "SSP5-8.5 (Pessimista)"],
            help="SSP1-2.6: Desenvolvimento sustentável com forte mitigação\nSSP5-8.5: Uso intensivo de combustíveis fósseis"
        )
        
        # Period selection - now fixed to 2081-2100
        period = "2081-2100 (Futuro distante)"
        st.info("📅 Período fixado em 2081-2100 para análise de impactos de longo prazo")
        
        # Display options
        st.subheader("Opções de Visualização")
        
        # Threshold selection
        st.subheader("Threshold para Mapa Binário")
        threshold_method = st.selectbox(
            "Método de threshold",
            ["Manual", "Usar do mapa atual", "Média das Predições", "Percentil 50", "Percentil 10"]
        )
        
        if threshold_method == "Manual":
            threshold = st.slider("Threshold manual", 0.0, 1.0, 0.5, step=0.01)
        elif threshold_method == "Usar do mapa atual":
            if 'projection_threshold' in st.session_state:
                threshold = st.session_state['projection_threshold']
                st.info(f"Usando threshold do mapa atual: {threshold:.3f}")
            else:
                threshold = 0.5
                st.warning("Threshold do mapa atual não encontrado. Usando 0.5")
        else:
            threshold = None  # Will be calculated based on data
        
        show_probability_maps = st.checkbox("Mostrar mapas de probabilidade", value=False)
        show_change_map = st.checkbox("Mostrar mapa de mudanças", value=True)
        show_metrics = st.checkbox("Mostrar métricas de mudança", value=True)
    
    # Extract scenario and period codes
    scenario_code = "ssp126" if "SSP1-2.6" in scenario else "ssp585"
    period_code = "2081-2100"  # Fixed period
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Projeção de Distribuição Futura")
        
        if st.button("Gerar Projeção Futura", type="primary"):
            with st.spinner("Preparando dados climáticos futuros..."):
                try:
                    # Future climate data path
                    future_climate_path = Path(f"data/worldclim_future/{scenario_code}_{period_code}")
                    
                    # Check if future data exists
                    if not future_climate_path.exists():
                        st.error(f"Dados climáticos futuros não encontrados em: {future_climate_path}")
                        st.info("💡 Execute o script de download de dados futuros primeiro.")
                        return
                    
                    # Load current prediction for comparison
                    current_prediction = st.session_state.get('last_prediction')
                    if current_prediction is None:
                        st.warning("Execute uma projeção espacial atual primeiro para comparação.")
                        return
                    
                    # Load future climate data
                    st.info("Carregando dados climáticos futuros...")
                    
                    # Get spatial reference from first file
                    first_var = selected_vars[0]
                    var_num = int(first_var.replace('bio', ''))
                    ref_file = future_climate_path / f"wc2.1_2.5m_bioc_MPI-ESM1-2-HR_{scenario_code}_{period_code}_bio{var_num}.tif"
                    
                    if not ref_file.exists():
                        st.error(f"Arquivo de referência não encontrado: {ref_file}")
                        return
                    
                    with rasterio.open(ref_file) as src:
                        crs = src.crs
                        transform = src.transform
                        height = src.height
                        width = src.width
                        bounds = src.bounds
                    
                    # Create arrays for future climate data
                    n_vars = len(selected_vars)
                    bio_data_future = np.zeros((n_vars, height, width))
                    
                    # Load selected bioclimatic variables
                    progress_bar = st.progress(0)
                    for i, var in enumerate(selected_vars):
                        var_num = int(var.replace('bio', ''))
                        tif_path = future_climate_path / f"wc2.1_2.5m_bioc_MPI-ESM1-2-HR_{scenario_code}_{period_code}_bio{var_num}.tif"
                        
                        if not tif_path.exists():
                            st.error(f"Arquivo não encontrado: {tif_path}")
                            continue
                        
                        with rasterio.open(tif_path) as src:
                            data = src.read(1)
                            
                            # Apply temperature conversion if needed
                            if var_num in [1,2,3,4,5,6,7,8,9,10,11]:
                                mask = data != -9999
                                data[mask] = data[mask] / 10.0
                            
                            bio_data_future[i] = data
                        
                        progress_bar.progress((i + 1) / n_vars)
                    
                    # Reshape for prediction
                    X_future = bio_data_future.reshape(n_vars, -1).T
                    
                    # Remove NoData
                    valid_mask = ~np.any(X_future == -9999, axis=1)
                    X_valid = X_future[valid_mask]
                    
                    # Make predictions
                    st.info("Gerando predições para clima futuro...")
                    predictions_future = model.predict_proba(X_valid)[:, 1]
                    
                    # Create prediction map
                    prediction_map_future = np.full(X_future.shape[0], np.nan)
                    prediction_map_future[valid_mask] = predictions_future
                    prediction_map_future = prediction_map_future.reshape(height, width)
                    
                    # Calculate threshold if needed
                    if threshold is None:
                        valid_probs = prediction_map_future[~np.isnan(prediction_map_future)]
                        
                        if threshold_method == "Média das Predições":
                            threshold = np.mean(valid_probs)
                        elif threshold_method == "Percentil 50":
                            threshold = np.percentile(valid_probs, 50)
                        elif threshold_method == "Percentil 10":
                            threshold = np.percentile(valid_probs, 10)
                        else:
                            threshold = 0.5
                    
                    # Create binary maps
                    binary_map_future = np.full_like(prediction_map_future, np.nan)
                    valid_pixels = ~np.isnan(prediction_map_future)
                    binary_map_future[valid_pixels] = (prediction_map_future[valid_pixels] >= threshold).astype(float)
                    
                    # Get current binary map
                    if 'binary_map' in st.session_state:
                        binary_map_current = st.session_state['binary_map']
                    else:
                        st.error("Mapa binário atual não encontrado. Execute uma projeção espacial primeiro.")
                        return
                    
                    # Create visualizations
                    tabs = st.tabs(["Comparação", "Mudanças", "Métricas", "Exportar"])
                    
                    with tabs[0]:
                        st.subheader("Comparação: Presente vs Futuro")
                        
                        # Binary maps comparison
                        st.markdown("### Mapas Binários (Presença/Ausência)")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Distribuição Atual")
                            fig_current_binary = go.Figure(data=go.Heatmap(
                                z=binary_map_current[::-1],
                                colorscale=[[0, 'white'], [1, 'darkgreen']],
                                showscale=True,
                                colorbar=dict(title="Presença", tickvals=[0, 1], ticktext=['Ausente', 'Presente'])
                            ))
                            fig_current_binary.update_layout(
                                title=f"Presente (threshold: {st.session_state.get('projection_threshold', threshold):.3f})",
                                xaxis_title="Longitude",
                                yaxis_title="Latitude",
                                height=400
                            )
                            st.plotly_chart(fig_current_binary, use_container_width=True)
                        
                        with col2:
                            st.markdown(f"#### Distribuição Futura ({period_code})")
                            fig_future_binary = go.Figure(data=go.Heatmap(
                                z=binary_map_future[::-1],
                                colorscale=[[0, 'white'], [1, 'darkgreen']],
                                showscale=True,
                                colorbar=dict(title="Presença", tickvals=[0, 1], ticktext=['Ausente', 'Presente'])
                            ))
                            fig_future_binary.update_layout(
                                title=f"{scenario} - {period} (threshold: {threshold:.3f})",
                                xaxis_title="Longitude",
                                yaxis_title="Latitude",
                                height=400
                            )
                            st.plotly_chart(fig_future_binary, use_container_width=True)
                        
                        # Probability maps (optional)
                        if show_probability_maps:
                            st.markdown("### Mapas de Probabilidade")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### Probabilidade Atual")
                                fig_current_prob = go.Figure(data=go.Heatmap(
                                    z=current_prediction['map'][::-1],
                                    colorscale='Viridis',
                                    showscale=True,
                                    colorbar=dict(title="Probabilidade")
                                ))
                                fig_current_prob.update_layout(
                                    title="Presente",
                                    xaxis_title="Longitude",
                                    yaxis_title="Latitude",
                                    height=400
                                )
                                st.plotly_chart(fig_current_prob, use_container_width=True)
                            
                            with col2:
                                st.markdown(f"#### Probabilidade Futura ({period_code})")
                                fig_future_prob = go.Figure(data=go.Heatmap(
                                    z=prediction_map_future[::-1],
                                    colorscale='Viridis',
                                    showscale=True,
                                    colorbar=dict(title="Probabilidade")
                                ))
                                fig_future_prob.update_layout(
                                    title=f"{scenario} - {period}",
                                    xaxis_title="Longitude",
                                    yaxis_title="Latitude",
                                    height=400
                                )
                                st.plotly_chart(fig_future_prob, use_container_width=True)
                    
                    with tabs[1]:
                        if show_change_map:
                            st.subheader("Mapa de Mudanças")
                            
                            # Calculate change map for binary data
                            # -1: loss, 0: no change, 1: gain
                            change_map = binary_map_future - binary_map_current
                            
                            # Create custom colorscale for change map
                            colors = ['red', 'lightgray', 'green']
                            colorscale = [
                                [0.0, colors[0]],  # Loss (red)
                                [0.5, colors[1]],  # No change (gray)
                                [1.0, colors[2]]   # Gain (green)
                            ]
                            
                            fig_change = go.Figure(data=go.Heatmap(
                                z=change_map[::-1],
                                colorscale=colorscale,
                                zmid=0,
                                zmin=-1,
                                zmax=1,
                                showscale=True,
                                colorbar=dict(
                                    title="Mudança",
                                    tickvals=[-1, 0, 1],
                                    ticktext=['Perda', 'Sem mudança', 'Ganho']
                                )
                            ))
                            fig_change.update_layout(
                                title=f"Mudança na Distribuição (Futuro - Presente) - Threshold: {threshold:.3f}",
                                xaxis_title="Longitude",
                                yaxis_title="Latitude",
                                height=500
                            )
                            st.plotly_chart(fig_change, use_container_width=True)
                            
                            st.info("""
                            🔴 **Vermelho**: Áreas com perda de habitat adequado
                            🟢 **Verde**: Áreas com ganho de habitat adequado
                            ⚪ **Cinza**: Áreas sem mudança
                            """)
                    
                    with tabs[2]:
                        if show_metrics:
                            st.subheader("Métricas de Mudança")
                            
                            st.info(f"Threshold utilizado: {threshold:.3f}")
                            
                            # Calculate metrics from binary maps
                            current_suitable = np.nansum(binary_map_current)
                            future_suitable = np.nansum(binary_map_future)
                            
                            change_percent = ((future_suitable - current_suitable) / current_suitable) * 100 if current_suitable > 0 else 0
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Área Adequada Atual",
                                    f"{current_suitable:,} pixels",
                                    delta=None
                                )
                            
                            with col2:
                                st.metric(
                                    "Área Adequada Futura",
                                    f"{future_suitable:,} pixels",
                                    delta=f"{change_percent:.1f}%"
                                )
                            
                            with col3:
                                st.metric(
                                    "Mudança Absoluta",
                                    f"{future_suitable - current_suitable:,} pixels",
                                    delta=None
                                )
                            
                            # Area calculation (approximate)
                            pixel_area_km2 = 25  # ~5km resolution
                            current_area_km2 = current_suitable * pixel_area_km2
                            future_area_km2 = future_suitable * pixel_area_km2
                            
                            st.markdown("### Estimativa de Área")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric(
                                    "Área Atual",
                                    f"{current_area_km2:,.0f} km²"
                                )
                            
                            with col2:
                                st.metric(
                                    "Área Futura",
                                    f"{future_area_km2:,.0f} km²",
                                    delta=f"{change_percent:.1f}%"
                                )
                            
                            # Summary statistics
                            st.markdown("### Estatísticas de Mudança")
                            
                            # Areas of gain and loss from binary maps
                            valid_pixels = ~np.isnan(binary_map_current) & ~np.isnan(binary_map_future)
                            
                            gain_mask = (binary_map_future == 1) & (binary_map_current == 0) & valid_pixels
                            loss_mask = (binary_map_future == 0) & (binary_map_current == 1) & valid_pixels
                            stable_present_mask = (binary_map_future == 1) & (binary_map_current == 1) & valid_pixels
                            stable_absent_mask = (binary_map_future == 0) & (binary_map_current == 0) & valid_pixels
                            
                            gain_area = gain_mask.sum() * pixel_area_km2
                            loss_area = loss_mask.sum() * pixel_area_km2
                            stable_present_area = stable_present_mask.sum() * pixel_area_km2
                            stable_absent_area = stable_absent_mask.sum() * pixel_area_km2
                            
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=['Ganho', 'Perda', 'Habitat Estável', 'Sem Habitat Estável'],
                                values=[gain_area, loss_area, stable_present_area, stable_absent_area],
                                hole=.3,
                                marker_colors=['green', 'red', 'darkgreen', 'lightgray']
                            )])
                            fig_pie.update_layout(title="Distribuição de Mudanças")
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                            # Additional metrics
                            st.markdown("### Resumo Detalhado")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Ganho de Habitat", f"{gain_area:,.0f} km²")
                                st.metric("Habitat Estável", f"{stable_present_area:,.0f} km²")
                            
                            with col2:
                                st.metric("Perda de Habitat", f"{loss_area:,.0f} km²")
                                st.metric("Mudança Líquida", f"{(gain_area - loss_area):,.0f} km²")
                    
                    with tabs[3]:
                        st.subheader("Exportar Resultados")
                        
                        # Save future prediction
                        st.session_state['future_prediction'] = {
                            'probability_map': prediction_map_future,
                            'binary_map': binary_map_future,
                            'threshold': threshold,
                            'scenario': scenario,
                            'period': period,
                            'bounds': bounds,
                            'crs': crs,
                            'transform': transform
                        }
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("Exportar Mapa Binário Futuro (GeoTIFF)"):
                                export_path = Path(f"exports/future_binary_{scenario_code}_{period_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif")
                                export_path.parent.mkdir(exist_ok=True)
                                
                                with rasterio.open(
                                    export_path,
                                    'w',
                                    driver='GTiff',
                                    height=height,
                                    width=width,
                                    count=1,
                                    dtype=binary_map_future.dtype,
                                    crs=crs,
                                    transform=transform
                                ) as dst:
                                    dst.write(binary_map_future, 1)
                                
                                st.success(f"Mapa binário exportado: {export_path}")
                            
                            if st.button("Exportar Mapa de Probabilidade Futuro (GeoTIFF)"):
                                export_path = Path(f"exports/future_probability_{scenario_code}_{period_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif")
                                export_path.parent.mkdir(exist_ok=True)
                                
                                with rasterio.open(
                                    export_path,
                                    'w',
                                    driver='GTiff',
                                    height=height,
                                    width=width,
                                    count=1,
                                    dtype=prediction_map_future.dtype,
                                    crs=crs,
                                    transform=transform
                                ) as dst:
                                    dst.write(prediction_map_future, 1)
                                
                                st.success(f"Mapa de probabilidade exportado: {export_path}")
                        
                        with col2:
                            if st.button("Exportar Mapa de Mudanças (GeoTIFF)"):
                                export_path = Path(f"exports/change_{scenario_code}_{period_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif")
                                export_path.parent.mkdir(exist_ok=True)
                                
                                with rasterio.open(
                                    export_path,
                                    'w',
                                    driver='GTiff',
                                    height=height,
                                    width=width,
                                    count=1,
                                    dtype=change_map.dtype,
                                    crs=crs,
                                    transform=transform
                                ) as dst:
                                    dst.write(change_map, 1)
                                
                                st.success(f"Mapa de mudanças exportado: {export_path}")
                
                except Exception as e:
                    st.error(f"Erro ao gerar projeção futura: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    with col2:
        st.markdown("### ℹ️ Sobre esta análise")
        st.info("""
        Esta análise projeta a distribuição futura da espécie considerando:
        
        • **Mudanças climáticas** baseadas em modelos globais
        • **Cenários socioeconômicos** (SSPs)
        • **Períodos temporais** específicos
        
        **Limitações:**
        • Assume que as relações espécie-ambiente permanecem constantes
        • Não considera dispersão ou barreiras geográficas
        • Incerteza inerente aos modelos climáticos
        """)
        
        st.markdown("### 📚 Conceitos Importantes")
        
        with st.expander("O que são SSPs?"):
            st.markdown("""
            **Shared Socioeconomic Pathways (SSPs)** são cenários que descrevem 
            futuros alternativos de desenvolvimento socioeconômico:
            
            • **SSP2-4.5**: Caminho intermediário com algumas políticas climáticas
            • **SSP5-8.5**: Desenvolvimento baseado em combustíveis fósseis
            """)
        
        with st.expander("Por que MPI-ESM1-2-HR?"):
            st.markdown("""
            **MPI-ESM1-2-HR** (Max Planck Institute Earth System Model) é um 
            modelo climático de alta resolução desenvolvido na Alemanha que:
            
            • Excelente resolução espacial (HR = High Resolution)
            • Ótima representação de processos atmosféricos tropicais
            • Validado extensivamente para América do Sul
            • Um dos modelos mais confiáveis do CMIP6
            """)
        
        with st.expander("O que é um Ensemble de Modelos?"):
            st.markdown("""
            Um **ensemble de modelos climáticos** é um conjunto de múltiplos GCMs 
            que são usados juntos para:
            
            • **Capturar incerteza**: Diferentes modelos têm diferentes projeções
            • **Robustez estatística**: Média de múltiplos modelos é mais confiável
            • **Identificar consenso**: Áreas onde todos os modelos concordam
            • **Quantificar variabilidade**: Range de possíveis futuros climáticos
            
            **Por que não usamos ensemble no TAIPA?**
            - Simplificação pedagógica
            - Redução de complexidade computacional
            - Foco no aprendizado de conceitos básicos
            - Limitações de recursos
            
            ⚠️ **Em pesquisa real, sempre use múltiplos modelos!**
            """)

if __name__ == "__main__":
    render_page()