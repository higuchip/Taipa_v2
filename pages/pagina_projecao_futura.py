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
import geopandas as gpd
from rasterio.mask import mask
from utils.brazil_boundary import get_brazil_boundary, get_brazil_gdf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def render_page():
    st.title("🌡️ Projeção Futura - Mudanças Climáticas")
    st.markdown("Analise como as mudanças climáticas podem afetar a distribuição da espécie")
    
    # Exibir informações da espécie
    if 'species_name' in st.session_state:
        st.info(f"🌿 Espécie: **{st.session_state['species_name']}**")
    else:
        st.warning("⚠️ Nenhuma espécie selecionada. Por favor, comece pela busca de espécies.")
        return
    
    # Verificar se o modelo está treinado
    if not st.session_state.get('model_trained'):
        st.warning("⚠️ Treine um modelo na aba de Modelagem primeiro.")
        return
    
    modelo = st.session_state['trained_model']
    variaveis_selecionadas = st.session_state['selected_vars']
    
    # Barra lateral de configuração
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
        
        # Seleção de cenário
        cenario = st.selectbox(
            "Cenário climático",
            ["SSP1-2.6 (Otimista)", "SSP5-8.5 (Pessimista)"],
            help="SSP1-2.6: Desenvolvimento sustentável com forte mitigação\nSSP5-8.5: Uso intensivo de combustíveis fósseis"
        )
        
        # Seleção de período - agora fixado em 2081-2100
        periodo = "2081-2100 (Futuro distante)"
        st.info("📅 Período fixado em 2081-2100 para análise de impactos de longo prazo")
        
        # Opções de exibição
        st.subheader("Opções de Visualização")
        
        # Seleção de threshold
        st.subheader("Threshold para Mapa Binário")
        metodo_threshold = st.selectbox(
            "Método de threshold",
            ["Manual", "Usar do mapa atual", "Média das Predições", "Percentil 50", "Percentil 10"]
        )
        
        if metodo_threshold == "Manual":
            threshold = st.slider("Threshold manual", 0.0, 1.0, 0.5, step=0.01)
        elif metodo_threshold == "Usar do mapa atual":
            if 'projection_threshold' in st.session_state:
                threshold = st.session_state['projection_threshold']
                st.info(f"Usando threshold do mapa atual: {threshold:.3f}")
            else:
                threshold = 0.5
                st.warning("Threshold do mapa atual não encontrado. Usando 0.5")
        else:
            threshold = None  # Será calculado com base nos dados
        
    
    # Extrair códigos de cenário e período
    codigo_cenario = "ssp126" if "SSP1-2.6" in cenario else "ssp585"
    codigo_periodo = "2081-2100"  # Período fixo
    
    # Conteúdo principal
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Projeção de Distribuição Futura")
        
        # Usar o estado da sessão para manter o estado
        if 'future_projection_done' not in st.session_state:
            st.session_state.future_projection_done = False
        
        if st.button("Gerar Projeção Futura", type="primary"):
            st.session_state.future_projection_done = True
        
        if st.session_state.future_projection_done:
            with st.spinner("Preparando dados climáticos futuros..."):
                try:
                    # Caminho dos dados climáticos futuros
                    caminho_clima_futuro = Path(f"data/worldclim_future/{codigo_cenario}_{codigo_periodo}")
                    
                    # Verificar se os dados futuros existem
                    if not caminho_clima_futuro.exists():
                        st.error(f"Dados climáticos futuros não encontrados em: {caminho_clima_futuro}")
                        st.info("💡 Execute o script de download de dados futuros primeiro.")
                        return
                    
                    # Carregar previsão atual para comparação
                    previsao_atual = st.session_state.get('last_prediction')
                    if previsao_atual is None:
                        st.warning("Execute uma projeção espacial atual primeiro para comparação.")
                        return
                    
                    # Carregar dados climáticos futuros
                    st.info("Carregando dados climáticos futuros e aplicando máscara do Brasil...")
                    
                    # Obter fronteira do Brasil como GeoDataFrame
                    brasil_gdf = get_brazil_gdf()
                    if brasil_gdf.crs != 'EPSG:4326':
                        brasil_gdf = brasil_gdf.to_crs('EPSG:4326')
                    
                    # Get spatial reference from first file
                    first_var = variaveis_selecionadas[0]
                    var_num = int(first_var.replace('bio', ''))
                    ref_file = caminho_clima_futuro / f"wc2.1_2.5m_bioc_MPI-ESM1-2-HR_{codigo_cenario}_{codigo_periodo}_bio{var_num}.tif"
                    
                    if not ref_file.exists():
                        st.error(f"Arquivo de referência não encontrado: {ref_file}")
                        return
                    
                    # Process Brazil boundary for masking
                    with rasterio.open(ref_file) as src:
                        # Reproject Brazil boundary to match raster CRS if needed
                        if brasil_gdf.crs != src.crs:
                            brasil_gdf_proj = brasil_gdf.to_crs(src.crs)
                        else:
                            brasil_gdf_proj = brasil_gdf
                        
                        # Get the geometry for masking
                        brazil_geom = [brasil_gdf_proj.geometry[0]]
                        
                        # Get masked bounds and transform for Brazil
                        out_image, out_transform = mask(src, brazil_geom, crop=True)
                        out_meta = src.meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform
                        })
                        
                        # Use Brazil-cropped dimensions
                        height = out_meta['height']
                        width = out_meta['width']
                        transform = out_transform
                        crs = src.crs
                        bounds = rasterio.transform.array_bounds(height, width, transform)
                    
                    # Create arrays for future climate data
                    n_vars = len(variaveis_selecionadas)
                    bio_data_future = np.zeros((n_vars, height, width))
                    
                    # Load selected bioclimatic variables with Brazil mask
                    progress_bar = st.progress(0)
                    for i, var in enumerate(variaveis_selecionadas):
                        var_num = int(var.replace('bio', ''))
                        tif_path = caminho_clima_futuro / f"wc2.1_2.5m_bioc_MPI-ESM1-2-HR_{codigo_cenario}_{codigo_periodo}_bio{var_num}.tif"
                        
                        if not tif_path.exists():
                            st.error(f"Arquivo não encontrado: {tif_path}")
                            continue
                        
                        with rasterio.open(tif_path) as src:
                            # Apply Brazil mask to the data
                            masked_data, _ = mask(src, brazil_geom, crop=True)
                            data = masked_data[0]  # Get first band
                            
                            # Apply temperature conversion if needed
                            if var_num in [1,2,3,4,5,6,7,8,9,10,11]:
                                valid_mask = data != src.nodata
                                data[valid_mask] = data[valid_mask] / 10.0
                            
                            bio_data_future[i] = data
                        
                        progress_bar.progress((i + 1) / n_vars)
                    
                    # Reshape for prediction
                    X_future = bio_data_future.reshape(n_vars, -1).T
                    
                    # Remove NoData
                    valid_mask = ~np.any(X_future == -9999, axis=1)
                    X_valid = X_future[valid_mask]
                    
                    # Make predictions
                    st.info("Gerando predições para clima futuro...")
                    predictions_future = modelo.predict_proba(X_valid)[:, 1]
                    
                    # Create prediction map
                    prediction_map_future = np.full(X_future.shape[0], np.nan)
                    prediction_map_future[valid_mask] = predictions_future
                    prediction_map_future = prediction_map_future.reshape(height, width)
                    
                    # Calculate threshold if needed
                    if threshold is None:
                        valid_probs = prediction_map_future[~np.isnan(prediction_map_future)]
                        
                        if metodo_threshold == "Média das Predições":
                            threshold = np.mean(valid_probs)
                        elif metodo_threshold == "Percentil 50":
                            threshold = np.percentile(valid_probs, 50)
                        elif metodo_threshold == "Percentil 10":
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
                    tabs = st.tabs(["Comparação Visual", "Análise de Mudanças", "Métricas Detalhadas", "Exportar Mapas"])
                    
                    with tabs[0]:
                        st.subheader("Comparação Visual: Presente vs Futuro")
                        
                        # Obter fronteira do Brasil para plotagem
                        geometria_brasil_plot = brasil_gdf.geometry[0]
                        
                        # Extrair coordenadas para plotar fronteira do Brasil
                        if geometria_brasil_plot.geom_type == 'MultiPolygon':
                            # Para MultiPolygon, precisamos lidar com múltiplas partes
                            brasil_x = []
                            brasil_y = []
                            for poligono in geometria_brasil_plot.geoms:
                                x, y = poligono.exterior.coords.xy
                                brasil_x.extend(list(x) + [None])  # Adicionar None para criar quebras entre polígonos
                                brasil_y.extend(list(y) + [None])
                        else:
                            # Para Polígono único
                            brasil_x, brasil_y = geometria_brasil_plot.exterior.coords.xy
                        
                        # Seção 1: Comparação de mapas binários
                        st.markdown("#### 1. Mapas Binários (Presença/Ausência)")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("##### Distribuição Atual")
                            fig_current_binary = go.Figure()
                            
                            # Add heatmap
                            fig_current_binary.add_trace(go.Heatmap(
                                z=binary_map_current[::-1],
                                colorscale=[[0, 'white'], [1, 'darkgreen']],
                                showscale=True,
                                colorbar=dict(title="Presença", tickvals=[0, 1], ticktext=['Ausente', 'Presente']),
                                x=np.linspace(bounds[0], bounds[2], width),  # west to east
                                y=np.linspace(bounds[1], bounds[3], height)  # south to north
                            ))
                            
                            # Add Brazil boundary
                            fig_current_binary.add_trace(go.Scattergl(
                                x=brasil_x,
                                y=brasil_y,
                                mode='lines',
                                line=dict(color='black', width=2),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                            
                            fig_current_binary.update_layout(
                                title=f"Presente (threshold: {st.session_state.get('projection_threshold', threshold):.3f})",
                                xaxis_title="Longitude",
                                yaxis_title="Latitude",
                                height=400,
                                xaxis=dict(scaleanchor='y', scaleratio=1)
                            )
                            st.plotly_chart(fig_current_binary, use_container_width=True)
                        
                        with col2:
                            st.markdown(f"#### Distribuição Futura ({codigo_periodo})")
                            fig_future_binary = go.Figure()
                            
                            # Add heatmap
                            fig_future_binary.add_trace(go.Heatmap(
                                z=binary_map_future[::-1],
                                colorscale=[[0, 'white'], [1, 'darkgreen']],
                                showscale=True,
                                colorbar=dict(title="Presença", tickvals=[0, 1], ticktext=['Ausente', 'Presente']),
                                x=np.linspace(bounds[0], bounds[2], width),  # west to east
                                y=np.linspace(bounds[1], bounds[3], height)  # south to north
                            ))
                            
                            # Add Brazil boundary
                            fig_future_binary.add_trace(go.Scattergl(
                                x=brasil_x,
                                y=brasil_y,
                                mode='lines',
                                line=dict(color='black', width=2),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                            
                            fig_future_binary.update_layout(
                                title=f"{cenario} - {periodo} (threshold: {threshold:.3f})",
                                xaxis_title="Longitude",
                                yaxis_title="Latitude",
                                height=400,
                                xaxis=dict(scaleanchor='y', scaleratio=1)
                            )
                            st.plotly_chart(fig_future_binary, use_container_width=True)
                        
                        # Section 2: Probability maps
                        st.markdown("---")
                        st.markdown("#### 2. Mapas de Probabilidade")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Probabilidade Atual")
                            fig_current_prob = go.Figure(data=go.Heatmap(
                                z=previsao_atual['map'][::-1],
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
                            # Add Brazil boundary
                            fig_current_prob.add_trace(go.Scattergl(
                                x=brasil_x,
                                y=brasil_y,
                                mode='lines',
                                line=dict(color='black', width=2),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                            st.plotly_chart(fig_current_prob, use_container_width=True)
                        
                        with col2:
                            st.markdown(f"#### Probabilidade Futura ({codigo_periodo})")
                            fig_future_prob = go.Figure(data=go.Heatmap(
                                z=prediction_map_future[::-1],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Probabilidade")
                            ))
                            fig_future_prob.update_layout(
                                title=f"{cenario} - {periodo}",
                                xaxis_title="Longitude",
                                yaxis_title="Latitude",
                                height=400
                            )
                            # Add Brazil boundary
                            fig_future_prob.add_trace(go.Scattergl(
                                x=brasil_x,
                                y=brasil_y,
                                mode='lines',
                                line=dict(color='black', width=2),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                            st.plotly_chart(fig_future_prob, use_container_width=True)
                    
                    with tabs[1]:
                        st.subheader("Análise de Mudanças")
                        
                        # Calculate change map with 4 categories
                        # -1: loss (1->0), 0: stable absent (0->0), 1: gain (0->1), 2: stable present (1->1)
                        change_map = np.full_like(binary_map_current, np.nan, dtype=float)
                        
                        # Create masks for each category
                        valid_pixels = ~np.isnan(binary_map_current) & ~np.isnan(binary_map_future)
                        
                        # Loss: present to absent
                        loss_mask = (binary_map_current == 1) & (binary_map_future == 0) & valid_pixels
                        change_map[loss_mask] = -1
                        
                        # Stable absent: absent to absent
                        stable_absent_mask = (binary_map_current == 0) & (binary_map_future == 0) & valid_pixels
                        change_map[stable_absent_mask] = 0
                        
                        # Gain: absent to present
                        gain_mask = (binary_map_current == 0) & (binary_map_future == 1) & valid_pixels
                        change_map[gain_mask] = 1
                        
                        # Stable present: present to present
                        stable_present_mask = (binary_map_current == 1) & (binary_map_future == 1) & valid_pixels
                        change_map[stable_present_mask] = 2
                        
                        # Create custom colorscale for 4 categories
                        # Define discrete colorscale
                        colorscale = [
                            # Value -1: Red (loss)
                            [0.0, 'rgb(220, 20, 20)'],
                            [0.2, 'rgb(220, 20, 20)'],
                            # Value 0: Light gray (stable absent)
                            [0.2, 'rgb(220, 220, 220)'],
                            [0.4, 'rgb(220, 220, 220)'],
                            # Value 1: Green (gain)
                            [0.4, 'rgb(34, 139, 34)'],
                            [0.6, 'rgb(34, 139, 34)'],
                            # Value 2: Dark gray (stable present)
                            [0.6, 'rgb(105, 105, 105)'],
                            [1.0, 'rgb(105, 105, 105)']
                        ]
                        
                        # Normalize change_map values to 0-1 range for colorscale
                        # -1 -> 0.1, 0 -> 0.3, 1 -> 0.5, 2 -> 0.9
                        z_normalized = change_map.copy()
                        z_normalized[change_map == -1] = 0.1
                        z_normalized[change_map == 0] = 0.3
                        z_normalized[change_map == 1] = 0.5
                        z_normalized[change_map == 2] = 0.9
                        
                        fig_change = go.Figure(data=go.Heatmap(
                            z=z_normalized[::-1],
                            colorscale=colorscale,
                            zmin=0,
                            zmax=1,
                            showscale=True,
                            colorbar=dict(
                                title="Mudança",
                                tickvals=[0.1, 0.3, 0.5, 0.9],
                                ticktext=['Perda', 'Sem adequabilidade', 'Ganho', 'Adequabilidade mantida'],
                                tickmode='array'
                            )
                        ))
                        fig_change.update_layout(
                            title=f"Mudança na Distribuição (Futuro - Presente) - Threshold: {threshold:.3f}",
                            xaxis_title="Longitude",
                            yaxis_title="Latitude",
                            height=500
                        )
                        # Add Brazil boundary
                        fig_change.add_trace(go.Scattergl(
                            x=brasil_x,
                            y=brasil_y,
                            mode='lines',
                            line=dict(color='black', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        st.plotly_chart(fig_change, use_container_width=True)
                        
                        st.info("""
                        🔴 **Vermelho**: Áreas com perda de habitat adequado (adequado → não adequado)
                        🟢 **Verde**: Áreas com ganho de habitat adequado (não adequado → adequado)
                        ⚪ **Cinza claro**: Áreas que permanecem sem adequabilidade
                        ⚫ **Cinza escuro**: Áreas que permanecem com adequabilidade
                        """)
                    
                    with tabs[2]:
                        st.subheader("Métricas Detalhadas de Mudança")
                        
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
                            labels=['Ganho de Habitat', 'Perda de Habitat', 'Adequabilidade Mantida', 'Sem Adequabilidade'],
                            values=[gain_area, loss_area, stable_present_area, stable_absent_area],
                            hole=.3,
                            marker_colors=['green', 'red', 'darkgray', 'lightgray']
                        )])
                        fig_pie.update_layout(title="Distribuição de Mudanças na Adequabilidade")
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                        # Additional metrics
                        st.markdown("### Resumo Detalhado")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Ganho de Adequabilidade", f"{gain_area:,.0f} km²")
                            st.metric("Adequabilidade Mantida", f"{stable_present_area:,.0f} km²")
                        
                        with col2:
                            st.metric("Perda de Adequabilidade", f"{loss_area:,.0f} km²")
                            st.metric("Mudança Líquida", f"{(gain_area - loss_area):,.0f} km²")
                    
                    with tabs[3]:
                        st.subheader("Exportar Resultados")
                        
                        # Save future prediction
                        st.session_state['future_prediction'] = {
                            'probability_map': prediction_map_future,
                            'binary_map': binary_map_future,
                            'threshold': threshold,
                            'scenario': cenario,
                            'period': periodo,
                            'bounds': bounds,
                            'crs': crs,
                            'transform': transform
                        }
                        
                        # Only JPEG export section
                        st.markdown("### 📥 Exportar Mapas em JPEG")
                        st.info("Clique nos botões abaixo para baixar os mapas em alta resolução")
                        
                        import io
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Create JPEG in memory for binary map
                            binary_jpeg_buffer = io.BytesIO()
                            
                            # Create colorful visualization
                            fig, ax = plt.subplots(figsize=(10, 8))
                            
                            # Create custom colormap for binary map (white for 0, dark green for 1)
                            cmap = mcolors.ListedColormap(['white', 'darkgreen'])
                            # Use correct orientation with origin parameter
                            im = ax.imshow(binary_map_future, cmap=cmap, extent=[bounds[0], bounds[2], bounds[1], bounds[3]], origin='upper')
                            
                            # Add Brazil boundary
                            ax.plot(brasil_x, brasil_y, 'k-', linewidth=2)
                            
                            # Add labels and title
                            ax.set_xlabel('Longitude')
                            ax.set_ylabel('Latitude')
                            ax.set_title(f'Distribuição Futura - {cenario} ({periodo})')
                            
                            # Add colorbar
                            cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
                            cbar.set_label('Presença')
                            cbar.ax.set_yticklabels(['Ausente', 'Presente'])
                            
                            # Save to buffer
                            plt.tight_layout()
                            plt.savefig(binary_jpeg_buffer, format='jpeg', dpi=300, bbox_inches='tight')
                            plt.close()
                            binary_jpeg_buffer.seek(0)
                            
                            st.download_button(
                                label="⬇️ Mapa Binário",
                                data=binary_jpeg_buffer,
                                file_name=f"future_binary_{codigo_cenario}_{codigo_periodo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                                mime="image/jpeg",
                                key="download_binary_jpeg"
                            )
                                
                            with col2:
                                # Create JPEG in memory for probability map
                                prob_jpeg_buffer = io.BytesIO()
                                
                                # Create colorful visualization
                                fig, ax = plt.subplots(figsize=(10, 8))
                                
                                # Use Viridis colormap for probability
                                im = ax.imshow(prediction_map_future, cmap='viridis', extent=[bounds[0], bounds[2], bounds[1], bounds[3]], vmin=0, vmax=1, origin='upper')
                                
                                # Add Brazil boundary
                                ax.plot(brasil_x, brasil_y, 'k-', linewidth=2)
                                
                                # Add labels and title
                                ax.set_xlabel('Longitude')
                                ax.set_ylabel('Latitude')
                                ax.set_title(f'Probabilidade de Ocorrência Futura - {cenario} ({periodo})')
                                
                                # Add colorbar
                                cbar = plt.colorbar(im, ax=ax)
                                cbar.set_label('Probabilidade')
                                
                                # Save to buffer
                                plt.tight_layout()
                                plt.savefig(prob_jpeg_buffer, format='jpeg', dpi=300, bbox_inches='tight')
                                plt.close()
                                prob_jpeg_buffer.seek(0)
                                
                                st.download_button(
                                    label="⬇️ Mapa de Probabilidade",
                                    data=prob_jpeg_buffer,
                                    file_name=f"future_probability_{codigo_cenario}_{codigo_periodo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                                    mime="image/jpeg",
                                    key="download_prob_jpeg"
                                )
                            
                            with col3:
                                # Create JPEG in memory for change map
                                change_jpeg_buffer = io.BytesIO()
                                
                                # Create colorful visualization
                                fig, ax = plt.subplots(figsize=(10, 8))
                                
                                # Create custom colormap for change map (red for loss, gray for no change, green for gain)
                                cmap = mcolors.ListedColormap(['red', 'lightgray', 'green'])
                                bounds_cmap = [-1.5, -0.5, 0.5, 1.5]
                                norm = mcolors.BoundaryNorm(bounds_cmap, cmap.N)
                                
                                im = ax.imshow(change_map, cmap=cmap, norm=norm, extent=[bounds[0], bounds[2], bounds[1], bounds[3]], origin='upper')
                                
                                # Add Brazil boundary
                                ax.plot(brasil_x, brasil_y, 'k-', linewidth=2)
                                
                                # Add labels and title
                                ax.set_xlabel('Longitude')
                                ax.set_ylabel('Latitude')
                                ax.set_title(f'Mudança na Distribuição - {cenario} ({periodo})')
                                
                                # Add colorbar with custom labels
                                cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
                                cbar.set_label('Mudança')
                                cbar.ax.set_yticklabels(['Perda', 'Sem mudança', 'Ganho'])
                                
                                # Save to buffer
                                plt.tight_layout()
                                plt.savefig(change_jpeg_buffer, format='jpeg', dpi=300, bbox_inches='tight')
                                plt.close()
                                change_jpeg_buffer.seek(0)
                                
                                st.download_button(
                                    label="⬇️ Mapa de Mudanças",
                                    data=change_jpeg_buffer,
                                    file_name=f"change_{codigo_cenario}_{codigo_periodo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                                    mime="image/jpeg",
                                    key="download_change_jpeg"
                                )
                
                except Exception as e:
                    st.error(f"Erro ao gerar projeção futura: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
        
        # Add reset button to go back to initial state
        if st.session_state.future_projection_done:
            st.markdown("---")
            if st.button("Nova Projeção", type="secondary"):
                st.session_state.future_projection_done = False
                st.rerun()
    
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