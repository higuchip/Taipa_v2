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
import matplotlib.pyplot as plt
import seaborn as sns

def render_page():
    st.title("🗺️ Projeção Espacial do Modelo")
    st.markdown("Gere mapas de adequabilidade ambiental usando o modelo treinado")
    
    # Check if model is trained
    if not st.session_state.get('model_trained'):
        st.warning("⚠️ Treine um modelo na aba de Modelagem primeiro.")
        return
    
    model = st.session_state['trained_model']
    selected_vars = st.session_state['selected_vars']
    
    # Debug info
    st.write(f"Variáveis selecionadas: {selected_vars}")
    
    # Check model type and parameters
    if hasattr(model, 'model'):
        st.write(f"Tipo do modelo: {type(model.model)}")
        if hasattr(model.model, 'named_steps'):
            st.write(f"Pipeline steps: {list(model.model.named_steps.keys())}")
    
    # Check training data statistics if available
    if 'X_model' in st.session_state:
        X_train = st.session_state['X_model']
        st.write("Estatísticas dos dados de treinamento:")
        st.write(X_train.describe())
        
        # Show geographic distribution of training points
        if 'bioclim_data' in st.session_state:
            bioclim_data = st.session_state['bioclim_data']
            presence_data = bioclim_data[bioclim_data['point_type'] == 'presence']
            
            st.write(f"Pontos de presença no treinamento: {len(presence_data)}")
            st.write("Distribuição geográfica dos pontos de presença:")
            st.write(f"Latitude: {presence_data['latitude'].min():.2f} a {presence_data['latitude'].max():.2f}")
            st.write(f"Longitude: {presence_data['longitude'].min():.2f} a {presence_data['longitude'].max():.2f}")
    
    # Configuration
    with st.sidebar:
        st.header("⚙️ Configurações da Projeção")
        
        # Info about fixed parameters
        st.info("""
        📍 **Área**: Brasil completo
        📏 **Resolução**: 2.5 minutos (~5 km)
        """)
        
        # Threshold for binary map
        st.subheader("Limiarização")
        
        threshold_method = st.selectbox(
            "Método de threshold",
            ["Manual", "Média das Predições", "Percentil 50", "Percentil 10", 
             "Maximiza TSS (requer dados de teste)", "Maximiza Kappa (requer dados de teste)"]
        )
        
        if threshold_method == "Manual":
            # Adjust the default based on the actual prediction range
            threshold = st.slider("Threshold manual", 0.0, 1.0, 0.25)
        else:
            threshold = None
    
    # Main projection
    if st.button("Gerar Projeção", type="primary"):
        with st.spinner("Preparando dados para projeção..."):
            try:
                # Get WorldClim data for the selected area
                worldclim_path = Path("data/worldclim_brazil")
                
                # Load the first raster to get the spatial reference
                with rasterio.open(worldclim_path / "bio1_brazil.tif") as src:
                    crs = src.crs
                    transform = src.transform
                    height = src.height
                    width = src.width
                    bounds = src.bounds
                
                # Create arrays to store all bioclimatic variables
                n_vars = len(selected_vars)
                bio_data = np.zeros((n_vars, height, width))
                
                # Load selected bioclimatic variables
                progress_bar = st.progress(0)
                for i, var in enumerate(selected_vars):
                    var_num = int(var.replace('bio', ''))
                    tif_path = worldclim_path / f"bio{var_num}_brazil.tif"
                    
                    if not tif_path.exists():
                        st.error(f"Arquivo não encontrado: {tif_path}")
                        continue
                    
                    with rasterio.open(tif_path) as src:
                        data = src.read(1)
                        
                        # Apply the same conversion as in training data extraction
                        # Temperature variables (bio1-11) are stored as °C * 10
                        if var_num in [1,2,3,4,5,6,7,8,9,10,11]:
                            # Only convert non-nodata values
                            mask = data != -9999
                            data[mask] = data[mask] / 10.0
                        
                        bio_data[i] = data
                        
                        # Debug statistics
                        valid_data_mask = data > -9999 if var_num > 11 else data > -999
                        valid_data_temp = data[valid_data_mask]
                        st.write(f"Carregado {var}: min={valid_data_temp.min():.2f}, max={valid_data_temp.max():.2f}, média={valid_data_temp.mean():.2f}")
                    
                    progress_bar.progress((i + 1) / n_vars)
                
                st.success("✅ Dados carregados com sucesso!")
                
                # Prepare data for prediction
                st.write("Preparando dados para predição...")
                
                # Reshape data for prediction
                # From (n_vars, height, width) to (height*width, n_vars)
                bio_flat = bio_data.reshape(n_vars, -1).T
                
                # Create mask for valid data (no nodata values)
                # After temperature conversion, nodata values are -999.9 for temperature vars
                valid_mask = ~np.any(np.logical_or(bio_flat <= -999, np.isnan(bio_flat)), axis=1)
                
                st.write(f"Pixels válidos: {np.sum(valid_mask)} de {len(valid_mask)}")
                
                # Predict only on valid pixels
                predictions = np.full(height * width, np.nan)
                if np.any(valid_mask):
                    valid_data = bio_flat[valid_mask]
                    
                    st.write(f"Shape dos dados válidos: {valid_data.shape}")
                    st.write(f"Min/Max dos dados: {valid_data.min():.2f} / {valid_data.max():.2f}")
                    
                    # Create DataFrame with correct column names
                    df_predict = pd.DataFrame(valid_data, columns=selected_vars)
                    
                    # Debug data before prediction
                    st.write("Debug do DataFrame antes da predição:")
                    st.write(f"Shape: {df_predict.shape}")
                    st.write(f"Colunas: {df_predict.columns.tolist()}")
                    st.write("Primeiras 5 linhas:")
                    st.write(df_predict.head())
                    st.write("Estatísticas descritivas:")
                    st.write(df_predict.describe())
                    
                    # Check scaling if model has scaler
                    if hasattr(model, 'model') and hasattr(model.model, 'named_steps'):
                        if 'scaler' in model.model.named_steps:
                            scaler = model.model.named_steps['scaler']
                            st.write("Parâmetros do StandardScaler (mean e std do treinamento):")
                            for i, var in enumerate(selected_vars):
                                st.write(f"{var}: mean={scaler.mean_[i]:.4f}, std={scaler.scale_[i]:.4f}")
                            
                            # Show how data would be transformed
                            st.write("Exemplo de transformação (primeiras 5 linhas):")
                            transformed_sample = scaler.transform(df_predict.head())
                            st.write(pd.DataFrame(transformed_sample, columns=selected_vars))
                    
                    # Make predictions
                    try:
                        probabilities = model.predict_proba(df_predict)[:, 1]
                        predictions[valid_mask] = probabilities
                        st.write(f"Predições realizadas com sucesso!")
                        st.write(f"Probabilidades - Min: {probabilities.min():.6f}, Max: {probabilities.max():.6f}, Média: {probabilities.mean():.6f}")
                        
                        # Check for suspicious patterns
                        prob_std = probabilities.std()
                        st.write(f"Desvio padrão das probabilidades: {prob_std:.6f}")
                        
                        if prob_std < 0.1:
                            st.warning("⚠️ As probabilidades têm baixa variação, o que pode indicar problemas com o modelo ou dados.")
                        
                        # Check prediction at known presence locations
                        if 'bioclim_data' in st.session_state:
                            bioclim_data = st.session_state['bioclim_data']
                            presence_points = bioclim_data[bioclim_data['point_type'] == 'presence']
                            
                            # Sample a few presence points
                            sample_presence = presence_points.head(5)
                            st.write("Predições em pontos de presença conhecidos:")
                            for idx, row in sample_presence.iterrows():
                                pred_prob = model.predict_proba(row[selected_vars].values.reshape(1, -1))[0, 1]
                                st.write(f"Lat: {row['latitude']:.2f}, Lon: {row['longitude']:.2f} -> Prob: {pred_prob:.4f}")
                                
                    except Exception as e:
                        st.error(f"Erro na predição: {e}")
                        st.write(f"Colunas do DataFrame: {df_predict.columns.tolist()}")
                        st.write(f"Shape do DataFrame: {df_predict.shape}")
                        raise
                
                # Reshape predictions back to 2D
                prediction_map = predictions.reshape(height, width)
                
                # Debug predictions
                st.write(f"Estatísticas das predições:")
                st.write(f"Min: {np.nanmin(prediction_map):.6f}, Max: {np.nanmax(prediction_map):.6f}")
                st.write(f"Média: {np.nanmean(prediction_map):.6f}, Std: {np.nanstd(prediction_map):.6f}")
                st.write(f"Pixels com predição > 0: {np.sum(prediction_map > 0)}")
                st.write(f"Pixels válidos (não-NaN): {np.sum(~np.isnan(prediction_map))}")
                
                # Apply threshold
                if threshold_method != "Manual":
                    valid_probs = prediction_map[~np.isnan(prediction_map)]
                    
                    if threshold_method == "Média das Predições":
                        threshold = np.mean(valid_probs)
                    elif threshold_method == "Percentil 50":
                        threshold = np.percentile(valid_probs, 50)
                    elif threshold_method == "Percentil 10":
                        threshold = np.percentile(valid_probs, 10)
                    elif "TSS" in threshold_method or "Kappa" in threshold_method:
                        # Check if test data is available
                        if 'y_test' not in st.session_state or 'y_pred_proba' not in st.session_state:
                            st.warning(f"⚠️ {threshold_method} requer dados de teste. Usando Percentil 50.")
                            threshold = np.percentile(valid_probs, 50)
                        else:
                            threshold = calculate_optimal_threshold(
                                st.session_state['y_test'],
                                st.session_state['y_pred_proba'],
                                method=threshold_method
                            )
                
                st.write(f"Threshold aplicado: {threshold:.4f}")
                
                # Create binary map
                # First create as float to handle NaN values
                binary_map = np.full_like(prediction_map, np.nan)
                valid_predictions = ~np.isnan(prediction_map)
                binary_map[valid_predictions] = (prediction_map[valid_predictions] >= threshold).astype(float)
                
                # Store results
                st.session_state['prediction_map'] = prediction_map
                st.session_state['binary_map'] = binary_map
                st.session_state['projection_threshold'] = threshold
                st.session_state['projection_metadata'] = {
                    'crs': crs,
                    'transform': transform,
                    'bounds': bounds
                }
                
                st.success("✅ Projeção concluída!")
                
            except Exception as e:
                st.error(f"Erro na projeção: {e}")
                return
    
    # Display results
    if 'prediction_map' in st.session_state:
        st.header("Resultados da Projeção")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Mapa de Probabilidade")
            
            # Show prediction range
            pred_map = st.session_state['prediction_map']
            valid_preds = pred_map[~np.isnan(pred_map)]
            st.info(f"Intervalo de probabilidades: {valid_preds.min():.3f} - {valid_preds.max():.3f}")
            
            fig_prob = create_probability_map(
                pred_map,
                st.session_state['projection_metadata']['bounds']
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            
            # Download probability raster
            if st.button("💾 Baixar Raster de Probabilidade"):
                save_raster(
                    st.session_state['prediction_map'],
                    st.session_state['projection_metadata'],
                    "probability_map.tif"
                )
        
        with col2:
            st.subheader("Mapa Binário")
            st.info(f"Threshold: {st.session_state['projection_threshold']:.3f}")
            
            fig_binary = create_binary_map(
                st.session_state['binary_map'],
                st.session_state['projection_metadata']['bounds']
            )
            st.plotly_chart(fig_binary, use_container_width=True)
            
            # Download binary raster
            if st.button("💾 Baixar Raster Binário"):
                save_raster(
                    st.session_state['binary_map'],
                    st.session_state['projection_metadata'],
                    "binary_map.tif"
                )
        
        # Statistics
        st.header("Estatísticas da Projeção")
        
        valid_pixels = ~np.isnan(st.session_state['prediction_map'])
        total_pixels = np.sum(valid_pixels)
        # Count suitable pixels, handling NaN values
        suitable_pixels = np.nansum(st.session_state['binary_map'] == 1)
        
        # Calculate pixel area from transform
        transform = st.session_state['projection_metadata']['transform']
        pixel_area_km2 = abs(transform.a * transform.e) * (111.32 * 111.32)  # Convert degrees to km
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Área Total", f"{total_pixels * pixel_area_km2:,.0f} km²")
        
        with col2:
            st.metric("Área Adequada", f"{suitable_pixels * pixel_area_km2:,.0f} km²")
        
        with col3:
            st.metric("Porcentagem Adequada", f"{(suitable_pixels/total_pixels)*100:.1f}%")
        
        # Histogram of probabilities
        st.subheader("Distribuição das Probabilidades")
        
        valid_probs = st.session_state['prediction_map'][valid_pixels]
        fig_hist = px.histogram(
            valid_probs, 
            nbins=50, 
            title="Histograma de Probabilidades de Adequabilidade",
            labels={'value': 'Probabilidade', 'count': 'Frequência'}
        )
        fig_hist.add_vline(
            x=st.session_state['projection_threshold'], 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Threshold = {st.session_state['projection_threshold']:.3f}"
        )
        fig_hist.update_layout(
            xaxis_title="Probabilidade",
            yaxis_title="Número de pixels"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Comparison map with training points
        if st.checkbox("Mostrar pontos de treinamento sobre o mapa"):
            st.subheader("Comparação com Pontos de Treinamento")
            
            fig_comparison = create_probability_map(
                st.session_state['prediction_map'],
                st.session_state['projection_metadata']['bounds']
            )
            
            # Add training points if available
            if 'bioclim_data' in st.session_state:
                bioclim_data = st.session_state['bioclim_data']
                presence_data = bioclim_data[bioclim_data['point_type'] == 'presence']
                absence_data = bioclim_data[bioclim_data['point_type'] == 'absence']
                
                # Add presence points
                fig_comparison.add_trace(go.Scatter(
                    x=presence_data['longitude'],
                    y=presence_data['latitude'],
                    mode='markers',
                    name='Presença',
                    marker=dict(color='red', size=8, symbol='circle')
                ))
                
                # Add absence points  
                fig_comparison.add_trace(go.Scatter(
                    x=absence_data['longitude'],
                    y=absence_data['latitude'],
                    mode='markers',
                    name='Pseudo-ausência',
                    marker=dict(color='blue', size=6, symbol='x')
                ))
                
                fig_comparison.update_layout(
                    title='Projeção com Pontos de Treinamento',
                    height=700
                )
            
            st.plotly_chart(fig_comparison, use_container_width=True)

def calculate_optimal_threshold(y_true, y_pred_proba, method="Maximiza TSS"):
    """Calculate optimal threshold based on different methods"""
    if y_true is None or y_pred_proba is None:
        return 0.5  # Default if no test data available
    
    from sklearn.metrics import confusion_matrix
    
    thresholds = np.linspace(0.01, 0.99, 100)  # Avoid 0 and 1 to prevent errors
    metrics = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        # Handle edge cases where all predictions are same class
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        except:
            continue
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        if "TSS" in method:
            score = sensitivity + specificity - 1
        elif "Kappa" in method:
            # Cohen's Kappa
            n = len(y_true)
            observed_agreement = np.sum(y_true == y_pred) / n
            
            p_yes_true = np.sum(y_true == 1) / n
            p_yes_pred = np.sum(y_pred == 1) / n
            p_no_true = np.sum(y_true == 0) / n
            p_no_pred = np.sum(y_pred == 0) / n
            
            expected_agreement = (p_yes_true * p_yes_pred) + (p_no_true * p_no_pred)
            
            if expected_agreement >= 0.999:  # Avoid division by zero
                score = 1.0
            else:
                score = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        else:  # Default to TSS
            score = sensitivity + specificity - 1
        
        metrics.append(score)
    
    if not metrics:  # If no valid metrics calculated
        return 0.5
    
    optimal_idx = np.argmax(metrics)
    return thresholds[optimal_idx]

def create_probability_map(prediction_map, bounds):
    """Create probability map visualization"""
    # Create coordinate arrays
    height, width = prediction_map.shape
    lon = np.linspace(bounds.left, bounds.right, width)
    lat = np.linspace(bounds.bottom, bounds.top, height)  # Changed order
    
    # Use actual min/max for better visualization
    valid_data = prediction_map[~np.isnan(prediction_map)]
    min_val = valid_data.min() if len(valid_data) > 0 else 0
    max_val = valid_data.max() if len(valid_data) > 0 else 1
    
    fig = go.Figure(data=go.Heatmap(
        z=prediction_map[::-1],  # Flip the data vertically
        x=lon,
        y=lat,
        colorscale='Viridis',
        colorbar_title='Probabilidade',
        zmin=min_val,
        zmax=max_val
    ))
    
    fig.update_layout(
        title='Mapa de Probabilidade de Adequabilidade',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        height=600,
        xaxis=dict(scaleanchor='y')
        # Removed yaxis autorange='reversed'
    )
    
    return fig

def create_binary_map(binary_map, bounds):
    """Create binary map visualization"""
    # Create coordinate arrays
    height, width = binary_map.shape
    lon = np.linspace(bounds.left, bounds.right, width)
    lat = np.linspace(bounds.bottom, bounds.top, height)  # Changed order
    
    # Custom colorscale for binary map
    colorscale = [[0, 'lightgray'], [0.5, 'lightgray'], [0.5, 'darkgreen'], [1, 'darkgreen']]
    
    fig = go.Figure(data=go.Heatmap(
        z=binary_map[::-1],  # Flip the data vertically
        x=lon,
        y=lat,
        colorscale=colorscale,
        colorbar=dict(
            tickvals=[0, 1],
            ticktext=['Inadequado', 'Adequado']
        ),
        zmin=0,
        zmax=1
    ))
    
    fig.update_layout(
        title='Mapa Binário de Adequabilidade',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        height=600,
        xaxis=dict(scaleanchor='y')
        # Removed yaxis autorange='reversed'
    )
    
    return fig

def save_raster(data, metadata, filename):
    """Save raster to file and provide download"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
            # Ensure data is float32 for compatibility
            data_to_save = data.astype(np.float32)
            
            # Write raster
            with rasterio.open(
                tmp_file.name,
                'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype='float32',
                crs=metadata['crs'],
                transform=metadata['transform'],
                nodata=-9999
            ) as dst:
                # Replace NaN with nodata value
                data_to_save[np.isnan(data_to_save)] = -9999
                dst.write(data_to_save, 1)
            
            # Read file for download
            with open(tmp_file.name, 'rb') as f:
                st.download_button(
                    label=f"📥 Clique para baixar {filename}",
                    data=f.read(),
                    file_name=filename,
                    mime="image/tiff"
                )
            
            # Clean up
            os.unlink(tmp_file.name)
            
    except Exception as e:
        st.error(f"Erro ao salvar raster: {e}")

if __name__ == "__main__":
    render_page()