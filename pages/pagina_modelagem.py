import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime

from utils.modeling import SDMModel
from utils.model_evaluation import create_roc_curve, create_confusion_matrix_plot

def render_page():
    st.title("🤖 Modelagem e Resultados")
    st.markdown("Treine modelos de distribuição de espécies e visualize os resultados")
    
    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = False
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Configurações do Modelo")
        
        # General settings
        st.subheader("Configurações Gerais")
        model_name = st.text_input("Nome do Modelo", value="SDM_Model")
        random_state = st.number_input("Random State", value=42, min_value=0)
        
        # Training settings
        st.subheader("Configurações de Treinamento")
        use_cross_validation = st.checkbox("Usar Validação Cruzada (Stratified)", value=True)
        
        if use_cross_validation:
            n_folds = st.slider("Número de Folds", min_value=3, max_value=10, value=5)
        else:
            test_size = st.slider("Tamanho do Conjunto de Teste", 
                                min_value=0.1, max_value=0.5, value=0.3, step=0.05)
        
        # Model hyperparameters
        st.subheader("Hiperparâmetros do Random Forest")
        n_estimators = st.slider("Número de Árvores", 
                               min_value=50, max_value=500, value=100, step=50)
        max_depth = st.slider("Profundidade Máxima", 
                            min_value=5, max_value=30, value=10, step=5)
        min_samples_split = st.slider("Amostras Mínimas para Split", 
                                    min_value=2, max_value=20, value=2)
        min_samples_leaf = st.slider("Amostras Mínimas por Folha", 
                                   min_value=1, max_value=10, value=1)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Preparação de Dados",
        "2. Treinamento",
        "3. Avaliação",
        "4. Salvar/Carregar Modelo"
    ])
    
    with tab1:
        st.header("Preparação de Dados para Modelagem")
        
        # Debug: show what's in session state
        if st.checkbox("Debug: Ver session_state"):
            relevant_keys = [k for k in st.session_state.keys() if any(x in k for x in ['bioclim', 'occurrence', 'pseudo', 'absence', 'data'])]
            st.write("Keys relevantes no session_state:", relevant_keys)
            for key in relevant_keys:
                st.write(f"{key}: {type(st.session_state[key])}")
                if isinstance(st.session_state[key], pd.DataFrame):
                    st.write(f"  Shape: {st.session_state[key].shape}")
                    st.write(f"  Columns: {list(st.session_state[key].columns)}")
        
        # Check if bioclim data is available
        if 'bioclim_data' not in st.session_state:
            st.warning("⚠️ Nenhum dado preparado para modelagem encontrado.")
            st.info("""
            Para continuar com a modelagem, você precisa:
            1. Buscar dados de ocorrência no módulo GBIF
            2. Gerar pseudo-ausências
            3. Extrair valores bioclimáticos no módulo de Análise Bioclimática
            """)
            return
            
        # Get the processed data
        bioclim_data = st.session_state['bioclim_data']
        
        # Check if we have both presence and absence data
        if 'point_type' not in bioclim_data.columns:
            st.error("⚠️ Dados incompletos. O tipo de ponto (presença/ausência) não foi encontrado.")
            return
            
        # Data summary
        presence_data = bioclim_data[bioclim_data['point_type'] == 'presence']
        absence_data = bioclim_data[bioclim_data['point_type'] == 'absence']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pontos de Presença", len(presence_data))
        with col2:
            st.metric("Pontos de Pseudo-ausência", len(absence_data))
        with col3:
            st.metric("Total de Pontos", len(bioclim_data))
            
        # Variable selection
        st.subheader("Seleção de Variáveis")
        
        # Get environmental variables (excluding latitude, longitude, and point_type)
        env_vars = [col for col in bioclim_data.columns 
                   if col not in ['latitude', 'longitude', 'point_type']]
        
        st.success(f"✅ Dados bioclimáticos disponíveis com {len(env_vars)} variáveis")
        
        # Check if we have selected variables from the bioclim analysis
        if 'selected_bioclim_vars' in st.session_state:
            default_vars = st.session_state['selected_bioclim_vars']
            st.info(f"Usando {len(default_vars)} variáveis selecionadas na análise bioclimática.")
        else:
            default_vars = env_vars
        
        # Allow user to select/modify variables
        selected_vars = st.multiselect(
            "Selecione as variáveis para o modelo",
            options=env_vars,
            default=default_vars,
            help="Selecione as variáveis bioclimáticas para incluir no modelo"
        )
        
        if not selected_vars:
            st.warning("Por favor, selecione pelo menos uma variável.")
            return
            
        # Variable correlation analysis
        if st.checkbox("Mostrar Análise de Correlação"):
            # Use bioclim data with selected variables only
            all_data = bioclim_data[selected_vars]
            
            # Correlation matrix
            corr_matrix = all_data.corr()
            
            # Plot correlation heatmap
            st.subheader("Matriz de Correlação")
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                      square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                      ax=ax, fmt='.2f', annot_kws={'size': 8})
            ax.set_title('Correlação entre Variáveis Bioclimáticas', fontsize=16, pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show highly correlated pairs
            st.subheader("Pares Altamente Correlacionados (|r| > 0.7)")
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append({
                            'Variável 1': corr_matrix.columns[i],
                            'Variável 2': corr_matrix.columns[j],
                            'Correlação': corr_matrix.iloc[i, j]
                        })
            
            if high_corr:
                high_corr_df = pd.DataFrame(high_corr)
                st.dataframe(high_corr_df)
            else:
                st.info("Nenhum par de variáveis com correlação > 0.7")
        
        # Prepare final dataset
        st.subheader("Conjunto de Dados Final")
        
        # Create target variable (1 for presence, 0 for absence)
        y = (bioclim_data['point_type'] == 'presence').astype(int)
        X = bioclim_data[selected_vars]
        
        st.write(f"Tamanho do conjunto de dados: {X.shape}")
        st.write(f"Distribuição das classes: Presença={y.sum()}, Ausência={len(y)-y.sum()}")
        
        # Store prepared data
        st.session_state['X_model'] = X
        st.session_state['y_model'] = y
        st.session_state['selected_vars'] = selected_vars
        
        st.success("✅ Dados preparados para modelagem!")
    
    with tab2:
        st.header("Treinamento do Modelo")
        
        if 'X_model' not in st.session_state:
            st.warning("⚠️ Prepare os dados na aba anterior primeiro.")
            return
        
        X = st.session_state['X_model']
        y = st.session_state['y_model']
        
        # Model initialization
        model_params = {
            'model_type': 'random_forest',
            'random_state': random_state,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf
        }
        
        # Train model
        if st.button("Treinar Modelo", type="primary"):
            with st.spinner("Treinando o modelo..."):
                
                # Initialize model
                sdm_model = SDMModel(**model_params)
                
                if use_cross_validation:
                    # Cross-validation training
                    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
                    
                    cv_scores = {
                        'accuracy': [],
                        'precision': [],
                        'recall': [],
                        'f1': [],
                        'auc': []
                    }
                    
                    progress_bar = st.progress(0)
                    
                    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        
                        # Train on fold
                        sdm_model.train(X_train, y_train)
                        
                        # Predict on validation set
                        y_pred = sdm_model.predict(X_val)
                        y_proba = sdm_model.predict_proba(X_val)[:, 1]
                        
                        # Calculate metrics
                        cv_scores['accuracy'].append(accuracy_score(y_val, y_pred))
                        cv_scores['precision'].append(precision_score(y_val, y_pred))
                        cv_scores['recall'].append(recall_score(y_val, y_pred))
                        cv_scores['f1'].append(f1_score(y_val, y_pred))
                        cv_scores['auc'].append(roc_auc_score(y_val, y_proba))
                        
                        progress_bar.progress((fold + 1) / n_folds)
                    
                    # Train final model on all data
                    sdm_model.train(X, y)
                    
                    # Show CV results
                    st.subheader("Resultados da Validação Cruzada")
                    
                    cv_results = pd.DataFrame(cv_scores)
                    cv_summary = cv_results.describe()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(cv_results)
                    
                    with col2:
                        # Plot CV results
                        fig, ax = plt.subplots(figsize=(8, 6))
                        cv_results.boxplot(ax=ax)
                        ax.set_ylabel('Score')
                        ax.set_title('Distribuição das Métricas - CV')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Show mean scores
                    st.subheader("Métricas Médias")
                    mean_scores = cv_results.mean()
                    
                    cols = st.columns(len(mean_scores))
                    for i, (metric, score) in enumerate(mean_scores.items()):
                        with cols[i]:
                            st.metric(metric.title(), f"{score:.3f}")
                    
                else:
                    # Simple train-test split
                    from sklearn.model_selection import train_test_split
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )
                    
                    # Train model
                    sdm_model.train(X_train, y_train)
                    
                    # Predict on test set
                    y_pred = sdm_model.predict(X_test)
                    y_proba = sdm_model.predict_proba(X_test)[:, 1]
                    
                    # Calculate metrics
                    test_metrics = {
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred),
                        'Recall': recall_score(y_test, y_pred),
                        'F1-Score': f1_score(y_test, y_pred),
                        'AUC': roc_auc_score(y_test, y_proba)
                    }
                    
                    # Show results
                    st.subheader("Resultados no Conjunto de Teste")
                    
                    cols = st.columns(len(test_metrics))
                    for i, (metric, score) in enumerate(test_metrics.items()):
                        with cols[i]:
                            st.metric(metric, f"{score:.3f}")
                
                # Feature importance
                st.subheader("Importância das Variáveis")
                
                feature_importance = pd.DataFrame({
                    'variable': X.columns,
                    'importance': sdm_model.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Plot feature importance
                fig = px.bar(feature_importance, x='importance', y='variable', 
                           orientation='h', title='Importância das Variáveis')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Store model and results
                st.session_state['trained_model'] = sdm_model
                st.session_state['feature_importance'] = feature_importance
                st.session_state['model_trained'] = True
                
                st.success("✅ Modelo treinado com sucesso!")
    
    with tab3:
        st.header("Avaliação do Modelo")
        
        if not st.session_state.get('model_trained'):
            st.warning("⚠️ Treine o modelo na aba anterior primeiro.")
            return
        
        sdm_model = st.session_state['trained_model']
        X = st.session_state['X_model']
        y = st.session_state['y_model']
        
        # Make predictions on full dataset for visualization
        y_pred = sdm_model.predict(X)
        y_proba = sdm_model.predict_proba(X)[:, 1]
        
        # Overall metrics
        st.subheader("Métricas Gerais")
        
        overall_metrics = {
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred),
            'Recall': recall_score(y, y_pred),
            'F1-Score': f1_score(y, y_pred),
            'AUC': roc_auc_score(y, y_proba)
        }
        
        cols = st.columns(len(overall_metrics))
        for i, (metric, score) in enumerate(overall_metrics.items()):
            with cols[i]:
                st.metric(metric, f"{score:.3f}")
        
        # ROC Curve
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Curva ROC")
            fig_roc = create_roc_curve(y, y_proba)
            st.pyplot(fig_roc)
        
        with col2:
            st.subheader("Matriz de Confusão")
            fig_cm = create_confusion_matrix_plot(y, y_pred)
            st.pyplot(fig_cm)
        
        # Probability distribution
        st.subheader("Distribuição das Probabilidades Preditas")
        
        fig = px.histogram(x=y_proba, color=y.astype(str), 
                         nbins=50, opacity=0.7,
                         labels={'x': 'Probabilidade', 'color': 'Classe Real'},
                         title='Distribuição das Probabilidades por Classe')
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance recap
        st.subheader("Importância das Variáveis (Recap)")
        feature_importance = st.session_state['feature_importance']
        st.dataframe(feature_importance)
    
    with tab4:
        st.header("Salvar e Carregar Modelos")
        
        # Save model section
        st.subheader("Salvar Modelo")
        
        if st.session_state.get('model_trained'):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                save_name = st.text_input(
                    "Nome do arquivo",
                    value=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                )
            
            with col2:
                if st.button("Salvar Modelo", type="primary"):
                    try:
                        # Create models directory if it doesn't exist
                        os.makedirs("models", exist_ok=True)
                        
                        # Save model
                        model_path = f"models/{save_name}.pkl"
                        joblib.dump(st.session_state['trained_model'], model_path)
                        
                        # Save metadata
                        metadata = {
                            'model_name': save_name,
                            'saved_date': datetime.now().isoformat(),
                            'selected_vars': st.session_state['selected_vars'],
                            'n_samples': len(st.session_state['X_model']),
                            'feature_importance': st.session_state['feature_importance'].to_dict()
                        }
                        
                        metadata_path = f"models/{save_name}_metadata.json"
                        import json
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        st.success(f"✅ Modelo salvo em: {model_path}")
                    except Exception as e:
                        st.error(f"Erro ao salvar modelo: {e}")
        else:
            st.info("⚠️ Treine um modelo primeiro antes de salvar.")
        
        # Load model section
        st.subheader("Carregar Modelo")
        
        # List available models
        model_files = []
        if os.path.exists("models"):
            model_files = [f for f in os.listdir("models") if f.endswith('.pkl')]
        
        if model_files:
            selected_model = st.selectbox("Selecione um modelo", model_files)
            
            if st.button("Carregar Modelo"):
                try:
                    # Load model
                    model_path = f"models/{selected_model}"
                    loaded_model = joblib.load(model_path)
                    
                    # Load metadata if available
                    metadata_path = model_path.replace('.pkl', '_metadata.json')
                    if os.path.exists(metadata_path):
                        import json
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        st.write("Informações do Modelo:")
                        st.json(metadata)
                    
                    st.session_state['trained_model'] = loaded_model
                    st.session_state['model_trained'] = True
                    
                    st.success("✅ Modelo carregado com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao carregar modelo: {e}")
        else:
            st.info("Nenhum modelo salvo encontrado.")

if __name__ == "__main__":
    render_page()