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
from utils.spatial_cv import SpatialKFold
from utils.bioclim_labels import get_bioclim_label, format_bioclim_var

def check_species_change():
    """Check if species has changed and clear model state if needed"""
    if 'species_name' not in st.session_state:
        return False
    
    current_species = st.session_state.get('species_name', '')
    last_modeled_species = st.session_state.get('model_species', '')
    
    if current_species != last_modeled_species and st.session_state.get('model_trained', False):
        # Species has changed and we have a trained model
        st.session_state['model_trained'] = False
        st.session_state['trained_model'] = None
        st.session_state['selected_vars'] = None
        st.session_state['X_model'] = None
        st.session_state['y_model'] = None
        st.session_state['feature_importance'] = None
        
        st.warning(f"⚠️ A espécie mudou de '{last_modeled_species}' para '{current_species}'. Por favor, volte à aba 'Preparação de Dados' e retreine o modelo.")
        return True
    
    return False

def render_page():
    # Make sure pandas is available
    global pd
    if 'pd' not in globals():
        import pandas as pd
    
    st.title("🤖 Modelagem e Resultados")
    st.markdown("Treine modelos de distribuição de espécies e visualize os resultados")
    
    # Check for species change
    species_changed = check_species_change()
    
    # Display current species
    if 'species_name' in st.session_state:
        current_species = st.session_state['species_name']
        model_species = st.session_state.get('model_species', 'Nenhum modelo treinado')
        
        if current_species != model_species and st.session_state.get('model_trained', False):
            st.error(f"⚠️ ATENÇÃO: A espécie mudou de '{model_species}' para '{current_species}'. Por favor, retreine o modelo.")
            st.warning("💡 Dica: Volte à aba 'Preparação de Dados' para recarregar os dados da nova espécie.")
    
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
        
        col1, col2 = st.columns(2)
        with col1:
            use_cross_validation = st.checkbox("Usar Validação Cruzada", value=True)
        
        with col2:
            test_size = st.slider("Proporção de Teste (%)", 
                                min_value=10, max_value=40, value=20, step=5,
                                help="Porcentagem dos dados para teste (usado quando CV está desativado)")
        
        if use_cross_validation:
            cv_type = st.radio("Tipo de Validação Cruzada", 
                              ["Estratificada (Padrão)", "Espacial (Recomendada para SDM)"],
                              index=1,
                              help="Validação espacial evita autocorrelação espacial e fornece métricas mais realistas")
            
            n_folds = st.slider("Número de Folds", min_value=3, max_value=10, value=5,
                              help="Número de partições para validação cruzada")
            
            if cv_type == "Espacial (Recomendada para SDM)":
                buffer_distance = st.number_input("Distância de Buffer (km)", 
                                                min_value=0.0, max_value=100.0, value=10.0, step=5.0,
                                                help="Distância de buffer ao redor dos pontos de teste para remover do treino")
                st.info(f"💡 Validação cruzada espacial divide os dados em {n_folds} grupos espaciais com buffer de {buffer_distance}km")
            else:
                st.info(f"💡 Validação cruzada estratificada divide os dados em {n_folds} partes mantendo proporções de classes")
        else:
            st.info(f"💡 {test_size}% dos dados serão usados para teste, {100-test_size}% para treino")
        
        # Model hyperparameters
        st.subheader("Hiperparâmetros do Random Forest")
        st.info("💡 Valores padrão otimizados para Species Distribution Modeling")
        
        n_estimators = st.slider("Número de Árvores", 
                               min_value=100, max_value=1000, value=500, step=50,
                               help="Quantas árvores de decisão no modelo. Mais árvores = maior precisão, mas mais tempo de treino")
        max_depth = st.slider("Profundidade Máxima", 
                            min_value=5, max_value=50, value=20, step=5,
                            help="Profundidade máxima de cada árvore. Valores muito altos podem causar overfitting")
        min_samples_split = st.slider("Amostras Mínimas para Split", 
                                    min_value=2, max_value=20, value=5,
                                    help="Número mínimo de amostras para dividir um nó interno. Valores maiores = modelo mais generalizado")
        min_samples_leaf = st.slider("Amostras Mínimas por Folha", 
                                   min_value=1, max_value=10, value=2,
                                   help="Número mínimo de amostras em um nó folha. Valores maiores evitam overfitting")
        
        # Hyperparameter explanation
        with st.expander("📚 Entenda os hiperparâmetros"):
            st.markdown("""
            **Random Forest para SDM:**
            
            - **Número de Árvores**: Mais árvores melhoram a estabilidade. Para SDM, recomenda-se 300-1000.
            
            - **Profundidade Máxima**: Controla a complexidade do modelo. Valores entre 10-30 são adequados para SDM.
            
            - **Amostras Mínimas para Split**: Evita splits em poucos dados. Valores entre 5-10 são recomendados.
            
            - **Amostras Mínimas por Folha**: Define o tamanho mínimo de grupos finais. Valores 1-3 para SDM.
            
            **Dica**: Para dados ecológicos, é melhor ter um modelo ligeiramente conservador (mais generalizado) 
            do que um modelo que se ajusta perfeitamente aos dados de treino (overfitting).
            """)
        
        # Spatial CV explanation
        with st.expander("🌍 Por que usar Validação Cruzada Espacial?"):
            st.markdown("""
            **Autocorrelação Espacial em Dados Ecológicos:**
            
            Dados de distribuição de espécies frequentemente apresentam **autocorrelação espacial** - 
            pontos próximos tendem a ser mais similares do que pontos distantes. Isso pode levar a:
            
            - **Métricas infladas**: Validação padrão pode resultar em precisão irrealisticamente alta
            - **Overfitting espacial**: Modelo memoriza padrões locais ao invés de relações ecológicas
            - **Previsões ruins**: Desempenho pobre ao prever em novas áreas
            
            **Como a Validação Espacial Resolve:**
            
            1. **Agrupamento espacial**: Divide dados em grupos geograficamente separados
            2. **Buffer de separação**: Remove pontos de treino próximos aos pontos de teste
            3. **Métricas realistas**: Simula melhor a aplicação real do modelo em novas áreas
            
            **Recomendação**: Use sempre validação espacial para SDM, especialmente se planeja 
            aplicar o modelo em áreas não amostradas.
            """)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Preparação de Dados",
        "2. Treinamento",
        "3. Avaliação",
        "4. Salvar/Carregar Modelo"
    ])
    
    with tab1:
        st.header("Preparação de Dados para Modelagem")
        
        
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
            format_func=lambda x: format_bioclim_var(x),
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
            
            # Create correlation matrix with translated labels
            translated_labels = [format_bioclim_var(var, include_unit=False) for var in corr_matrix.columns]
            corr_matrix_display = corr_matrix.copy()
            corr_matrix_display.index = translated_labels
            corr_matrix_display.columns = translated_labels
            
            # Plot correlation heatmap
            st.subheader("Matriz de Correlação")
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix_display, annot=True, cmap='coolwarm', center=0,
                      square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                      ax=ax, fmt='.2f', annot_kws={'size': 8})
            ax.set_title('Correlação entre Variáveis Bioclimáticas', fontsize=16, pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show highly correlated pairs
            st.subheader("Pares Altamente Correlacionados (|r| > 0.7)")
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append({
                            'Variável 1': format_bioclim_var(corr_matrix.columns[i]),
                            'Variável 2': format_bioclim_var(corr_matrix.columns[j]),
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
        
        # Check class distribution
        n_presence = y.sum()
        n_absence = len(y) - n_presence
        
        st.write(f"Distribuição das classes: Presença={n_presence}, Ausência={n_absence}")
        
        # Verify minimum requirements
        if n_presence < 2 or n_absence < 2:
            st.error(f"""
            ⚠️ **Erro**: Dados insuficientes para modelagem!
            
            **Mínimo necessário:** 2 amostras de cada classe
            **Atual:** {n_presence} presenças, {n_absence} ausências
            
            **Soluções:**
            1. Adicione mais pontos de ocorrência se houver poucas presenças
            2. Gere mais pseudo-ausências se houver poucas ausências
            3. Verifique se os dados foram processados corretamente
            """)
            return
        
        if n_presence < 10 or n_absence < 10:
            st.warning("""
            ⚠️ **Aviso**: Poucos dados para um modelo robusto!
            
            Recomenda-se ter pelo menos 10 amostras de cada classe.
            Os resultados podem não ser confiáveis com poucos dados.
            """)
        
        # Store prepared data
        st.session_state['X_model'] = X
        st.session_state['y_model'] = y
        st.session_state['selected_vars'] = selected_vars
        st.session_state['bioclim_data'] = bioclim_data  # Store full data for later use
        
        st.success("✅ Dados preparados para modelagem!")
    
    with tab2:
        st.header("Treinamento do Modelo")
        
        # Check if species changed
        if species_changed:
            st.error("⚠️ A espécie mudou. Por favor, prepare os dados novamente na aba anterior.")
            return
        
        if 'X_model' not in st.session_state:
            st.warning("⚠️ Prepare os dados na aba anterior primeiro.")
            return
        
        # Verify that we have current data
        if 'bioclim_data' in st.session_state and 'selected_vars' in st.session_state:
            current_vars = st.session_state['selected_vars']
            bioclim_data = st.session_state['bioclim_data']
            
            # Check if selected variables exist in current data
            missing_vars = [var for var in current_vars if var not in bioclim_data.columns]
            if missing_vars:
                st.error(f"⚠️ Variáveis selecionadas não encontradas nos dados atuais: {', '.join(missing_vars)}")
                st.warning("Por favor, volte à aba de Preparação de Dados e reselecione as variáveis.")
                return
        
        X = st.session_state['X_model']
        y = st.session_state['y_model']
        
        # Check class balance
        unique_classes, class_counts = np.unique(y, return_counts=True)
        st.write("Distribuição das classes:")
        for cls, count in zip(unique_classes, class_counts):
            class_name = "Presença" if cls == 1 else "Ausência"
            st.write(f"- {class_name}: {count} amostras")
        
        # Check minimum samples per class
        min_samples_per_class = min(class_counts)
        if min_samples_per_class < 2:
            st.error(f"""
            ⚠️ **Erro**: Dados insuficientes para treinamento!
            
            A classe com menos amostras tem apenas {min_samples_per_class} exemplo(s).
            São necessários pelo menos 2 exemplos de cada classe.
            
            **Soluções:**
            1. Adicione mais pontos de ocorrência da espécie
            2. Gere mais pseudo-ausências
            3. Verifique se os dados foram carregados corretamente
            """)
            return
        
        if min_samples_per_class < n_folds:
            st.warning(f"""
            ⚠️ **Aviso**: Poucos dados para validação cruzada!
            
            A classe com menos amostras tem apenas {min_samples_per_class} exemplos,
            mas foram solicitados {n_folds} folds para validação cruzada.
            
            Reduzindo automaticamente para {min_samples_per_class} folds.
            """)
            n_folds = min_samples_per_class
        
        # Model initialization parameters
        
        # Train model
        if st.button("Treinar Modelo", type="primary"):
            with st.spinner("Treinando o modelo..."):
                
                # Initialize model with only random_state
                sdm_model = SDMModel(random_state=random_state)
                
                if use_cross_validation:
                    # Cross-validation training
                    if cv_type == "Espacial (Recomendada para SDM)":
                        # Check if we have coordinates
                        if 'latitude' not in bioclim_data.columns or 'longitude' not in bioclim_data.columns:
                            st.error("⚠️ Coordenadas não encontradas. Usando validação estratificada.")
                            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
                            coordinates = None
                        else:
                            # Extract coordinates
                            coordinates = bioclim_data[['longitude', 'latitude']].values
                            # Convert buffer distance from km to degrees (approximate)
                            buffer_degrees = buffer_distance / 111.0  # 1 degree ≈ 111 km
                            cv = SpatialKFold(n_splits=n_folds, buffer_distance=buffer_degrees)
                            st.info(f"📍 Usando validação cruzada espacial com {len(coordinates)} pontos")
                    else:
                        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
                        coordinates = None
                    
                    cv_scores = {
                        'accuracy': [],
                        'precision': [],
                        'recall': [],
                        'f1': [],
                        'auc': []
                    }
                    
                    # Translation dictionary for metrics
                    metric_translations = {
                        'accuracy': 'Acurácia',
                        'precision': 'Precisão',
                        'recall': 'Sensibilidade',
                        'f1': 'F1-Score',
                        'auc': 'AUC-ROC'
                    }
                    
                    progress_bar = st.progress(0)
                    
                    # Get splits based on CV type
                    if coordinates is not None and cv_type == "Espacial (Recomendada para SDM)":
                        splits = list(cv.split(X, y, coordinates=coordinates))
                    else:
                        splits = list(cv.split(X, y))
                    
                    for fold, (train_idx, val_idx) in enumerate(splits):
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        
                        # Log fold information for spatial CV
                        if cv_type == "Espacial (Recomendada para SDM)" and coordinates is not None:
                            st.text(f"Fold {fold+1}: {len(train_idx)} treino, {len(val_idx)} validação")
                        
                        # Train on fold with hyperparameters
                        sdm_model.train(X_train, y_train, 
                                      n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf)
                        
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
                    sdm_model.train(X, y, 
                                  n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf)
                    
                    # Show CV results
                    cv_title = "Resultados da Validação Cruzada Espacial" if cv_type == "Espacial (Recomendada para SDM)" else "Resultados da Validação Cruzada Estratificada"
                    st.subheader(cv_title)
                    
                    cv_results = pd.DataFrame(cv_scores)
                    cv_summary = cv_results.describe()
                    
                    # Create translated version for display
                    cv_results_display = cv_results.copy()
                    cv_results_display.columns = [metric_translations.get(col, col) for col in cv_results_display.columns]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(cv_results_display)
                    
                    with col2:
                        # Plot CV results with translated labels
                        fig, ax = plt.subplots(figsize=(8, 6))
                        cv_results_display.boxplot(ax=ax)
                        ax.set_ylabel('Valor')
                        ax.set_title('Distribuição das Métricas - Validação Cruzada')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Show mean scores
                    st.subheader("Métricas Médias")
                    mean_scores = cv_results.mean()
                    
                    cols = st.columns(len(mean_scores))
                    for i, (metric, score) in enumerate(mean_scores.items()):
                        with cols[i]:
                            metric_label = metric_translations.get(metric, metric.title())
                            st.metric(metric_label, f"{score:.3f}")
                    
                else:
                    # Simple train-test split
                    from sklearn.model_selection import train_test_split
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=random_state, stratify=y
                    )
                    
                    # Train model with hyperparameters
                    sdm_model.train(X_train, y_train,
                                  n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf)
                    
                    # Predict on test set
                    y_pred = sdm_model.predict(X_test)
                    y_proba = sdm_model.predict_proba(X_test)[:, 1]
                    
                    # Calculate metrics
                    test_metrics = {
                        'Acurácia': accuracy_score(y_test, y_pred),
                        'Precisão': precision_score(y_test, y_pred),
                        'Sensibilidade': recall_score(y_test, y_pred),
                        'F1-Score': f1_score(y_test, y_pred),
                        'AUC-ROC': roc_auc_score(y_test, y_proba)
                    }
                    
                    # Show results
                    st.subheader("Resultados no Conjunto de Teste")
                    
                    with st.expander("📚 Por que usar conjunto de teste?", expanded=False):
                        st.markdown("""
                        ### Conceito Fundamental em Machine Learning
                        
                        Para avaliar se um modelo realmente aprendeu padrões gerais (e não apenas memorizou os dados), 
                        dividimos os dados em:
                        
                        - **Conjunto de Treinamento (80%)**: Usado para o modelo aprender
                        - **Conjunto de Teste (20%)**: Usado para avaliar o modelo em dados "novos"
                        
                        As métricas do conjunto de teste são mais realistas porque:
                        - O modelo nunca viu esses dados durante o aprendizado
                        - Simulam o desempenho em dados reais futuros
                        - Revelam se há overfitting (memorização excessiva)
                        
                        💡 **Dica**: Se a accuracy no teste for muito menor que no treinamento, 
                        pode indicar overfitting!
                        """)
                    
                    cols = st.columns(len(test_metrics))
                    for i, (metric, score) in enumerate(test_metrics.items()):
                        with cols[i]:
                            st.metric(metric, f"{score:.3f}")
                    
                    # Store test data for threshold calculation in projection
                    st.session_state['y_test'] = y_test
                    st.session_state['y_pred_proba'] = y_proba
                
                # Feature importance
                st.subheader("Importância das Variáveis")
                
                # Get feature importances from the Random Forest classifier inside the pipeline
                rf_classifier = sdm_model.model.named_steps['classifier']
                feature_importance = pd.DataFrame({
                    'variable': X.columns,
                    'importance': rf_classifier.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Add translated labels
                feature_importance['variable_label'] = feature_importance['variable'].apply(
                    lambda x: format_bioclim_var(x)
                )
                
                # Plot feature importance with translated labels
                fig = px.bar(feature_importance, x='importance', y='variable_label', 
                           orientation='h', title='Importância das Variáveis')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Store model and results
                st.session_state['trained_model'] = sdm_model
                st.session_state['feature_importance'] = feature_importance
                st.session_state['model_trained'] = True
                st.session_state['model_species'] = st.session_state.get('species_name', 'Unknown')
                
                st.success("✅ Modelo treinado com sucesso!")
                st.info(f"Modelo treinado para: {st.session_state.get('species_name', 'espécie não identificada')}")
                
                # Add metrics glossary
                with st.expander("📊 Entenda as métricas"):
                    st.markdown("""
                    - **Acurácia**: Proporção de predições corretas (presença e ausência)
                    - **Precisão**: Das predições de presença, quantas estavam corretas
                    - **Sensibilidade (Recall)**: Das presenças reais, quantas foram detectadas
                    - **F1-Score**: Média harmônica entre Precisão e Sensibilidade
                    - **AUC-ROC**: Área sob a curva ROC (0.5 = aleatório, 1.0 = perfeito)
                    
                    💡 Para SDM, **Sensibilidade** alta é importante para não perder áreas de ocorrência!
                    """)
    
    with tab3:
        st.header("Avaliação do Modelo")
        
        # Check if species changed
        if species_changed:
            st.error("⚠️ A espécie mudou. Por favor, retreine o modelo.")
            return
        
        if not st.session_state.get('model_trained'):
            st.warning("⚠️ Treine o modelo na aba anterior primeiro.")
            return
        
        sdm_model = st.session_state['trained_model']
        
        # Get the data used for training
        trained_vars = st.session_state.get('selected_vars', [])
        
        # Check if we have valid data
        if not trained_vars:
            st.error("Variáveis de treinamento não encontradas. Por favor, retreine o modelo.")
            return
        
        # Get original bioclim data to use the same features
        if 'bioclim_data' not in st.session_state:
            st.error("Dados bioclimáticos não encontrados. Por favor, volte à aba de Preparação de Dados.")
            return
            
        bioclim_data = st.session_state['bioclim_data']
        
        # Check if the current bioclim data has the required variables
        missing_vars = [var for var in trained_vars if var not in bioclim_data.columns]
        
        if missing_vars:
            st.error("⚠️ As variáveis bioclimáticas usadas no modelo não estão disponíveis nos dados atuais!")
            st.write(f"❌ Variáveis faltantes: {', '.join(missing_vars)}")
            st.warning("Isso ocorre quando você muda de espécie. Por favor, volte à aba 'Preparação de Dados' e retreine o modelo.")
            
            # Clear the model trained flag to force retraining
            st.session_state['model_trained'] = False
            return
        
        # Use only the variables that were used during training
        X = bioclim_data[trained_vars]
        y = (bioclim_data['point_type'] == 'presence').astype(int)
        
        st.info(f"Usando as mesmas {len(trained_vars)} variáveis do treinamento: {', '.join(trained_vars)}")
        
        if 'model_species' in st.session_state:
            st.info(f"Modelo treinado para: {st.session_state['model_species']}")
        
        # Make predictions on full dataset for visualization
        y_pred = sdm_model.predict(X)
        y_proba = sdm_model.predict_proba(X)[:, 1]
        
        # Overall metrics
        st.subheader("Métricas Gerais")
        
        with st.expander("📚 Por que as métricas são diferentes do conjunto de teste?", expanded=True):
            st.markdown("""
            ### Diferença entre Métricas
            
            Você pode notar que as métricas aqui são geralmente **maiores** que no conjunto de teste. 
            Isso é normal e esperado!
            
            **Aqui (Avaliação Geral):**
            - Usamos 100% dos dados (treinamento + teste)
            - O modelo já "conhece" 80% desses dados
            - Acurácia tipicamente maior (~0.93)
            
            **Conjunto de Teste:**
            - Apenas 20% dos dados (nunca vistos)
            - Métrica mais realista
            - Acurácia tipicamente menor (~0.82)
            
            ### Qual métrica usar?
            
            - **Para publicações científicas**: Use as métricas do conjunto de teste
            - **Para entender o modelo**: Analise ambas
            - **Para projeções espaciais**: O modelo usa todo o conhecimento adquirido
            
            💡 **Interpretação**: Uma diferença pequena entre as métricas 
            (como 0.82 vs 0.93) indica que o modelo generaliza bem!
            """)
        
        overall_metrics = {
            'Acurácia': accuracy_score(y, y_pred),
            'Precisão': precision_score(y, y_pred),
            'Sensibilidade': recall_score(y, y_pred),
            'F1-Score': f1_score(y, y_pred),
            'AUC-ROC': roc_auc_score(y, y_proba)
        }
        
        cols = st.columns(len(overall_metrics))
        for i, (metric, score) in enumerate(overall_metrics.items()):
            with cols[i]:
                st.metric(metric, f"{score:.3f}")
        
        # Metrics explanation
        with st.expander("📊 O que significam essas métricas?", expanded=False):
            st.markdown("""
            ### Glossário de Métricas
            
            - **Acurácia**: Porcentagem de predições corretas (presença + ausência)
            - **Precisão**: Quando o modelo diz "presença", quantas vezes está certo?
            - **Sensibilidade**: Das presenças reais, quantas o modelo encontrou?
            - **F1-Score**: Média harmônica entre Precisão e Sensibilidade
            - **AUC-ROC**: Área sob a curva ROC (0.5 = aleatório, 1.0 = perfeito)
            
            ### Para SDM (Modelagem de Distribuição de Espécies):
            
            - **AUC > 0.7**: Modelo aceitável
            - **AUC > 0.8**: Modelo bom
            - **AUC > 0.9**: Modelo excelente
            
            ⚠️ **Importante**: Em SDM, o Recall é especialmente importante 
            porque queremos capturar todas as áreas onde a espécie pode ocorrer.
            """)
        
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
        
        # Response curves
        st.subheader("Curvas de Resposta")
        
        # Select variables for response curves
        # Filter feature importance to only include variables that exist in current data
        available_for_curves = [var for var in trained_vars if var in X.columns]
        default_vars = [var for var in feature_importance.head(5)['variable'].tolist() if var in available_for_curves]
        
        selected_vars_response = st.multiselect(
            "Selecione variáveis para visualizar curvas de resposta",
            options=available_for_curves,
            default=default_vars
        )
        
        if selected_vars_response:
            # Create response curves
            n_cols = min(3, len(selected_vars_response))
            n_rows = (len(selected_vars_response) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for idx, var in enumerate(selected_vars_response):
                ax = axes[idx]
                
                # Create range of values for the variable
                var_values = np.linspace(X[var].min(), X[var].max(), 100)
                
                # Create mean values for other variables
                mean_values = X.mean()
                
                # Create prediction dataframe
                pred_df = pd.DataFrame([mean_values] * 100)
                pred_df[var] = var_values
                
                # Make predictions using the correct variable names (same as training)
                predictions = sdm_model.predict_proba(pred_df[trained_vars])[:, 1]
                
                # Plot
                ax.plot(var_values, predictions, 'b-', linewidth=2)
                ax.set_xlabel(var)
                ax.set_ylabel('Probabilidade')
                ax.set_title(f'Curva de Resposta - {var}')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
            
            # Remove empty subplots
            for idx in range(len(selected_vars_response), len(axes)):
                fig.delaxes(axes[idx])
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show partial dependence analysis
            st.info("💡 As curvas de resposta mostram como a probabilidade de presença varia com cada variável, mantendo as outras variáveis em seus valores médios.")
    
    with tab4:
        st.header("Salvar e Carregar Modelos")
        
        # Save model section
        st.subheader("Salvar Modelo")
        
        if st.session_state.get('model_trained'):
            save_name = st.text_input(
                "Nome do arquivo",
                value=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
            
            col1, col2 = st.columns(2)
            
            # Prepare model data
            try:
                import joblib
                import io
                
                # Serialize model to bytes using joblib (more robust than pickle)
                model_buffer = io.BytesIO()
                joblib.dump(st.session_state['trained_model'], model_buffer)
                model_buffer.seek(0)
                model_data = model_buffer.getvalue()
                
                # Prepare metadata
                metadata = {
                    'model_name': save_name,
                    'saved_date': datetime.now().isoformat(),
                    'selected_vars': st.session_state['selected_vars'],
                    'n_samples': len(st.session_state['X_model']),
                    'feature_importance': st.session_state['feature_importance'].to_dict(),
                    'model_type': 'Random Forest SDM',
                    'version': '1.0'
                }
                
                import json
                metadata_json = json.dumps(metadata, indent=2)
                
                with col1:
                    # Download model button
                    st.download_button(
                        label="📥 Baixar Modelo (.pkl)",
                        data=model_data,
                        file_name=f"{save_name}.pkl",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                
                with col2:
                    # Download metadata button
                    st.download_button(
                        label="📄 Baixar Metadados (.json)",
                        data=metadata_json,
                        file_name=f"{save_name}_metadata.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"Erro ao preparar arquivos para download: {e}")
            
            st.info("💡 Dica: Salve tanto o modelo quanto os metadados para ter informações completas sobre o modelo treinado.")
        else:
            st.info("⚠️ Treine um modelo primeiro antes de salvar.")
        
        # Load model section
        st.subheader("Carregar Modelo")
        
        uploaded_model = st.file_uploader(
            "Selecione um arquivo de modelo (.pkl)",
            type=['pkl'],
            help="Carregue um modelo Random Forest SDM previamente treinado"
        )
        
        uploaded_metadata = st.file_uploader(
            "Selecione um arquivo de metadados (.json) - Opcional",
            type=['json'],
            help="Carregue os metadados associados ao modelo para ver informações sobre as variáveis usadas"
        )
        
        if uploaded_model is not None:
            if st.button("Carregar Modelo", type="primary"):
                try:
                    # Load model from uploaded file
                    import joblib
                    loaded_model = joblib.load(uploaded_model)
                    
                    # Store loaded model in session state
                    st.session_state['loaded_model'] = loaded_model
                    
                    # Load metadata if provided
                    if uploaded_metadata is not None:
                        import json
                        metadata = json.load(uploaded_metadata)
                        
                        st.write("Informações do Modelo:")
                        st.json(metadata)
                        
                        # If metadata contains selected_vars, update session state
                        if 'selected_vars' in metadata:
                            st.session_state['selected_vars'] = metadata['selected_vars']
                        if 'feature_importance' in metadata:
                            # Convert feature importance back to DataFrame
                            import pandas as pd
                            fi_dict = metadata['feature_importance']
                            st.session_state['feature_importance'] = pd.DataFrame(fi_dict)
                    
                    st.session_state['trained_model'] = loaded_model
                    st.session_state['model_trained'] = True
                    
                    st.success("✅ Modelo carregado com sucesso!")
                    st.info("💡 Você pode usar este modelo carregado para fazer previsões na aba de Avaliação.")
                except Exception as e:
                    st.error(f"Erro ao carregar modelo: {e}")

if __name__ == "__main__":
    render_page()