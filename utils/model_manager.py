import os
import json
import joblib
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import streamlit as st

class ModelManager:
    """Manage saved SDM models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def list_models(self) -> List[Dict]:
        """List all saved models with metadata"""
        models = []
        
        if not os.path.exists(self.models_dir):
            return models
        
        for file in os.listdir(self.models_dir):
            if file.endswith('.joblib'):
                model_info = {
                    'filename': file,
                    'name': file.replace('.joblib', ''),
                    'path': os.path.join(self.models_dir, file),
                    'size': os.path.getsize(os.path.join(self.models_dir, file)) / 1024 / 1024,  # MB
                    'modified': datetime.fromtimestamp(os.path.getmtime(os.path.join(self.models_dir, file)))
                }
                
                # Try to load metadata
                metadata_path = os.path.join(self.models_dir, file.replace('.joblib', '_metadata.json'))
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            model_info.update(metadata)
                    except:
                        pass
                
                models.append(model_info)
        
        return sorted(models, key=lambda x: x['modified'], reverse=True)
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a saved model"""
        try:
            model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
            metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
            
            if os.path.exists(model_path):
                os.remove(model_path)
            
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            return True
        except Exception as e:
            st.error(f"Erro ao deletar modelo: {str(e)}")
            return False
    
    def export_model_info(self, model_name: str) -> Dict:
        """Export model information as dictionary"""
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
        
        info = {
            'name': model_name,
            'path': model_path,
            'exists': os.path.exists(model_path)
        }
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    info.update(metadata)
            except:
                pass
        
        return info
    
    def render_model_manager(self):
        """Render model management interface"""
        st.header("Gerenciador de Modelos")
        
        models = self.list_models()
        
        if not models:
            st.info("Nenhum modelo salvo encontrado")
            return
        
        # Create models dataframe
        df_data = []
        for model in models:
            df_data.append({
                'Nome': model['name'],
                'Espécie': model.get('species', 'Unknown'),
                'Variáveis': len(model.get('variables', [])),
                'Método': model.get('validation_method', 'Unknown'),
                'AUC': model.get('metrics', {}).get('auc_roc', 'N/A'),
                'Tamanho (MB)': f"{model['size']:.2f}",
                'Modificado': model['modified'].strftime('%Y-%m-%d %H:%M')
            })
        
        df = pd.DataFrame(df_data)
        
        # Display models table
        selected_indices = st.multiselect("Selecione modelos para gerenciar:", 
                                         df.index, 
                                         format_func=lambda x: df.iloc[x]['Nome'])
        
        if selected_indices:
            st.dataframe(df.iloc[selected_indices])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Ver Detalhes", type="primary"):
                    for idx in selected_indices:
                        model = models[idx]
                        st.subheader(f"Detalhes: {model['name']}")
                        
                        with st.expander("Informações do Modelo", expanded=True):
                            st.write(f"**Espécie:** {model.get('species', 'Unknown')}")
                            st.write(f"**Descrição:** {model.get('description', 'N/A')}")
                            st.write(f"**Variáveis:** {', '.join(model.get('variables', []))}")
                            st.write(f"**Método de Validação:** {model.get('validation_method', 'Unknown')}")
                            
                            if 'model_params' in model:
                                st.write("**Parâmetros do Modelo:**")
                                for param, value in model['model_params'].items():
                                    st.write(f"  - {param}: {value}")
                            
                            if 'metrics' in model:
                                st.write("**Métricas:**")
                                metrics = model['metrics']
                                cols = st.columns(5)
                                metric_names = ['auc_roc', 'accuracy', 'precision', 'recall', 'f1_score']
                                
                                for i, metric in enumerate(metric_names):
                                    if metric in metrics:
                                        cols[i % 5].metric(metric.upper(), f"{metrics[metric]:.3f}")
            
            with col2:
                if st.button("Carregar Modelo"):
                    # This would load the model into session state
                    st.info("Funcionalidade de carregamento será integrada com a página de modelagem")
            
            with col3:
                if st.button("Deletar", type="secondary"):
                    if st.checkbox("Confirmar exclusão"):
                        for idx in selected_indices:
                            model = models[idx]
                            if self.delete_model(model['name']):
                                st.success(f"Modelo {model['name']} deletado com sucesso!")
                        st.experimental_rerun()
        
        else:
            st.dataframe(df)
            st.info("Selecione modelos para ver mais opções")