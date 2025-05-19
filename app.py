import streamlit as st
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import page modules
from pages import pagina_busca_api, pagina_pseudoausencias, pagina_analise_bioclimatica, pagina_modelagem, pagina_projecao_espacial

# Page configuration
st.set_page_config(
    page_title="TAIPA - Tecnologia Aplicada para Pesquisa Ambiental",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E7D32;
        padding: 2rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #5D4E37;
        padding-bottom: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌿 TAIPA SDM</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Species Distribution Modeling Educational Platform</p>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "Home",
        "1. Busca de Espécies (GBIF)",
        "2. Pseudo-ausências",
        "3. Análise Bioclimática",
        "4. Modelagem e Resultados",
        "5. Projeção Espacial"
    ]
)

# Page content routing
if page == "Home":
    st.header("Welcome to TAIPA SDM Platform")
    
    # Overview
    st.markdown("""
    ### About TAIPA
    TAIPA (Tecnologia Aplicada para Pesquisa Ambiental) is an educational platform for Species Distribution Modeling (SDM). 
    This tool guides users through the complete workflow of creating distribution models for any species.
    
    ### 🚀 Workflow Overview
    """)
    
    # Workflow steps
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Species Data Collection**
        - Search for species occurrences using GBIF
        - Visualize distribution points on interactive maps
        - Filter and clean occurrence data
        
        **2. Pseudo-absence Generation**
        - Generate background points using buffer strategy
        - Control sampling density and exclusion zones
        - Balance presence/absence data
        """)
    
    with col2:
        st.markdown("""
        **3. Bioclimatic Analysis**
        - Extract environmental variables from WorldClim
        - Analyze variable correlations and VIF
        - Select optimal variables for modeling
        
        **4. Model Training & Evaluation**
        - Train Random Forest models
        - Cross-validation and performance metrics
        - Save and load models for future use
        
        **5. Spatial Projection**
        - Generate habitat suitability maps
        - Apply optimal thresholds
        - Export results as GeoTIFF
        """)
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Modules", "5")
    with col2:
        st.metric("SDM Workflow", "Complete")
    with col3:
        st.metric("Platform Status", "Active")
    
    # Getting started
    st.markdown("---")
    st.subheader("🎯 Getting Started")
    st.info("""
    1. Begin with **Module 1** to search for species occurrence data
    2. Follow the workflow sequentially through each module
    3. Use the sidebar to navigate between modules
    4. All data is automatically passed between modules
    """)
    
    # Features
    st.subheader("✨ Key Features")
    features = {
        "🌍 GBIF Integration": "Access global biodiversity data",
        "🗺️ Interactive Maps": "Visualize and filter occurrence points",
        "🌡️ Environmental Variables": "19 WorldClim bioclimatic layers",
        "🤖 Machine Learning": "Random Forest with cross-validation",
        "📊 Model Evaluation": "Comprehensive performance metrics",
        "💾 Model Persistence": "Save and load trained models"
    }
    
    for icon_title, description in features.items():
        st.markdown(f"**{icon_title}**: {description}")

elif page == "1. Busca de Espécies (GBIF)":
    pagina_busca_api.render_page()

elif page == "2. Pseudo-ausências":
    pagina_pseudoausencias.render_page()

elif page == "3. Análise Bioclimática":
    pagina_analise_bioclimatica.render_page()

elif page == "4. Modelagem e Resultados":
    pagina_modelagem.render_page()

elif page == "5. Projeção Espacial":
    pagina_projecao_espacial.render_page()