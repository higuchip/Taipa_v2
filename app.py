import streamlit as st
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import page modules
from pages import pagina_busca_api, pagina_variaveis_ambientais, pagina_pseudoausencias

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
        "3. Variáveis Ambientais",
        "4. Statistical Analysis",
        "5. ML Modeling",
        "6. Outputs"
    ]
)

# Page content routing
if page == "Home":
    st.header("Welcome to TAIPA SDM Platform")
    
    st.markdown("""
    ### About TAIPA
    
    TAIPA (Tecnologia Aplicada para Pesquisa Ambiental) is an educational platform designed 
    to teach Species Distribution Modeling (SDM) concepts and techniques.
    
    ### Platform Modules
    
    1. **Busca de Espécies**: Search and visualize species occurrences from GBIF
    2. **Pseudo-ausências**: Generate pseudo-absence points for SDM
    3. **Variáveis Ambientais**: Analyze bioclimatic variables and their correlations
    4. **Statistical Analysis Module**: Explore statistical methods for species distribution
    5. **Machine Learning Modeling Module**: Apply ML algorithms to predict species distribution
    6. **Outputs Module**: Generate and interpret distribution maps and reports
    
    ### Getting Started
    
    Use the sidebar to navigate through the different modules. Each module contains 
    interactive tutorials, visualizations, and hands-on exercises.
    
    ### Phase 1 Features (MVP)
    
    - **GBIF Integration**: Search species occurrences with country filters
    - **Interactive Maps**: Visualize occurrences with Folium
    - **Environmental Variables**: Analyze 19 bioclimatic variables
    - **Correlation Analysis**: Calculate VIF and correlation matrices
    - **Data Export**: Download results in CSV format
    """)
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Modules", "5")
    with col2:
        st.metric("Phase 1 Complete", "2/5")
    with col3:
        st.metric("Platform Status", "Active")

elif page == "1. Busca de Espécies (GBIF)":
    pagina_busca_api.render_page()

elif page == "2. Pseudo-ausências":
    pagina_pseudoausencias.render_page()

elif page == "3. Variáveis Ambientais":
    pagina_variaveis_ambientais.render_page()

elif page == "4. Statistical Analysis":
    st.header("Module 4: Statistical Analysis for SDM")
    st.info("Explore statistical methods used in species distribution modeling")
    st.markdown("""
    ### Topics Covered:
    - Descriptive statistics
    - Correlation analysis
    - Variable selection
    - Model validation
    - Performance metrics
    """)
    st.warning("Module content to be implemented in Phase 2")

elif page == "5. ML Modeling":
    st.header("Module 5: Machine Learning for Species Distribution")
    st.info("Apply machine learning algorithms to predict species distribution")
    st.markdown("""
    ### Topics Covered:
    - Random Forest
    - Support Vector Machines
    - Neural Networks
    - Ensemble methods
    - Model optimization
    """)
    st.warning("Module content to be implemented in Phase 3")

elif page == "6. Outputs":
    st.header("Module 6: Generating and Interpreting Outputs")
    st.info("Create distribution maps and interpret modeling results")
    st.markdown("""
    ### Topics Covered:
    - Distribution map generation
    - Uncertainty visualization
    - Report generation
    - Conservation applications
    - Future projections
    """)
    st.warning("Module content to be implemented in Phase 3")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>TAIPA SDM Platform © 2025 | "
    "Developed for Environmental Research Education | Phase 1 MVP</div>",
    unsafe_allow_html=True
)