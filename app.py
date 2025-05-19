import streamlit as st
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import page modules
from pages import pagina_busca_api, pagina_pseudoausencias, pagina_analise_bioclimatica, pagina_modelagem, pagina_projecao_espacial, pagina_projecao_futura

# Page configuration
st.set_page_config(
    page_title="TAIPA - Tecnologia Aplicada para Pesquisa Ambiental",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
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
st.markdown('<p class="sub-header">Plataforma Educacional de Modelagem de Distribuição de Espécies</p>', unsafe_allow_html=True)

# Navegação na barra lateral
st.sidebar.title("Navegação")
pagina = st.sidebar.radio(
    "Selecione o Módulo",
    [
        "Início",
        "1. Busca de Espécies (GBIF)",
        "2. Pseudo-ausências",
        "3. Análise Bioclimática",
        "4. Modelagem e Resultados",
        "5. Projeção Espacial",
        "6. Projeção Futura"
    ]
)

# Roteamento do conteúdo da página
if pagina == "Início":
    st.header("Bem-vindo à Plataforma TAIPA SDM")
    
    # Visão geral
    st.markdown("""
    ### Sobre o TAIPA
    TAIPA (Tecnologia Aplicada para Pesquisa Ambiental) é uma plataforma educacional para Modelagem de Distribuição de Espécies (SDM). 
    Esta ferramenta guia os usuários através do fluxo completo de criação de modelos de distribuição para qualquer espécie.
    
    ### 🚀 Visão Geral do Fluxo de Trabalho
    """)
    
    # Etapas do fluxo de trabalho
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Coleta de Dados de Espécies**
        - Buscar ocorrências de espécies usando GBIF
        - Visualizar pontos de distribuição em mapas interativos
        - Filtrar e limpar dados de ocorrência
        
        **2. Geração de Pseudo-ausências**
        - Gerar pontos de background usando estratégia de buffer
        - Controlar densidade de amostragem e zonas de exclusão
        - Balancear dados de presença/ausência
        """)
    
    with col2:
        st.markdown("""
        **3. Análise Bioclimática**
        - Extrair variáveis ambientais do WorldClim
        - Analisar correlações e VIF das variáveis
        - Selecionar variáveis ótimas para modelagem
        
        **4. Treinamento e Avaliação do Modelo**
        - Treinar modelos Random Forest
        - Validação cruzada e métricas de desempenho
        - Salvar e carregar modelos para uso futuro
        
        **5. Projeção Espacial**
        - Gerar mapas de adequabilidade de habitat
        - Aplicar thresholds ótimos
        - Exportar resultados como GeoTIFF
        
        **6. Projeção Futura**
        - Análise de impacto das mudanças climáticas
        - Cenários SSP1-2.6 vs SSP5-8.5
        - Projeções 2081-2100
        - Mapas de mudança e estabilidade
        - *Nota: GCM único para fins didáticos*
        """)
    
    # Estatísticas rápidas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Módulos", "6")
    with col2:
        st.metric("Fluxo SDM", "Completo")
    with col3:
        st.metric("Status da Plataforma", "Ativa")
    
    # Começando
    st.markdown("---")
    st.subheader("🎯 Como Começar")
    st.info("""
    1. Comece com o **Módulo 1** para buscar dados de ocorrência de espécies
    2. Siga o fluxo de trabalho sequencialmente através de cada módulo
    3. Use a barra lateral para navegar entre os módulos
    4. Todos os dados são automaticamente transferidos entre módulos
    """)
    
    # Recursos
    st.subheader("✨ Principais Recursos")
    recursos = {
        "🌍 Integração GBIF": "Acesso a dados globais de biodiversidade",
        "🗺️ Mapas Interativos": "Visualizar e filtrar pontos de ocorrência",
        "🌡️ Variáveis Ambientais": "19 camadas bioclimáticas WorldClim",
        "🤖 Machine Learning": "Random Forest com validação cruzada",
        "📊 Avaliação do Modelo": "Métricas abrangentes de desempenho",
        "💾 Persistência do Modelo": "Salvar e carregar modelos treinados",
        "🌡️ Projeções Climáticas": "Cenários futuros (SSP1-2.6, SSP5-8.5)"
    }
    
    for titulo_icone, descricao in recursos.items():
        st.markdown(f"**{titulo_icone}**: {descricao}")

elif pagina == "1. Busca de Espécies (GBIF)":
    pagina_busca_api()

elif pagina == "2. Pseudo-ausências":
    pagina_pseudoausencias()

elif pagina == "3. Análise Bioclimática":
    pagina_analise_bioclimatica()

elif pagina == "4. Modelagem e Resultados":
    pagina_modelagem()

elif pagina == "5. Projeção Espacial":
    pagina_projecao_espacial()

elif pagina == "6. Projeção Futura":
    pagina_projecao_futura()