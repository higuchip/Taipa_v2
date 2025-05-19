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

# Função helper para criar tooltips
def criar_tooltip(texto):
    return f'<span title="{texto}">ℹ️</span>'

# Função para preservar o estado entre navegações
def preservar_estado():
    # Lista de chaves importantes que devem ser preservadas
    chaves_importantes = [
        'species_name', 'occurrence_data', 'n_occurrences', 'original_occurrences',
        'gbif_data', 'pseudo_absences', 'selected_vars', 'bioclim_data',
        'model_trained', 'trained_model', 'model_results', 'model_metrics',
        'last_prediction', 'projection_threshold', 'future_projection_done',
        'projection_data', 'binary_map', 'future_prediction'
    ]
    
    # Garantir que o estado persista
    for chave in chaves_importantes:
        if chave in st.session_state:
            # Apenas preserva, não sobrescreve se já existe
            pass

# Função para calcular progresso
def calcular_progresso():
    etapas = {
        'especies_buscadas': st.session_state.get('species_name', None) is not None and 
                           st.session_state.get('occurrence_data', None) is not None,
        'pseudoausencias_geradas': st.session_state.get('pseudo_absences', None) is not None,
        'analise_bioclimatica': st.session_state.get('selected_vars', None) is not None,
        'modelo_treinado': st.session_state.get('model_trained', False),
        'projecao_espacial': st.session_state.get('last_prediction', None) is not None,
        'projecao_futura': st.session_state.get('future_projection_done', False)
    }
    
    completas = sum(etapas.values())
    total = len(etapas)
    
    return completas, total, etapas

# Inicializar variáveis de sessão se necessário
if 'navigation_state' not in st.session_state:
    st.session_state.navigation_state = {}

# Preservar estado entre navegações
preservar_estado()

# Navegação na barra lateral
st.sidebar.title("Navegação")

# Exibir progresso
completas, total, etapas = calcular_progresso()
progresso = completas / total if total > 0 else 0

st.sidebar.markdown("### 📊 Seu Progresso")
st.sidebar.progress(progresso)
st.sidebar.caption(f"{completas} de {total} etapas concluídas")

# Debug - Mostrar estado atual (apenas em desenvolvimento)
with st.sidebar.expander("🔧 Debug - Estado Atual"):
    st.write("Espécie:", st.session_state.get('species_name', 'Não definida'))
    st.write("Ocorrências:", st.session_state.get('n_occurrences', 0))
    st.write("Pseudo-ausências:", 'Sim' if 'pseudo_absences' in st.session_state else 'Não')
    st.write("Modelo treinado:", st.session_state.get('model_trained', False))

# Mostrar checkmarks para etapas completas
st.sidebar.markdown("### ✅ Etapas Completas")
etapas_nomes = {
    'especies_buscadas': '1. Busca de Espécies',
    'pseudoausencias_geradas': '2. Pseudo-ausências',
    'analise_bioclimatica': '3. Análise Bioclimática',
    'modelo_treinado': '4. Modelagem',
    'projecao_espacial': '5. Projeção Espacial',
    'projecao_futura': '6. Projeção Futura'
}

for etapa, completa in etapas.items():
    if completa:
        st.sidebar.markdown(f"✅ {etapas_nomes[etapa]}")
    else:
        st.sidebar.markdown(f"⭕ {etapas_nomes[etapa]}")

st.sidebar.markdown("---")

# Criar labels para o menu com indicadores de status
menu_labels = ["Início"]
status_icons = {
    True: "✅",  # Completo
    False: "⭕"  # Pendente
}

# Adicionar status aos labels do menu
modulos = [
    ("1. Busca de Espécies (GBIF)", etapas['especies_buscadas']),
    ("2. Pseudo-ausências", etapas['pseudoausencias_geradas']),
    ("3. Análise Bioclimática", etapas['analise_bioclimatica']),
    ("4. Modelagem e Resultados", etapas['modelo_treinado']),
    ("5. Projeção Espacial", etapas['projecao_espacial']),
    ("6. Projeção Futura", etapas['projecao_futura'])
]

for nome, completo in modulos:
    menu_labels.append(f"{status_icons[completo]} {nome}")

pagina_selecionada = st.sidebar.radio(
    "Selecione o Módulo",
    menu_labels
)

# Converter de volta para o nome original do módulo
if pagina_selecionada != "Início":
    pagina = pagina_selecionada[2:]  # Remove o ícone e espaço
else:
    pagina = pagina_selecionada

# Roteamento do conteúdo da página
if pagina == "Início":
    st.header("Bem-vindo à Plataforma TAIPA SDM")
    
    # Mensagem personalizada baseada no progresso
    if completas == 0:
        st.info("""
        🌱 **Primeira vez aqui?**
        
        Comece pela Busca de Espécies (Módulo 1) para iniciar sua jornada no mundo da Modelagem de Distribuição de Espécies!
        """)
    elif completas == total:
        st.success("""
        🎉 **Parabéns! Você completou todos os módulos!**
        
        Você agora domina o fluxo completo de SDM. Que tal experimentar com uma nova espécie?
        """)
    else:
        proxima_etapa = None
        for etapa, completa in etapas.items():
            if not completa:
                proxima_etapa = etapas_nomes[etapa]
                break
        
        st.info(f"""
        🚀 **Continue sua jornada!**
        
        Você já completou {completas} de {total} etapas. 
        Próximo passo: **{proxima_etapa}**
        """)
    
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