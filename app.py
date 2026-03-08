import streamlit as st
import sys
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def load_config():
    """Load module unlock configuration"""
    config_path = Path(__file__).parent / "config.json"
    try:
        with open(config_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"unlocked_modules": 6}  # Default: all unlocked

APP_CONFIG = load_config()

# Import page modules
from pages import pagina_busca_api, pagina_pseudoausencias, pagina_analise_bioclimatica, pagina_modelagem, pagina_projecao_espacial, pagina_projecao_futura

# Page configuration
st.set_page_config(
    page_title="TAIPA - Tecnologia Aplicada para Pesquisa Ambiental | SDM",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",  # Collapsed by default for cleaner interface
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Custom CSS for better aesthetics and navigation
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2E7D32;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #5D4E37;
        padding-bottom: 2rem;
    }
    .nav-button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 2rem;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    .step-indicator {
        text-align: center;
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .progress-container {
        background-color: #e0e0e0;
        border-radius: 25px;
        padding: 3px;
        margin: 20px 0;
    }
    .progress-bar {
        background-color: #4CAF50;
        height: 30px;
        border-radius: 25px;
        text-align: center;
        line-height: 30px;
        color: white;
        font-weight: bold;
        transition: width 0.3s ease;
    }
    .workflow-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 20px 0;
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    .workflow-step {
        text-align: center;
        flex: 1;
        padding: 10px;
    }
    .workflow-arrow {
        color: #ccc;
        font-size: 24px;
    }
    .workflow-step.completed {
        color: #4CAF50;
        font-weight: bold;
    }
    .workflow-step.current {
        color: #2196F3;
        font-weight: bold;
        background-color: #e3f2fd;
        border-radius: 10px;
    }
    .workflow-step.locked {
        color: #999;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0

# Define pages
pages = [
    {
        "title": "Início",
        "icon": "🏠",
        "function": None,
        "required_state": [],
        "sets_state": []
    },
    {
        "title": "Busca de Espécies",
        "icon": "🔍",
        "function": pagina_busca_api,
        "required_state": [],
        "sets_state": ["species_name", "occurrence_data"]
    },
    {
        "title": "Pseudo-ausências",
        "icon": "📍",
        "function": pagina_pseudoausencias,
        "required_state": ["occurrence_data"],
        "sets_state": ["pseudo_absences"]
    },
    {
        "title": "Análise Bioclimática",
        "icon": "🌡️",
        "function": pagina_analise_bioclimatica,
        "required_state": ["occurrence_data", "pseudo_absences"],
        "sets_state": ["selected_bioclim_vars", "bioclim_data"]
    },
    {
        "title": "Modelagem",
        "icon": "🤖",
        "function": pagina_modelagem,
        "required_state": ["bioclim_data"],
        "sets_state": ["model_trained", "trained_model"]
    },
    {
        "title": "Projeção Espacial",
        "icon": "🗺️",
        "function": pagina_projecao_espacial,
        "required_state": ["model_trained"],
        "sets_state": ["last_prediction"]
    },
    {
        "title": "Projeção Futura",
        "icon": "🔮",
        "function": pagina_projecao_futura,
        "required_state": ["model_trained"],
        "sets_state": ["future_projection_done"]
    }
]

# Function to check if a module is unlocked by the professor
def is_module_unlocked(page_index):
    """Check if module is unlocked via config.json"""
    if page_index == 0:
        return True  # Home always accessible
    return page_index <= APP_CONFIG.get("unlocked_modules", 6)

# Function to check if requirements are met
def check_requirements(page_index):
    """Check if all required session states exist for a page AND module is unlocked"""
    if page_index >= len(pages):
        return False
    if not is_module_unlocked(page_index):
        return False

    page = pages[page_index]
    for req in page["required_state"]:
        if req not in st.session_state or st.session_state[req] is None:
            return False
    return True

# Function to get completion status
def get_completion_status():
    """Get completion status for each step"""
    status = []
    for page in pages[1:]:  # Skip home page
        completed = all(state in st.session_state and st.session_state[state] is not None 
                       for state in page["sets_state"])
        status.append(completed)
    return status

# Navigation header with progress
current_page = st.session_state.current_page
total_pages = len(pages) - 1  # Exclude home page

# Only show top header and progress on non-home pages
if current_page > 0:
    st.markdown('<h1 class="main-header">🌿 TAIPA</h1>', unsafe_allow_html=True)

if current_page > 0:  # Not on home page
    # Custom progress bar with percentage
    progress = (current_page - 1) / (total_pages - 1) if total_pages > 1 else 0
    progress_percentage = int(progress * 100)
    
    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {progress_percentage}%">
            {progress_percentage}% Completo
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual workflow
    workflow_html = '<div class="workflow-container">'
    for i, page in enumerate(pages[1:], 1):
        completed = i < current_page
        is_current = i == current_page
        
        step_class = "completed" if completed else "current" if is_current else "locked"
        icon = "✅" if completed else page["icon"]
        
        workflow_html += f'<div class="workflow-step {step_class}">{icon}<br>{page["title"]}</div>'
        
        if i < total_pages:
            workflow_html += '<div class="workflow-arrow">→</div>'
    
    workflow_html += '</div>'
    st.markdown(workflow_html, unsafe_allow_html=True)
    
    # Step indicator
    st.markdown(f"""
    <div class="step-indicator">
        <h3>{pages[current_page]["icon"]} Etapa {current_page} de {total_pages}: {pages[current_page]["title"]}</h3>
    </div>
    """, unsafe_allow_html=True)

# Navigation controls - only show on non-home pages
if current_page > 0:
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        if st.button("⬅️ Anterior", use_container_width=True, key="prev_btn", help="Voltar para a etapa anterior"):
            st.session_state.current_page -= 1
            st.rerun()

    with col2:
        page_options = [f"{p['icon']} {p['title']}" for i, p in enumerate(pages) if i > 0]
        accessible_pages = [i for i in range(1, len(pages)) if check_requirements(i)]

        if len(accessible_pages) > 1:
            available_options = page_options[:max(accessible_pages)]
            selected_index = current_page - 1
            # Clamp index to valid range
            selected_index = min(selected_index, len(available_options) - 1)
            selected_index = max(selected_index, 0)
            selected = st.selectbox(
                "Ir para:",
                options=available_options,
                index=selected_index,
                label_visibility="collapsed",
                help="Navegue rapidamente entre etapas já acessíveis"
            )
            new_index = page_options.index(selected) + 1
            if new_index != current_page and new_index in accessible_pages:
                st.session_state.current_page = new_index
                st.rerun()

    with col3:
        next_page = current_page + 1
        if next_page < len(pages):
            if not is_module_unlocked(next_page):
                st.button("🔒 Bloqueado", use_container_width=True, disabled=True, key="next_btn")
            else:
                can_proceed = check_requirements(next_page)
                current_requirements = pages[current_page].get("sets_state", [])
                current_requirements_met = any(
                    state in st.session_state and st.session_state[state] is not None
                    for state in current_requirements
                ) if current_requirements else True

                if can_proceed or not current_requirements_met:
                    button_text = "Próximo ➡️" if current_requirements_met else "Pular ⏭️"
                    button_help = "Ir para a próxima etapa" if current_requirements_met else "Pular esta etapa (opcional)"
                    if st.button(button_text, use_container_width=True, type="primary", key="next_btn", help=button_help):
                        st.session_state.current_page = next_page
                        st.rerun()
                else:
                    st.button("Próximo ➡️", use_container_width=True, disabled=True, key="next_btn")
                    st.caption("⚠️ Complete esta etapa primeiro")

    st.markdown("---")

# Display current page content
if current_page == 0:
    # Home page with improved design
    import plotly.graph_objects as go
    
    # Enhanced CSS for home page
    st.markdown("""
    <style>
    .hero-gradient {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 2.5rem 2.5rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: bold;
        color: #1b5e20;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        font-size: 1.3rem;
        color: #388e3c;
        margin-bottom: 0.8rem;
    }
    .hero-description {
        font-size: 1rem;
        color: #4a7c4f;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    .feature-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        height: 100%;
        transition: transform 0.2s;
    }
    .feature-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    .stats-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2e7d32;
    }
    .step-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        border-left: 4px solid;
        margin-bottom: 0.5rem;
    }
    .step-card h4 { margin: 0 0 0.4rem 0; }
    .step-card p { margin: 0; font-size: 0.9rem; color: #555; }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section with integrated CTA
    st.markdown("""
    <div class="hero-gradient">
        <h1 class="hero-title">🌿 TAIPA</h1>
        <p class="hero-subtitle">Tecnologia Aplicada para Pesquisa Ambiental</p>
        <p class="hero-description">
            Plataforma educacional completa para Modelagem de Distribuição de Espécies (SDM),<br>
            oferecendo um fluxo de trabalho integrado desde a coleta de dados até projeções futuras considerando mudanças climáticas.
        </p>
        <p style="font-size: 0.85rem; color: #6a9a6e; margin: 0;">Desenvolvido por <b>Pedro Higuchi</b></p>
    </div>
    """, unsafe_allow_html=True)

    # Check overall progress
    completion_status = get_completion_status()
    completed_steps = sum(completion_status)

    # CTA Section - right after hero, compact and centered
    if completed_steps == 0:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Iniciar Análise", type="primary", use_container_width=True):
                st.session_state.current_page = 1
                st.rerun()

    elif completed_steps < len(completion_status):
        next_step_idx = completed_steps + 1
        next_step = pages[next_step_idx] if next_step_idx < len(pages) else None

        # Compact progress + stats for returning users
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Espécie", st.session_state.get('species_name', '-'))
        with col2:
            n_pts = len(st.session_state.get('occurrence_data', [])) if 'occurrence_data' in st.session_state else 0
            st.metric("Ocorrências", n_pts)
        with col3:
            n_vars = len(st.session_state.get('selected_bioclim_vars', [])) if 'selected_bioclim_vars' in st.session_state else 0
            st.metric("Variáveis", n_vars)
        with col4:
            st.metric("Progresso", f"{completed_steps}/{len(completion_status)}")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(f"➡️ Continuar: {next_step['icon']} {next_step['title']}", type="primary", use_container_width=True):
                st.session_state.current_page = next_step_idx
                st.rerun()

    else:
        st.success("**Análise completa!** Todos os 6 passos foram concluídos.")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 Ver Resultados", use_container_width=True):
                st.session_state.current_page = 5
                st.rerun()
        with col2:
            if st.button("🔮 Ver Projeções", use_container_width=True):
                st.session_state.current_page = 6
                st.rerun()
        with col3:
            if st.button("🔄 Nova Análise", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key != 'current_page':
                        del st.session_state[key]
                st.session_state.current_page = 0
                st.rerun()
    
    st.markdown("---")

    # Funcionalidades + Fluxo combined in a cleaner layout
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### ✨ Funcionalidades")
        st.markdown("""
        <div class="feature-box">
            <p style="line-height: 2;">
                🌍 Integração com <b>GBIF</b> para busca de ocorrências<br>
                🗺️ Mapas interativos para visualização e filtragem<br>
                🌡️ <b>19 variáveis bioclimáticas</b> do WorldClim<br>
                🤖 Machine Learning com <b>Random Forest</b> otimizado<br>
                📊 <b>Validação Cruzada Espacial</b> para métricas realistas<br>
                🔮 Projeções futuras com cenários <b>SSP1-2.6</b> e <b>SSP5-8.5</b><br>
                💾 Gerenciamento de modelos com save/load<br>
                🚀 Interface intuitiva com navegação guiada
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("#### 🎓 Fluxo de Trabalho")
        steps = [
            ("🔍", "1. Coleta de Dados", "Busca de ocorrências no GBIF", "#1976D2"),
            ("📍", "2. Pseudo-ausências", "Geração de pontos de background", "#388E3C"),
            ("🌡️", "3. Análise Bioclimática", "Extração e seleção de variáveis", "#D32F2F"),
            ("🤖", "4. Modelagem", "Random Forest + validação espacial", "#7B1FA2"),
            ("🗺️", "5. Projeção Espacial", "Mapa de adequabilidade atual", "#E65100"),
            ("🔮", "6. Projeção Futura", "Cenários de mudanças climáticas", "#00695C"),
        ]
        unlocked_count = APP_CONFIG.get("unlocked_modules", 6)
        for idx, (icon, title, desc, color) in enumerate(steps, 1):
            if idx <= unlocked_count:
                st.markdown(f"""
                <div class="step-card" style="border-left-color: {color};">
                    <h4>{icon} {title}</h4>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="step-card" style="border-left-color: #ccc; opacity: 0.5;">
                    <h4>🔒 {title}</h4>
                    <p><em>Será liberado em aula futura</em></p>
                </div>
                """, unsafe_allow_html=True)

    # Resources section
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("💡 Dicas para melhores resultados"):
        st.markdown("""
        1. **Qualidade dos dados**: Sempre revise os pontos de ocorrência no mapa antes de prosseguir
        2. **Pseudo-ausências**: Use pelo menos o dobro do número de presenças
        3. **Variáveis**: Selecione variáveis biologicamente relevantes para sua espécie
        4. **Validação**: A validação cruzada espacial é essencial para evitar superestimação
        5. **Interpretação**: Considere o conhecimento biológico ao interpretar os mapas
        """)

    with st.expander("📚 Saiba mais sobre SDM"):
        st.markdown("""
        **SDM (Species Distribution Modeling)** é uma técnica que combina:
        - Dados de ocorrência de espécies
        - Variáveis ambientais (clima, topografia, etc.)
        - Algoritmos estatísticos/machine learning

        Para prever onde uma espécie pode ocorrer baseado em suas preferências ambientais.

        **Aplicações:**
        - Conservação da biodiversidade
        - Avaliação de impactos das mudanças climáticas
        - Identificação de áreas prioritárias
        - Prevenção de invasões biológicas
        """)

else:
    # Regular pages
    page = pages[current_page]

    # Check if module is locked by professor
    if not is_module_unlocked(current_page):
        st.warning(f"""
        🔒 **Módulo bloqueado**

        O módulo **{page['icon']} {page['title']}** será liberado pelo professor em uma aula futura.
        """)
        if st.button("⬅️ Voltar ao Início", type="primary"):
            st.session_state.current_page = 0
            st.rerun()

    # Check requirements
    elif not check_requirements(current_page):
        missing_states = [req for req in page['required_state'] if req not in st.session_state or st.session_state[req] is None]
        
        # Map technical names to user-friendly descriptions
        state_descriptions = {
            "species_name": "Nome da espécie",
            "occurrence_data": "Dados de ocorrência",
            "pseudo_absences": "Pontos de pseudo-ausência",
            "selected_bioclim_vars": "Variáveis bioclimáticas selecionadas",
            "bioclim_data": "Dados bioclimáticos extraídos",
            "model_trained": "Modelo treinado",
            "trained_model": "Arquivo do modelo"
        }
        
        missing_descriptions = [state_descriptions.get(state, state) for state in missing_states]
        
        st.error(f"""
        ⚠️ **Etapa anterior incompleta!**
        
        Esta etapa requer que você complete as seguintes informações primeiro:
        
        • {' • '.join(missing_descriptions)}
        """)
        
        # Find which step provides the missing data
        for i, p in enumerate(pages[1:current_page], 1):
            if any(state in p["sets_state"] for state in missing_states):
                if st.button(f"Ir para: {p['icon']} {p['title']}", type="primary", key="go_back"):
                    st.session_state.current_page = i
                    st.rerun()
                break
    else:
        # Display the page content
        if page["function"]:
            try:
                page["function"]()
            except Exception as e:
                st.error(f"Erro ao carregar a página: {str(e)}")
                st.info("Tente voltar para a etapa anterior e completá-la novamente.")

# Sidebar with overview (collapsible)
with st.sidebar:
    st.markdown("### 🧭 Visão Geral do Projeto")
    
    # Project summary
    if "species_name" in st.session_state and st.session_state.species_name:
        st.info(f"**Espécie:** {st.session_state.species_name}")
    
    # Show all steps with status
    st.markdown("#### Progresso:")
    for i, page in enumerate(pages):
        if i == 0:
            continue  # Skip home
            
        # Check if completed
        completed = all(state in st.session_state and st.session_state[state] is not None 
                       for state in page["sets_state"])
        
        # Check if accessible
        accessible = check_requirements(i)
        
        unlocked = is_module_unlocked(i)

        # Check if completed
        completed = completed and unlocked

        # Check if accessible
        accessible = check_requirements(i)

        if not unlocked:
            st.button(f"🔒 {page['title']}", key=f"nav_{i}", use_container_width=True, disabled=True, help="Será liberado em aula futura")
        elif i == current_page:
            st.markdown(f"**→ {page['icon']} {page['title']}** (atual)")
        elif completed:
            if st.button(f"✅ {page['title']}", key=f"nav_{i}", use_container_width=True, help="Etapa concluída - clique para revisar"):
                st.session_state.current_page = i
                st.rerun()
        elif accessible:
            if st.button(f"⭕ {page['title']}", key=f"nav_{i}", use_container_width=True, help="Etapa disponível - clique para acessar"):
                st.session_state.current_page = i
                st.rerun()
        else:
            missing = [req for req in page["required_state"] if req not in st.session_state or st.session_state[req] is None]
            button_help = f"Requer: {', '.join(missing[:2])}{'...' if len(missing) > 2 else ''}"
            st.button(f"🔒 {page['title']}", key=f"nav_{i}", use_container_width=True, disabled=True, help=button_help)
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("#### Ações Rápidas:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏠 Início", use_container_width=True, help="Voltar ao início"):
            st.session_state.current_page = 0
            st.rerun()
    
    with col2:
        if st.button("🔄 Novo", use_container_width=True, help="Iniciar novo projeto"):
            if st.session_state.get('confirm_reset', False):
                for key in list(st.session_state.keys()):
                    if key not in ['current_page', 'confirm_reset']:
                        del st.session_state[key]
                st.session_state.current_page = 0
                st.session_state.confirm_reset = False
                st.rerun()
            else:
                st.session_state.confirm_reset = True
                st.warning("Clique novamente para confirmar")
    
    # Help section
    with st.expander("❓ Ajuda"):
        st.markdown("""
        **Navegação:**
        - Use os botões **Anterior/Próximo** para navegar
        - Clique nas etapas concluídas na barra lateral para revisar
        - O progresso é salvo automaticamente
        
        **Ícones:**
        - ✅ Concluído
        - ⭕ Disponível
        - 🔒 Bloqueado
        - → Etapa atual
        """)