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

# Function to check if requirements are met
def check_requirements(page_index):
    """Check if all required session states exist for a page"""
    if page_index >= len(pages):
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

# Header
st.markdown('<h1 class="main-header">🌿 TAIPA SDM</h1>', unsafe_allow_html=True)

# Navigation header with progress
current_page = st.session_state.current_page
total_pages = len(pages) - 1  # Exclude home page

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

# Navigation controls at the top
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    if current_page > 0:
        if st.button("⬅️ Anterior", use_container_width=True, key="prev_btn", help="Voltar para a etapa anterior"):
            st.session_state.current_page -= 1
            st.rerun()

with col2:
    # Quick navigation dropdown (always visible for better UX)
    if current_page > 0:  # Show on all pages except home
        page_options = [f"{p['icon']} {p['title']}" for i, p in enumerate(pages) if i > 0]
        accessible_pages = [i for i in range(1, len(pages)) if check_requirements(i)]
        
        if len(accessible_pages) > 1:
            selected_index = current_page - 1
            selected = st.selectbox(
                "Ir para:",
                options=page_options[:max(accessible_pages)],
                index=selected_index if selected_index < len(accessible_pages) else 0,
                label_visibility="collapsed",
                help="Navegue rapidamente entre etapas já acessíveis"
            )
            new_index = page_options.index(selected) + 1
            if new_index != current_page and new_index in accessible_pages:
                st.session_state.current_page = new_index
                st.rerun()

with col3:
    if current_page < len(pages) - 1:
        # Check if can proceed
        can_proceed = check_requirements(current_page + 1)
        
        # Check if current step requirements are set
        current_requirements_met = True
        if current_page > 0:
            current_requirements = pages[current_page].get("sets_state", [])
            current_requirements_met = any(
                state in st.session_state and st.session_state[state] is not None 
                for state in current_requirements
            ) if current_requirements else True
        
        # Show different button states
        if current_page == 0:  # Home page
            if st.button("Começar ➡️", use_container_width=True, type="primary", key="next_btn"):
                st.session_state.current_page += 1
                st.rerun()
        else:
            if can_proceed or not current_requirements_met:
                button_text = "Próximo ➡️" if current_requirements_met else "Pular ⏭️"
                button_help = "Ir para a próxima etapa" if current_requirements_met else "Pular esta etapa (opcional)"
                if st.button(button_text, use_container_width=True, type="primary", key="next_btn", help=button_help):
                    st.session_state.current_page += 1
                    st.rerun()
            else:
                st.button("Próximo ➡️", use_container_width=True, disabled=True, key="next_btn")
                st.caption("⚠️ Complete esta etapa primeiro")

# Main content area
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
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: bold;
        color: #1b5e20;
        margin-bottom: 1rem;
    }
    .hero-subtitle {
        font-size: 1.3rem;
        color: #388e3c;
        margin-bottom: 1.5rem;
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
        transform: translateY(-5px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    .stats-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2e7d32;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-gradient">
        <h1 class="hero-title">🌿 TAIPA SDM</h1>
        <p class="hero-subtitle">Transforme dados de biodiversidade em conhecimento para conservação</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check overall progress
    completion_status = get_completion_status()
    completed_steps = sum(completion_status)
    
    # Progress visualization for returning users
    if completed_steps > 0:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Gauge chart for progress
            progress_percent = (completed_steps / len(completion_status)) * 100
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = progress_percent,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Progresso do Projeto", 'font': {'size': 20}},
                number = {'suffix': "%", 'font': {'size': 40}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1},
                    'bar': {'color': "#4CAF50"},
                    'steps': [
                        {'range': [0, 50], 'color': "#E8F5E9"},
                        {'range': [50, 80], 'color': "#C8E6C9"}
                    ],
                    'threshold': {
                        'line': {'color': "#1B5E20", 'width': 4},
                        'thickness': 0.75,
                        'value': 100
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    # Quick stats if available
    if completed_steps > 0 and "species_name" in st.session_state:
        st.markdown("### 📊 Resumo do Projeto Atual")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center;">
                <div class="stats-number">🦎</div>
                <div><b>{st.session_state.get('species_name', 'N/A')}</b></div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            n_points = len(st.session_state.get('occurrence_data', [])) if 'occurrence_data' in st.session_state else 0
            st.markdown(f"""
            <div style="text-align: center;">
                <div class="stats-number">{n_points}</div>
                <div><b>Ocorrências</b></div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            n_vars = len(st.session_state.get('selected_bioclim_vars', [])) if 'selected_bioclim_vars' in st.session_state else 0
            st.markdown(f"""
            <div style="text-align: center;">
                <div class="stats-number">{n_vars}</div>
                <div><b>Variáveis</b></div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            model_status = "✅" if st.session_state.get('model_trained', False) else "⏳"
            st.markdown(f"""
            <div style="text-align: center;">
                <div class="stats-number">{model_status}</div>
                <div><b>Modelo</b></div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Main CTA based on progress
    if completed_steps == 0:
        st.success("""
        ### 🎯 Pronto para começar sua análise?
        
        Em apenas 6 passos simples, você criará modelos de distribuição de espécies usando dados reais e técnicas profissionais.
        """)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Iniciar Primeira Análise", type="primary", use_container_width=True):
                st.session_state.current_page = 1
                st.rerun()
    
    elif completed_steps < len(completion_status):
        # Find next step
        next_step_idx = completed_steps + 1
        next_step = pages[next_step_idx] if next_step_idx < len(pages) else None
        
        st.info(f"""
        ### 🚀 Continue sua análise
        
        **Próxima etapa:** {next_step['icon']} {next_step['title']}
        """)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("➡️ Continuar de onde parei", type="primary", use_container_width=True):
                st.session_state.current_page = next_step_idx
                st.rerun()
    
    else:
        st.success("""
        ### 🎉 Análise completa!
        
        Parabéns! Você completou todo o fluxo de modelagem.
        """)
        
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
    
    # Feature cards with better styling
    st.markdown("### 🎓 O que é modelagem de distribuição de espécies?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3 style="color: #1976D2;">📊 Dados Reais</h3>
            <p>Acesse milhões de registros de ocorrência de espécies do GBIF - a maior rede de biodiversidade do mundo.</p>
            <hr style="border-color: #E3F2FD;">  
            <small>✓ Busca por nome científico<br>
            ✓ Filtros geográficos<br>
            ✓ Validação automática</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3 style="color: #388E3C;">🌡️ Análise Climática</h3>
            <p>Utilize 19 variáveis bioclimáticas do WorldClim para entender os requisitos ambientais das espécies.</p>
            <hr style="border-color: #E8F5E9;">
            <small>✓ Temperatura e precipitação<br>
            ✓ Sazonalidade climática<br>
            ✓ Extremos ambientais</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3 style="color: #D32F2F;">🤖 Machine Learning</h3>
            <p>Algoritmos de ponta (Random Forest) com validação espacial para criar modelos precisos e confiáveis.</p>
            <hr style="border-color: #FFEBEE;">
            <small>✓ Validação cruzada espacial<br>
            ✓ Métricas de desempenho<br>
            ✓ Mapas de adequabilidade</small>
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
    
    # Check requirements
    if not check_requirements(current_page):
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
        
        # Create button with appropriate styling and tooltip
        button_help = ""
        if not accessible and not completed:
            missing = [req for req in page["required_state"] if req not in st.session_state or st.session_state[req] is None]
            button_help = f"Requer: {', '.join(missing[:2])}{'...' if len(missing) > 2 else ''}"
        
        if i == current_page:
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