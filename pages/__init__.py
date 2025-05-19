from .pagina_busca_api import render_page as pagina_busca_api
from .pagina_pseudoausencias import render_page as pagina_pseudoausencias
from .pagina_analise_bioclimatica import render_page as pagina_analise_bioclimatica
from .pagina_modelagem import render_page as pagina_modelagem
from .pagina_projecao_espacial import render_page as pagina_projecao_espacial
from .pagina_projecao_futura import render_page as pagina_projecao_futura

__all__ = [
    'pagina_busca_api',
    'pagina_pseudoausencias',
    'pagina_analise_bioclimatica', 
    'pagina_modelagem',
    'pagina_projecao_espacial',
    'pagina_projecao_futura'
]