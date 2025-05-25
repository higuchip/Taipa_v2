"""
Dicionário de tradução e descrição das variáveis bioclimáticas WorldClim
"""

BIOCLIM_LABELS = {
    'bio1': 'Temperatura Média Anual',
    'bio2': 'Amplitude Térmica Diária Média',
    'bio3': 'Isotermalidade',
    'bio4': 'Sazonalidade da Temperatura',
    'bio5': 'Temperatura Máxima do Mês Mais Quente',
    'bio6': 'Temperatura Mínima do Mês Mais Frio',
    'bio7': 'Amplitude Térmica Anual',
    'bio8': 'Temperatura Média do Trimestre Mais Úmido',
    'bio9': 'Temperatura Média do Trimestre Mais Seco',
    'bio10': 'Temperatura Média do Trimestre Mais Quente',
    'bio11': 'Temperatura Média do Trimestre Mais Frio',
    'bio12': 'Precipitação Anual',
    'bio13': 'Precipitação do Mês Mais Úmido',
    'bio14': 'Precipitação do Mês Mais Seco',
    'bio15': 'Sazonalidade da Precipitação',
    'bio16': 'Precipitação do Trimestre Mais Úmido',
    'bio17': 'Precipitação do Trimestre Mais Seco',
    'bio18': 'Precipitação do Trimestre Mais Quente',
    'bio19': 'Precipitação do Trimestre Mais Frio'
}

BIOCLIM_DESCRIPTIONS = {
    'bio1': 'Temperatura média anual (°C)',
    'bio2': 'Média das amplitudes térmicas diárias mensais (°C)',
    'bio3': 'Isotermalidade (Bio2/Bio7) × 100',
    'bio4': 'Desvio padrão da temperatura × 100',
    'bio5': 'Temperatura máxima do mês mais quente (°C)',
    'bio6': 'Temperatura mínima do mês mais frio (°C)',
    'bio7': 'Amplitude térmica anual (Bio5-Bio6) (°C)',
    'bio8': 'Temperatura média do trimestre mais úmido (°C)',
    'bio9': 'Temperatura média do trimestre mais seco (°C)',
    'bio10': 'Temperatura média do trimestre mais quente (°C)',
    'bio11': 'Temperatura média do trimestre mais frio (°C)',
    'bio12': 'Precipitação total anual (mm)',
    'bio13': 'Precipitação do mês mais úmido (mm)',
    'bio14': 'Precipitação do mês mais seco (mm)',
    'bio15': 'Coeficiente de variação da precipitação',
    'bio16': 'Precipitação do trimestre mais úmido (mm)',
    'bio17': 'Precipitação do trimestre mais seco (mm)',
    'bio18': 'Precipitação do trimestre mais quente (mm)',
    'bio19': 'Precipitação do trimestre mais frio (mm)'
}

BIOCLIM_UNITS = {
    'bio1': '°C',
    'bio2': '°C',
    'bio3': '%',
    'bio4': 'CV',
    'bio5': '°C',
    'bio6': '°C',
    'bio7': '°C',
    'bio8': '°C',
    'bio9': '°C',
    'bio10': '°C',
    'bio11': '°C',
    'bio12': 'mm',
    'bio13': 'mm',
    'bio14': 'mm',
    'bio15': 'CV',
    'bio16': 'mm',
    'bio17': 'mm',
    'bio18': 'mm',
    'bio19': 'mm'
}

def get_bioclim_label(var_code: str) -> str:
    """
    Retorna o nome traduzido da variável bioclimática
    
    Args:
        var_code: Código da variável (ex: 'bio1')
    
    Returns:
        Nome traduzido ou código original se não encontrado
    """
    return BIOCLIM_LABELS.get(var_code, var_code)

def get_bioclim_description(var_code: str) -> str:
    """
    Retorna a descrição completa da variável bioclimática
    
    Args:
        var_code: Código da variável (ex: 'bio1')
    
    Returns:
        Descrição completa ou código original se não encontrado
    """
    return BIOCLIM_DESCRIPTIONS.get(var_code, var_code)

def get_bioclim_unit(var_code: str) -> str:
    """
    Retorna a unidade da variável bioclimática
    
    Args:
        var_code: Código da variável (ex: 'bio1')
    
    Returns:
        Unidade ou string vazia se não encontrado
    """
    return BIOCLIM_UNITS.get(var_code, '')

def format_bioclim_var(var_code: str, include_unit: bool = True) -> str:
    """
    Formata o nome da variável com código e descrição
    
    Args:
        var_code: Código da variável (ex: 'bio1')
        include_unit: Se deve incluir a unidade
    
    Returns:
        String formatada (ex: 'Bio1 - Temperatura Média Anual (°C)')
    """
    label = get_bioclim_label(var_code)
    unit = get_bioclim_unit(var_code) if include_unit else ''
    
    if unit:
        return f"{var_code.capitalize()} - {label} ({unit})"
    else:
        return f"{var_code.capitalize()} - {label}"