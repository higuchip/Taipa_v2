# Projeção Futura - Guia de Uso

## Visão Geral

O módulo de Projeção Futura do TAIPA permite analisar como as mudanças climáticas podem afetar a distribuição de espécies. Esta funcionalidade é essencial para:

- Planejamento de conservação
- Identificação de refúgios climáticos
- Avaliação de vulnerabilidade de espécies
- Educação sobre impactos das mudanças climáticas

## Configuração Pedagógica

Para uso educacional focado na Mata Atlântica, selecionamos:

### Modelo Climático Global (GCM)
- **MPI-ESM1-2-HR**: Modelo alemão de alta resolução do Instituto Max Planck

### Cenários Socioeconômicos (SSPs)
1. **SSP1-2.6** (Otimista): Desenvolvimento sustentável com forte mitigação climática
2. **SSP5-8.5** (Pessimista): Cenário de uso intensivo de combustíveis fósseis

### Período Temporal
- **2081-2100**: Futuro distante, mostrando impactos de longo prazo das mudanças climáticas

## Preparação dos Dados

### 1. Download dos Dados Climáticos Futuros

Execute o script de download:

```bash
python download_future_climate.py
```

Este script irá:
- Baixar dados multibanda do CMIP6/WorldClim
- Extrair 19 variáveis bioclimáticas individuais
- Recortar para a região do Brasil
- Organizar em pastas por cenário
- Total aproximado: ~400 MB por cenário (800 MB total)

### 2. Estrutura de Diretórios

```
data/
└── worldclim_future/
    ├── ssp126_2081-2100/
    │   └── [19 arquivos .tif]
    └── ssp585_2081-2100/
        └── [19 arquivos .tif]
```

## Uso no TAIPA

### 1. Pré-requisitos
- Modelo treinado na aba "Modelagem e Resultados"
- Projeção espacial atual gerada

### 2. Fluxo de Trabalho
1. Navegue para "6. Projeção Futura"
2. Selecione o cenário climático (SSP1-2.6 ou SSP5-8.5)
3. Escolha o método de threshold:
   - **Manual**: Ajuste o valor (padrão 0.5)
   - **Usar do mapa atual**: Usa o mesmo threshold da projeção atual
   - **Automatico**: Calcula baseado nos dados (média, percentis)
4. O período está fixado em 2081-2100
5. Clique em "Gerar Projeção Futura"

### 3. Visualizações Disponíveis
- **Mapas Binários**: Comparação de presença/ausência entre presente e futuro
- **Mapas de Probabilidade**: Opcional, mostra gradiente de adequabilidade
- **Mapa de mudanças**: Ganhos e perdas de habitat adequado
- **Métricas**: Mudança percentual de área adequada
- **Estatísticas**: Áreas de ganho, perda, habitat estável

### 4. Exportação
- Mapas binários futuros em formato GeoTIFF
- Mapas de probabilidade em formato GeoTIFF
- Mapas de mudança em formato GeoTIFF
- Compatível com QGIS, ArcGIS, etc.

## Interpretação dos Resultados

### Cores no Mapa de Mudanças
- 🔴 **Vermelho**: Perda de adequabilidade
- 🔵 **Azul**: Ganho de adequabilidade
- ⚪ **Branco**: Sem mudança significativa

### Métricas Importantes
- **% Mudança**: Variação percentual da área adequada
- **Área de Ganho**: Novas áreas adequadas no futuro
- **Área de Perda**: Áreas que deixarão de ser adequadas
- **Áreas Estáveis**: Refúgios climáticos potenciais

## Limitações e Considerações

### Limitações Metodológicas
1. Assume relações espécie-ambiente constantes
2. Não considera capacidade de dispersão
3. Ignora interações bióticas
4. Não inclui mudanças no uso do solo
5. **Uso de modelo único em vez de ensemble**

### Nota sobre Ensemble de Modelos
A recomendação científica padrão é utilizar um ensemble (conjunto) de múltiplos 
modelos climáticos globais (GCMs) para:
- Capturar a incerteza nas projeções
- Obter estimativas mais robustas
- Identificar áreas de consenso entre modelos
- Quantificar a variabilidade das projeções

**No TAIPA**, para fins didáticos, utilizamos apenas um modelo (MPI-ESM1-2-HR) 
para simplificar o processo de aprendizagem. Em aplicações científicas reais, 
sempre utilize múltiplos GCMs (recomenda-se no mínimo 5-10 modelos).

### Interpretação Cuidadosa
- Resultados são projeções, não previsões
- Incerteza aumenta com horizonte temporal
- Múltiplos cenários mostram range de possibilidades
- Útil para planejamento, não para decisões definitivas

## Conceitos Importantes

### SSPs (Shared Socioeconomic Pathways)
Narrativas de desenvolvimento socioeconômico futuro:
- **SSP1**: Desenvolvimento sustentável e inclusivo
- **SSP5**: Desenvolvimento baseado em combustíveis fósseis

### RCPs (Representative Concentration Pathways)
Trajetórias de concentração de gases de efeito estufa:
- **RCP 2.6**: ~490 ppm CO₂ em 2100 (SSP1-2.6)
- **RCP 8.5**: ~1370 ppm CO₂ em 2100 (SSP5-8.5)

## Melhores Práticas Científicas vs. Simplificações Didáticas

### O que fazemos no TAIPA (Didático)
- **1 modelo climático** (MPI-ESM1-2-HR)
- **2 cenários** (SSP1-2.6, SSP5-8.5)
- **1 período** (2081-2100)
- **Foco na compreensão** dos conceitos

### O que é recomendado em pesquisa real
- **5-10+ modelos climáticos** (ensemble)
- **Todos os cenários SSP** disponíveis
- **Múltiplos períodos** temporais
- **Análise de incerteza** completa
- **Validação cruzada** entre modelos
- **Métricas de consenso** entre modelos

## Aplicações Educacionais

### Atividades Sugeridas
1. Compare diferentes cenários para a mesma espécie
2. Identifique potenciais refúgios climáticos
3. Calcule velocidade de migração necessária
4. Analise espécies com diferentes tolerâncias térmicas
5. Discuta limitações do uso de modelo único

### Questões para Discussão
1. Quais áreas permanecerão adequadas em todos cenários?
2. Como a fragmentação pode afetar a migração?
3. Que medidas de conservação seriam necessárias?
4. Como a incerteza afeta o planejamento?

## Suporte

Para problemas ou dúvidas:
1. Verifique se os dados foram baixados corretamente
2. Confirme que há um modelo treinado
3. Consulte as mensagens de erro
4. Abra uma issue no GitHub do projeto

## Referências

- Fick, S.E. and Hijmans, R.J. (2017). WorldClim 2: new 1-km spatial resolution climate surfaces for global land areas.
- O'Neill, B.C. et al. (2016). The Scenario Model Intercomparison Project (ScenarioMIP) for CMIP6.
- Müller, W.A. et al. (2018). A Higher-resolution Version of the Max Planck Institute Earth System Model (MPI-ESM1.2-HR).