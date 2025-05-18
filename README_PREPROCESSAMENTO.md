# Preprocessamento de Dados WorldClim para TAIPA SDM

## Visão Geral

Este documento descreve o processo de preparação dos dados WorldClim para uso no TAIPA SDM. O preprocessamento é necessário para:

1. Baixar os dados globais do WorldClim
2. Recortar os dados para o território brasileiro
3. Preparar os dados para deploy no Streamlit Community

## Pré-requisitos

Instale as dependências necessárias:

```bash
pip install rasterio geopandas geobr requests tqdm
```

## Como Executar

### 1. Execute o Preprocessamento Localmente

```bash
python worldclim_preprocessor.py
```

Este script irá:
- Baixar os dados WorldClim (2.5 minutos de resolução)
- Recortar todos os 19 layers bioclimáticos para o Brasil
- Salvar os dados em `data/worldclim_brazil/`
- Criar um arquivo de metadados

### 2. Estrutura de Saída

Após o processamento, você terá:

```
data/
└── worldclim_brazil/
    ├── bio1_brazil.tif    # Annual Mean Temperature
    ├── bio2_brazil.tif    # Mean Diurnal Range
    ├── ...
    ├── bio19_brazil.tif   # Precipitation of Coldest Quarter
    └── metadata.json      # Informações sobre os layers
```

### 3. Deploy no Streamlit Community

Para o deploy, você deve:

1. Adicionar a pasta `data/worldclim_brazil/` ao seu repositório
2. Usar Git LFS para arquivos grandes (recomendado):

```bash
git lfs track "*.tif"
git add .gitattributes
git add data/worldclim_brazil/
git commit -m "Add preprocessed WorldClim data"
```

## Variáveis Bioclimáticas

O WorldClim fornece 19 variáveis bioclimáticas:

### Temperatura
- **BIO1**: Annual Mean Temperature
- **BIO2**: Mean Diurnal Range
- **BIO3**: Isothermality (BIO2/BIO7 × 100)
- **BIO4**: Temperature Seasonality
- **BIO5**: Max Temperature of Warmest Month
- **BIO6**: Min Temperature of Coldest Month
- **BIO7**: Temperature Annual Range
- **BIO8**: Mean Temperature of Wettest Quarter
- **BIO9**: Mean Temperature of Driest Quarter
- **BIO10**: Mean Temperature of Warmest Quarter
- **BIO11**: Mean Temperature of Coldest Quarter

### Precipitação
- **BIO12**: Annual Precipitation
- **BIO13**: Precipitation of Wettest Month
- **BIO14**: Precipitation of Driest Month
- **BIO15**: Precipitation Seasonality
- **BIO16**: Precipitation of Wettest Quarter
- **BIO17**: Precipitation of Driest Quarter
- **BIO18**: Precipitation of Warmest Quarter
- **BIO19**: Precipitation of Coldest Quarter

## Notas Importantes

1. **Espaço em Disco**: O download completo requer ~1GB temporário e ~200MB para dados finais
2. **Tempo de Processamento**: Cerca de 10-30 minutos dependendo da conexão
3. **Resolução**: 2.5 minutos (~5km no equador)
4. **Projeção**: WGS84 (EPSG:4326)

## Alternativas para Deploy

Se os arquivos forem muito grandes para o Streamlit Community:

1. Use serviços de armazenamento em nuvem (S3, Google Cloud Storage)
2. Crie uma API para servir os dados
3. Use resolução menor (10 minutos disponível no WorldClim)

## Suporte

Para problemas ou dúvidas:
- Verifique os logs de erro do preprocessamento
- Confirme que tem espaço em disco suficiente
- Verifique sua conexão com a internet para o download