# TAIPA SDM - Species Distribution Modeling Educational Platform

## Overview

TAIPA (Tecnologia Aplicada para Pesquisa Ambiental) is an educational platform designed to teach Species Distribution Modeling (SDM) concepts and techniques using interactive Streamlit applications.

## Project Structure

```
Taipa_v2/
├── app.py                      # Main Streamlit application with navigation
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── fundamentals_module/        # Module 1: SDM basics and theory
├── geographical_data_module/   # Module 2: Spatial data handling
├── statistical_analysis_module/# Module 3: Statistical methods
├── ml_modeling_module/         # Module 4: Machine learning approaches
└── outputs_module/            # Module 5: Results visualization
```

## Modules Description

### 1. Fundamentals Module
- Introduction to SDM concepts
- Ecological niche theory
- Types of distribution models
- Applications in conservation

### 2. Geographical Data Module
- Working with coordinate systems
- Environmental layers management
- Species occurrence data processing
- Spatial data visualization

### 3. Statistical Analysis Module
- Descriptive statistics for SDM
- Correlation analysis
- Variable selection techniques
- Model validation methods

### 4. Machine Learning Modeling Module
- Random Forest implementation
- Support Vector Machines
- Neural Networks for SDM
- Ensemble methods

### 5. Outputs Module
- Distribution map generation
- Uncertainty visualization
- Report generation
- Conservation planning applications

## Installation

1. Clone the repository or download the project files
2. Navigate to the project directory
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **GeoPandas**: Geospatial data handling
- **Rasterio**: Raster data I/O
- **Folium**: Interactive mapping
- **Plotly**: Interactive visualizations
- **Requests**: HTTP library for API calls

## Development Status

This is the initial structure setup. Module contents will be implemented progressively.

## Contributing

This is an educational platform. Contributions to improve the learning experience are welcome.

## License

Educational use - details to be specified.

## Contact

TAIPA SDM Platform - Environmental Research Education