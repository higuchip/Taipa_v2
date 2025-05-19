# Data Files Information

This application requires WorldClim bioclimatic data files to function properly. These files are included in the repository as they are essential for the Species Distribution Modeling workflow.

## Required Data Files

### WorldClim Bioclimatic Variables
Location: `data/worldclim_brazil/`

- **bio1_brazil.tif** - Annual Mean Temperature
- **bio2_brazil.tif** - Mean Diurnal Range
- **bio3_brazil.tif** - Isothermality
- **bio4_brazil.tif** - Temperature Seasonality
- **bio5_brazil.tif** - Max Temperature of Warmest Month
- **bio6_brazil.tif** - Min Temperature of Coldest Month
- **bio7_brazil.tif** - Temperature Annual Range
- **bio8_brazil.tif** - Mean Temperature of Wettest Quarter
- **bio9_brazil.tif** - Mean Temperature of Driest Quarter
- **bio10_brazil.tif** - Mean Temperature of Warmest Quarter
- **bio11_brazil.tif** - Mean Temperature of Coldest Quarter
- **bio12_brazil.tif** - Annual Precipitation
- **bio13_brazil.tif** - Precipitation of Wettest Month
- **bio14_brazil.tif** - Precipitation of Driest Month
- **bio15_brazil.tif** - Precipitation Seasonality
- **bio16_brazil.tif** - Precipitation of Wettest Quarter
- **bio17_brazil.tif** - Precipitation of Driest Quarter
- **bio18_brazil.tif** - Precipitation of Warmest Quarter
- **bio19_brazil.tif** - Precipitation of Coldest Quarter
- **metadata.json** - Layer metadata and descriptions

Total size: ~25MB

## Data Source

These files are pre-processed from WorldClim version 2.1 (https://www.worldclim.org/) at 2.5 arc-minutes resolution, clipped to Brazil's boundaries.

## Important Notes

1. These files are essential for the application to work
2. Do not remove or modify these files
3. If files are missing, run `worldclim_preprocessor.py` to regenerate them
4. The files are small enough to be included in the Git repository (< 100MB total)

## License

WorldClim data is freely available for academic and non-commercial use. Please cite:
Fick, S.E. and R.J. Hijmans, 2017. WorldClim 2: new 1-km spatial resolution climate surfaces for global land areas. International Journal of Climatology 37 (12): 4302-4315.