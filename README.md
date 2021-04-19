# feems-analysis

A workflow for the fast eems project (feems) by Joe Marcus and Wooseok Ha

```
# make sure geos is installed
#brew install geos
#module load geos

conda create -n=feems_analysis_e python=3.8.3 
conda activate feems_analysis_e

# install for feems
conda install -c conda-forge suitesparse=5.7.2 scikit-sparse=0.4.4 cartopy=0.18.0 jupyter=1.0.0 jupyterlab=2.1.5 sphinx=3.1.2 sphinx_rtd_theme=0.5.0 nbsphinx=0.7.1 sphinx-autodoc-typehints

pip install -r requirements.txt

# install for other popgen analysis
conda install -c bioconda plink admixture
```

For eems plotting:

```
conda install -c r rpy2 r-devtools r-rcpp r-rcppeigen r-raster r-rgeos r-sp r-tidyverse r-maps r-rcolorbrewer r-broom r-maptools

mkdir src
cd src 
git clone https://github.com/dipetkov/reemsplots2

# in R session
> install.packages("reemsplots2", repos=NULL, type="source")
> install.packages("rworldmap")
```
