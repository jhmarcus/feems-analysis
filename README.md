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

# install other python requirements needed for installing feemds
pip install -r requirements.txt

# install for other popgen analysis
conda install -c bioconda plink admixture
```

Install feems:

```
mkdir src
cd src/

git clone https://github.com/NovembreLab/feems
cd feems/
pip install .
```

For eems plotting:

```
conda install -c r rpy2=3.4.3 r-devtools=2.0.2 r-rcpp=1.0.1 r-rcppeigen=0.3.3.5.0 r-raster=2.8_19 r-rgeos=0.5_5 r-sp=1.3_1 r-tidyverse=1.2.1 r-maps=3.3.0 r-rcolorbrewer=1.1_2 r-broom=0.5.2 r-maptools=0.9_5

cd src/ 
git clone https://github.com/dipetkov/reemsplots2

# in R session
> install.packages("reemsplots2", repos=NULL, type="source")
> install.packages("rworldmap")
```
