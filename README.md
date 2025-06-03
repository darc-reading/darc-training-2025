# NERC/NCEO/DARC Training course on data assimilation and its interface with machine learning 2025
Practical material for NERC/NCEO/DARC Training course on data assimilation and its interface with machine learning 2025 in Reading

## On Google Colab (Recommended)
This year, the DARC training course is splitted into foundation and advanced course. In the foundation course, this repository covers:
- Building blocks of DA: [Streamlit](https://alisonfowler-da-training-oi-1-optimal-interpolation-3dxab6.streamlit.app/) and [Jupyter Notebook](https://colab.research.google.com/github/darc-reading/darc-training-2025/blob/main/Control_DA_components.ipynb) 
- Reanalysis: [Jupyter Notebook](https://colab.research.google.com/github/darc-reading/darc-training-2025/blob/main/reanalysis.ipynb)

In the advanced course, we provide practicals for 
- Variational DA: [Jupyter Notebook](https://colab.research.google.com/github/darc-reading/darc-training-2025/blob/main/Controls_L96_Var.ipynb) and [JEDI instructions](JEDI_SABER_exps.pdf)
- EnKF DA: [Jupyter Notebook](https://colab.research.google.com/github/darc-reading/darc-training-2025/blob/main/Controls_L96_Enkf.ipynb) and [Optional Notebook](https://colab.research.google.com/github/darc-reading/darc-training-2025/blob/main/Controls_L63_Enkf.ipynb)
- Machine learning and DA [To be released]

## Running on your own laptop locally
To run practicals on your local laptop, we recommend configure the Python environment using `conda`. Once you installed `conda`, you can launch the Jupyter notebook with the following command in your terminal or anaconda command prompt:
```bash
# create a new conda envrionment and install required packages
conda create -n da_course conda-forge::numpy conda-forge::scipy conda-forge::matplotlib conda-forge::cartopy conda-forge::ipywidgets conda-forge::jupyter conda-forge::xarray conda-forge::dask conda-forge::netCDF4 conda-forge::bottleneck conda-forge::kagglehub
# activate the conda environment
conda activate da_course
# launch jupyter notebook
jupyter notebook
```
