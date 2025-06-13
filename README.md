# NERC/NCEO/DARC Training course on data assimilation and its interface with machine learning 2025
Practical material for NERC/NCEO/DARC Training course on data assimilation and its interface with machine learning 2025 in Reading

## On Google Colab (Recommended)
This year, the DARC training course is splitted into foundation and advanced course. In the foundation course, this repository covers:
- Building blocks of DA: [Streamlit](https://alisonfowler-da-training-oi-1-optimal-interpolation-3dxab6.streamlit.app/) and [Jupyter Notebook](https://colab.research.google.com/github/darc-reading/darc-training-2025/blob/main/Control_DA_components.ipynb) 
- Reanalysis: [Jupyter Notebook](https://colab.research.google.com/github/darc-reading/darc-training-2025/blob/main/reanalysis.ipynb)

In the advanced course, we provide practicals for 
- Variational DA: [Jupyter Notebook](https://colab.research.google.com/github/darc-reading/darc-training-2025/blob/main/Controls_L96_Var.ipynb) and [JEDI instructions](JEDI_SABER_exps.pdf)
- EnKF DA: [Jupyter Notebook](https://colab.research.google.com/github/darc-reading/darc-training-2025/blob/main/Controls_L96_Enkf.ipynb), [Optional Notebook](https://colab.research.google.com/github/darc-reading/darc-training-2025/blob/main/Controls_L63_Enkf.ipynb) and [JEDI instructions](JEDI_LETKF_exps.pdf)
- [Machine learning and DA](https://colab.research.google.com/drive/1resUNqp3HHIwV-Ra-tli5Lgu4Qft-ZEB?usp=sharing)

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

# Setting up connections to Reading Academic Computing Cluster (RACC)
Advanced course participants will obtain a temporary RACC account. To utilise the computing resources on the computing cluster, users must connect to RACC via `ssh`. University of Reading provides detailed tutorials for the connection to the **previous** RACC. 
>[!NOTE]
>When follow the following tutorials, you should use `arc-ssh.reading.ac.uk` instead of `act-ssh.reading.ac.uk:2222` or `act-ssh.reading.ac.uk -p2222`, and `racc.rdg.ac.uk` instead of `cluster.act.rdg.ac.uk` whenever you come across them.

The tutorial for connecting to RACC on Windows can be found [here](https://research.reading.ac.uk/act/knowledgebase/act-ssh-with-mobaxterm-step-by-step-guide/). For Mac and Linux users, the tutorial can be found [here](https://research.reading.ac.uk/act/knowledgebase/act-ssh-for-linux-and-mac-users-step-by-step-guide/).



