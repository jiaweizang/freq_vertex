# Project Repository

This repository contains analytical codes PCA and deep learning analysis of dimensional reduction on frequency vertex.

## Directory Structure

### `/pca`
This directory holds the principal component analysis (PCA) codes, structured in a step-by-step manner:

- `01_save_K1K2K2prime.ipynb`: Stores the original 140 data into a single file for K1, K2, K2' components
- `02_save_vertex_total.ipynb`: Stores the original 140 data into a single file for the total vertex and Kelydish components.
- `03_pca_fb_cal.py`, `04_job.sh`: PCA analysis codes. Note that the `fbpca` package is required due to the complex value of the data.
- `05_PCAanalysis_all_svd.ipynb` to `08_PCAanalysis_K1.ipynb`: Detailed PCA analysis for separate components.

### `/code`
Contains scripts for deep learning models used in the analysis:

- Python modules like `cPCA.py`, `cost.py`, `models.py` provide the underlying functionality for deep learning.

### `/DL`
Focuses on the deep learning aspect of the project:

- Notebooks like `01_dl_training.ipynb` and `02_compare_PCA_DL.ipynb` are used for training deep learning models and comparing the results to PCA outcomes.
- `/DL/save`: Contains saved models from the deep learning runs.

## Requirements
- Ensure that the `fbpca` Python package is installed to handle PCA calculations with complex numbers. 