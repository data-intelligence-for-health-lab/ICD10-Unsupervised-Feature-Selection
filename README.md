# Unsupervised Feature Selection to Identify Important ICD-10 and ATC Codes for Machine Learning


This repository contains the code for the paper titled "[_Unsupervised Feature Selection to Identify Important 
ICD-10 Codes for Machine Learning: A Case Study on a Coronary Artery Disease Patient Cohort_](https://arxiv.org/abs/2303.14303)".
Please cite the following paper if you use this code in your research:


    @article{ghasemi2023unsupervised,
    title={Unsupervised Feature Selection to Identify Important ICD-10 Codes for Machine Learning: A Case Study on a Coronary Artery Disease Patient Cohort},
    author={Ghasemi, Peyman and Lee, Joon},
    journal={arXiv:2303.14303},
    year={2023}
    }


## Abstract
The use of International Classification of Diseases (ICD) codes in healthcare presents a challenge in selecting 
relevant codes as features for machine learning models due to this system's large number of codes. In this study, 
we compared several unsupervised feature selection methods for an ICD code database of 49,075 coronary artery disease 
patients in Alberta, Canada. Specifically, we employed Laplacian Score, Unsupervised Feature Selection for 
Multi-Cluster Data, Autoencoder Inspired Unsupervised Feature Selection, Principal Feature Analysis, and 
Concrete Autoencoders with and without ICD tree weight adjustment to select the 100 best features from over 
9,000 codes. We assessed the selected features based on their ability to reconstruct the initial feature space 
and predict 90-day mortality following discharge. Our findings revealed that the Concrete Autoencoder methods 
outperformed all other methods in both tasks. Furthermore, the weight adjustment in the Concrete Autoencoder method 
decreased the complexity of features.

## Requirements
    - Python 3.9
    - Packages included in requirements.txt

## Repository Structure
    ├── README.md                          # Contains information about the repository
    ├── requirements.txt                   # Contains required packages
    ├── ICD_10_CA_DX.csv *                 # List of ICD-10-CA codes and their descriptions (Canadian ICD codes)
    ├── constants.py                       # Contains constant values used in the project and data paths
    ├── concrete_autoencoder.py **         # Implements Concrete Autoencoders algorithm
    ├── process_icd_codes.py               # Processes raw ICD-10 codes data, and generates list of all ICD codes and their parent diseases (including Canadian ICD codes) and generates "all_codes_list.csv" file
    ├── one_hot_encode_icd_table.py        # Performs one-hot encoding of all ICD codes for all patients in DAD dataset
    ├── post_process_one_hot_table.py      # Performs post-processing on the one-hot encoded ICD table to resample the data to 3 months period
    ├── match_mortality_data_to_onehot_table.py   # Matches mortality data from VS dataset with one-hot encoded ICD table
    ├── icd_selection_concrete_ae_with_weights.py # selects features based on Concrete Autoencoders with ICD tree weight adjustment
    ├── icd_selection_concrete_ae_without_weights.py # selects features based on Concrete Autoencoders without ICD tree weight adjustment
    ├── icd_selection_aefs.py              # selects features based on Autoencoder Inspired Unsupervised Feature Selection
    ├── icd_selection_pfa.py               # selects features based on Principal Feature Analysis
    ├── reconstruction_test.py             # Tests the performance of selected features in feature reconstruction
    ├── mortality_prediction_test.py       # Tests the performance of selected features in predicting mortality
    ├── generate_tables_for_selected_features.py  # Generates tables of selected features for each method (including the selected features' names and their descriptions)
    ├── selected_features/                 # Contains text files and tables of selected features for each method
    │   ├── results/                       # Contains results of reconstruction and mortality prediction tests
    │   ├── concrete_with_weights.csv
    │   ├── concrete_without_weights.csv
    │   ├── lap_features.csv
    │   ├── mcfs_features.csv
    │   ├── selected_features_AEFS.csv
    │   └── selected_features_PFA.csv
    └── skfeature_job/                     # Contains code for Laplacian Score and MCFS methods (needs to be run on a high-memory-high-CPU cluster)
        ├── skfeature/ ***                 # Contains code for the feature selection methods from scikit-feature package
        ├── skf_job.py                     # Constructs affinity matrix and selects features based on Laplacian Score 
        └── mcfs.py                        # selects features based on Unsupervised Feature Selection for Multi-Cluster Data       

*The ICD-10-CA codes are not included in this repository (This file is just a sample). They can be downloaded from the [Canadian Institute for Health Information](https://secure.cihi.ca/estore/productSeries.htm?pc=PCC84) website. You may also find them [here](https://ext.cancercare.on.ca/ext/databook/db1718/Appendix/Appendix_1.18_-_ICD10CA_.htm) ;-) 

**The Concrete Autoencoder code is based on the [original implementation](https://github.com/mfbalin/Concrete-Autoencoders) by [mfbalin](https://github.com/mfbalin)

***The scikit-feature package is based on the [original implementation](https://github.com/jundongl/scikit-feature) by [jundongl](https://github.com/jundongl)

## How to Run
1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`
3. Download the ICD-10-CA codes and replace them with the sample file in the repository (unless you do not have Canadian ICD codes in your dataset. In that case, just leave an empty file with the same name in the repository).
4. Change the data paths in `constants.py` to match your data paths.
5. Run these bash scripts:


\# You can run the following commands on a regular computer (preferably with GPU)

    python process_icd_codes.py
    python one_hot_encode_icd_table.py
    python post_process_one_hot_table.py
    python match_mortality_data_to_onehot_table.py
    python icd_selection_concrete_ae_with_weights.py
    python icd_selection_concrete_ae_without_weights.py
    python icd_selection_aefs.py
    python icd_selection_pfa.py


\# Run the following commands on a high-memory-high-CPU cluster (make sure to put the selected features in the "selected_features" folder)

    python skfeature_job/skf_job.py
    python skfeature_job/mcfs.py


\# You can run the following commands on a regular computer (Evaluation of the selected features)

    python reconstruction_test.py
    python mortality_prediction_test.py
    python generate_tables_for_selected_features.py

6. You can see the results of the feature selection methods in the `selected_features` folder. I have already included
the results of my own experiments in this folder.

## Data
The data used in this project is not included in this repository.
You need to request the data from Alberta Health Services (AHS) or
the Canadian Institute for Health Information (CIHI). We used two AHS databases of DAD and VS.
You may find the dictionary of data [here](https://cumming.ucalgary.ca/centres/centre-health-informatics/data-and-analytic-services/data-resources/ahs-datasets)
. You need to save them as CSV files and modify their paths in the `constants.py`.

## Contact
If you have any questions, please do not hesitate to contact me 
at [peyman.ghasemi@ucalgary.ca](mailto:peyman.ghasemi@ucalgary.ca).
You can also open an issue in this repository.

