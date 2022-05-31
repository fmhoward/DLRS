# Deep Learning for Recurrence Score Prediction
Provides estimates for risk of recurrence from digital histology / multimodal predictions incorporating clinical features
<br>
<img src="https://github.com/fmhoward/DLRS/blob/main/overview.png?raw=true" width="600">

## Attribution
If you use this code in your work or find it helpful, please consider citing our preprint ***.
```
@article{howard_impact_2021,
	title = {The impact of site-specific digital histology signatures on deep learning model accuracy and bias},
	volume = {12},
	issn = {2041-1723},
	url = {https://www.nature.com/articles/s41467-021-24698-1},
	doi = {10.1038/s41467-021-24698-1},
	pages = {1--13},
	number = {1},
	journaltitle = {Nature Communications},
	shortjournal = {Nat Commun},
	author = {Howard, Frederick M. and Dolezal, James and Kochanny, Sara and Schulte, Jefree and Chen, Heather and Heij, Lara and Huo, Dezheng and Nanda, Rita and Olopade, Olufunmilayo I. and Kather, Jakob N. and Cipriani, Nicole and Grossman, Robert L. and Pearson, Alexander T.},
	date = {2021-07-20},
}
```

## Installation
The associated files can be downloaded to a project directory. Installation takes < 5 minutes on a standard desktop computer. Runtime for hyperparameter optimization is approximately 96 hours for 50 iterations. Runtime for model training of the tumor region of interest and recurence score predictive models was approximately 4 hours. The analysis of results is performed in < 1 minute. All software was tested on CentOS 8 with an AMD EPYC 7302 16-Core Processor and 4x A100 SXM 40 GB GPUs.

Requirements:
* python 3.8
* tensorflow 2.8.0
* opencv 4.5.5.62
* scipy 1.7.3
* scikit-image 0.18.3
* pixman 0.40.0
* OpenSlide
* Libvips 8.9
* pandas 1.3.5
* numpy 1.21.2
* IBM's CPLEX 12.10.0 and <a href='https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html'>CPLEX's python API</a>
For full environment used for model testing please see the requirements.yml file

## Overview
First, to replicate model results, the datasets file must be configured to navigate to local slide storage and region of interest directories. Additionally, please update the 'PROJECT_ROOT' variable in the model_training.py and model_analysis.py files. 

Slide extraction and model training can be performed by running model_training.py. By default, this will train models with our saved optimal hyperparameters without performing additional hyperparameter search. To run an additional hyperparameter search, pass hpsearch = 'run' to the train_test_models function.

```python
def train_test_models(hpsearch = 'old', prefix_hpopt = 'hp_new2'):
    """Extracts tiles to setup all projects

    Parameters
    ----------
    hpsearch - whether to perform a hyperparameter search. 'old' - will use saved hyperparameters. 'read' - will read hyperparameters from the model directory. 'run' - will run hyperparameter optimization
	prefix_hpopt - prefix for models trained under hyperparameter search
    """   
```

Finally, to generate statistics and figures, run the model_analysis.py file.

## Reproduction
To recreate our cross validation setup from publication, please ensure the RUN_FROM_OLD_STATS variable to True in model_analysis.py, which will perform the analysis on saved models. Digital slides from the TCGA cohort can be obtained from https://www.cbioportal.org/. To replicate our full model training, the CSV files with associated annotations are provided in the PROJECTS/UCH_RS and PROJECTS/TCGA_BRCA_ROI directories. Given the non-deterministic nature of some features of training such as dropout, results may vary slightly despite identical training parameters.


