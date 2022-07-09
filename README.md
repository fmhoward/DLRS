# Deep Learning for Recurrence Score Prediction
Provides estimates for risk of recurrence from digital histology / multimodal predictions incorporating clinical features
<br>
<img src="https://github.com/fmhoward/DLRS/blob/main/overview.png?raw=true" width="600">

## Attribution
If you use this code in your work or find it helpful, please consider citing our preprint in <a href='https://www.biorxiv.org/content/10.1101/2022.07.07.499039v1'>bioRxiv</a>.
```
@article {Howard2022.07.07.499039,
	author = {Howard, Frederick Matthew and Dolezal, James and Kochanny, Sara and Khramtsova, Galina and Vickery, Jasmine and Srisuwananukorn, Andrew and Woodard, Anna and Chen, Nan and Nanda, Rita and Perou, Charles M and Olopade, Olufunmilayo I. and Huo, Dezheng and Pearson, Alexander T.},
	title = {Multimodal Prediction of Breast Cancer Recurrence Assays and Risk of Recurrence},
	elocation-id = {2022.07.07.499039},
	year = {2022},
	doi = {10.1101/2022.07.07.499039},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Gene expression-based recurrence assays are strongly recommended to guide the use of chemotherapy in hormone receptor-positive, HER2-negative breast cancer, but such testing is expensive, can contribute to delays in care, and may not be available in low-resource settings. Here, we describe the training and independent validation of a deep learning model that predicts recurrence assay result and risk of recurrence using both digital histology and clinical risk factors. We demonstrate that this approach outperforms an established clinical nomogram (area under the receiver operating characteristic curve of 0.833 versus 0.765 in the validation cohort, p = 0.003), and can identify a subset of patients with excellent prognoses who may not need further genomic testing.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2022/07/08/2022.07.07.499039},
	eprint = {https://www.biorxiv.org/content/early/2022/07/08/2022.07.07.499039.full.pdf},
	journal = {bioRxiv}
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
First, to replicate model results, the datasets file must be configured to navigate to local slide storage and region of interest annotations for the TCGA and external validation / testing datasets. Please also specify directory for TFRECORD file storage.

## Slide Extraction
Slide extraction can be performed by running the model_training.py file with the -e option.

## Hyperparameter optimization
To perform hyperparameter optimization, run the model_training.py file with the following parameters (for 50 runs for hyperparameter optimization):
```
model_training.py --hpsearch run --hpprefix DESIRED_PREFIX --hpstart 0 --hpcount 50
```

## Model training
To train models for tumor detection and region of interest annotation, run model_training.py with the following parameters:
```
model_training.py -t --hpsearch read --hpprefix DESIRED_PREFIX --hpstart 0 --hpcount 50
```

Or, if you do not want to rerun hyperparameter optimization and would prefer to use stored hyperparameters from our optimization:
```
model_training.py -t --hpsearch old
```

## Model validation
To validate models in an external dataset, run model_training.py with the -v flag:
```
model_training.py -v
```

To evaluate performance characteristics of the trained model, run model_analysis.py. If you want to assess performance of the external validation dataset from our model training, you can use the saved predictions from our trained models with the -s flag:
```
model_analysis.py -s
```

To make new predictions using our trained models, please <a href='doi.org/10.5281/zenodo.6792391'>download the trained models from Zenodo</a> and extract the zip into the PROJECTS folder. Predictions can be made with the <a href='https://slideflow.dev/'>Slideflow evaluate function</a> or by specifying slides and associated clinical characteristics in the UCH_RS project folder, and running model_training.py -v and model_analysis.py as above.

## Model interpretation
To view heatmaps from trained models, run model_training.py with the --heatmaps_tumor_roi or --heatmaps_odx_roi, and specify 'TCGA' or 'UCH' depending on which dataset you want to generate heatmaps for:
```
model_training.py --heatmaps_tumor_roi TCGA
model_training.py --heatmaps_odx_roi TCGA
```
<img src="https://github.com/fmhoward/DLRS/blob/main/heatmaps.png?raw=true" width="600">


## Reproduction
To recreate our cross validation setup from publication, please ensure the RUN_FROM_OLD_STATS variable to True in model_analysis.py, which will perform the analysis on saved models. Digital slides from the TCGA cohort can be obtained from https://www.cbioportal.org/. To replicate our full model training, the CSV files with associated annotations are provided in the PROJECTS/UCH_RS and PROJECTS/TCGA_BRCA_ROI directories. Given the non-deterministic nature of some features of training such as dropout, results may vary slightly despite identical training parameters.


