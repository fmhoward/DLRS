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
This github repository should be downloaded to a project directory. Installation takes < 5 minutes on a standard desktop computer. Runtime for hyperparameter optimization is approximately 96 hours for 50 iterations. Runtime for model training of the tumor region of interest and recurence score predictive models was approximately 4 hours. The analysis of results is performed in < 1 minute. All software was tested on CentOS 8 with an AMD EPYC 7302 16-Core Processor and 4x A100 SXM 40 GB GPUs.

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
* smac 1.4.0
* IBM's CPLEX 12.10.0 and <a href='https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html'>CPLEX's python API</a>
For full environment used for model testing please see the requirements.yml file

## Setup
This package heavily utilizes the <a href='https://github.com/jamesdolezal/slideflow/tree/master/slideflow'>Slideflow repository</a>, and reading the associated <a href='https://slideflow.dev/'>extensive documentation<a> is recommended to familiarize users with the workflow used in this project.

After downloading the associated files, the first step is to edit the datasets.json (located in the main directory of this repository) to reflect the location of where slide images are stored for the TCGA and UCMC datasets. The TCGA slide images can be downloaded from <a href='https://portal.gdc.cancer.gov'>https://portal.gdc.cancer.gov</a>. The extracted anonymized tfrecord files from the UCMC dataset are available from Zenodo. 

Each 'dataset' within the datasets.json has four elements:
"slides": location of the whole slide images
"roi": location of region of interest annotations. We have provided our region of interest annotations from TCGA in the /roi/ directory
"tiles": location to extract free image tiles. We disable this in our extract image function
"tfrecords": location of tfrecords containing the extracted image tiles for slides

## Slide Extraction
Slide extraction can be performed by running the following command:
	
```	
python model_training.py -e
```
	
This will automatically extract tfrecords from associated slide images. The code assumes the PROJECTS folder which contains the list of slides to use for each project is located in the directory that the model_training.py script is run from. This script automatically performs extraction with the slideflow backbone, for example -

```
SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS")) # specifies the location of the project folder
SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "uch_brca_complete.csv") # specifies the location of the annotations file to use as a guide for slides to extract
SFP.sources = ["UCH_BRCA_RS"] # specifies the dataset location
SFP.extract_tiles(tile_px=tile_px, tile_um=tile_um, skip_missing_roi=False, save_tiles=False, skip_extracted=overwrite, roi_method = 'ignore', source="UCH_BRCA_RS", buffer=join(PROJECT_ROOT, "buffer")) # extracts tiles for the 'UCH_BRCA_RS' dataset using the specified tile pixel size (302), tile size in microns (299), without using regions of interest
```

The following sets of data are created for this project
```
UCH_BRCA_RS - extracted tiles from University of Chicago slides. These should be placed in the appropriate tfrecords directory from the datasets.json using the anonymized files uploaded to Zenodo
UCH_BRCA_RS_FULL_ROI - extracted tiles from University of Chicago slides using pathologist specified regions of interest - an optional dataset used to assess training models from University of Chicago and validating on TCGA, not used for primary analysis of this project. 
TCGA_BRCA_FULL_ROI - extracted tiles from TCGA using specified regions of interest
TCGA_BRCA_NO_ROI - extracted tiles from TCGA without regions of interest
TCGA_BRCA_NORMAL - extracted tiles from TCGA using inverse region of interest (essentially the normal tissue surrounding tumors in the dataset, used for training the tumor likelihood model)
TCGA_BRCA_FILTERED - a dataset where tiles are filtered from the TCGA using the tumor detection module rather than pathologist annotations - an optional dataset not used for the primary analysis of this project
```

## Hyperparameter optimization
To perform hyperparameter optimization, run the model_training.py file with the following parameters (example given for 50 runs for hyperparameter optimization):
```
python model_training.py --hpsearch run --hpprefix DESIRED_PREFIX --hpstart 0 --hpcount 50
```

This code will automatically run hyperparameter optimization for the specified iterations. The following range of parameters are used for the hyperparameter search
```
Dropout: 0 - 0.5
Hidden Layer Width (dimension of fully connected layers after Xception backbone): 128 - 1024
Hidden Layers (number of fully connected hidden layers): 1 - 5
Learning Rate: 0.00001 - 0.001
Learning Rate Decay Steps (# of batches until applying learning rate decay): 128 - 1024
Learning Rate Decay Ratio (ratio with which to reduce the learning rate): 0 - 1
Learning Rate Decay (whether to decay learning rate): True / False
Loss (for linear loss, will train models on numerical recurrence score; for categorical loss will train on high / low cutoff of recurrence score): mean squared error, mean absolute error, sparse categorical cross entropy
Batch Size: 8 - 128
Augment (‘x’ performs horizontal flipping, ‘y’ performs vertical flipping, ‘r’ performs rotation, ‘j’ performs random JPEG compression): xyr, xyrj, xyrjb
Normalizer (whether to apply stain normalization): reinhard, none
L1 weight: 0 - 0.1
L1 (whether to perform L1 regularization): True, False
L2 weight: 0 - 0.1
L2 (whether to perform L2 regularization): True, False
L1 dense weight: 0 - 0.1
L1 dense (whether to perform l1 regularization to dense layers): True, False
L2 dense weight: 0 - 0.1
L2 dense (whether to perform l1 regularization to dense layers): True, False
```

Optimization is done using the <a href='https://pypi.org/project/smac/'>SMAC package</a>. Optimization is performed across two sets of three cross folds (listed in the tcga_brca_complete.csv file as CV3_odx85_mip and CV3_mp85_mip. These cross folds were chosen using <a href='https://github.com/fmhoward/PreservedSiteCV'>site preserved</a> splits to optimize the balance of high versus low risk OncotypeDx and MammaPrint recurrence scores. These splits can be regenerated, if desired, using the createCrossfoldsRS function in the model_analysis.py file. 

<img src="https://github.com/fmhoward/DLRS/blob/main/bayesian.png?raw=true" width="600">


## Model training
To train models for tumor detection and region of interest annotation, run model_training.py with the following parameters:
```
python model_training.py -t --hpsearch read --hpprefix DESIRED_PREFIX --hpstart 0 --hpcount 50
```
	
This will search for the saved tile-level AUROC results from models stored within /PROJECTS/UCH_RS/models/ with the prefix DESIRED_PREFIX, identifying the hyperparameter combination with the highest tile-level AUROC.

Or, if you do not want to rerun hyperparameter optimization and would prefer to use stored hyperparameters from our optimization, they can automatically be  loaded as follows:
```
python model_training.py -t --hpsearch old
```

This command will train models on the entire TCGA dataset for prediction of MammaPrint and OncotypeDx scores, as well as three models trained for cross validation (again using the CV3_odx85_mip and CV3_mp85_mip headers to specify the folds). Predictions will be made for the held out 1/3 of the data, and saved in the /PROJECTS/UCH_RS/eval/ folder.

Model training within this script is performed automatically using the slideflow train command, which can be set up as follows:

```
SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS"))
SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "tcga_brca_complete.csv")
SFP.sources = ["TCGA_BRCA_FULL_ROI"]
SFP.train(exp_label=exp_label, outcome_label_headers=odx_train_name,  val_outcome_label_headers=odx_val_name, params = hp, filters = mergeDict(filters, {odx_val_name: ["H","L"]}), val_strategy = 'k-fold-manual', val_k_fold=3, val_k_fold_header = "CV3_odx85_mip", multi_gpu=True, save_predictions=True)
#exp_label - defaults to ODX_Final_BRCAROI - saved label for the experiment
#outcome_label_headers - defaults to the recurrence score header in the TCGA dataset - GHI_RS_Model_NJEM.2004_PMID.15591335 for Oncotype and Pcorr_NKI70_Good_Correlation_Nature.2002_PMID.11823860 for MammaPrint
#params - specifies model hyperparameters using the 
```
	
## Model validation
To validate the trained models in an external dataset, run model_training.py with the -v flag:
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


