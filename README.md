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
The associated files can be downloaded to a project directory. Installation takes < 5 minutes on a standard desktop computer. All software was tested on Windows 10 version 1909, with a Intel Core i5-10210U processor with integrated Intel UHD graphics processor.

Requirements:
* python 3.7
* pandas 1.0.5
* cvxpy 1.1.7
* numpy 1.19.0
* IBM's CPLEX 12.10.0 and <a href='https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html'>CPLEX's python API</a>

## Overview
Stratification can be performed by loading data from an annotations file into a DataFrame. Stratification can be performed with the following function:
```python
def generate(data, category, values, crossfolds = 3, target_column = 'CV3', patient_column = 'submitter_id', site_column = 'SITE', timelimit = 100):
    ''' Generates 3 site preserved cross folds with optimal stratification of category
    Input:
        data: dataframe with slides that must be split into crossfolds.
        category: the column in data to stratify by
        values: a list of possible values within category to include for stratification
        crossfolds: number of crossfolds to split data into
        target_column: name for target column to contain the assigned crossfolds for each patient in the output dataframe
        patient_column: column within dataframe indicating unique identifier for patient
        site_column: column within dataframe indicating designated site for a patient
        timelimit: maximum time to spend solving
    Output:
        dataframe with a new column, 'CV3' that contains values 1 - 3, indicating the assigned crossfold
    '''
```

## Example
Please see test.py for example use applied to the accompanying 'example.csv' data file.  In our tests on above hardware, execution of this code takes 0.34 seconds.
```python
import preservedsite.crossfolds as cv
import pandas as pd

data = pd.read_csv("example.csv", dtype=str)
data = cv.generate(data, "feature", ["A", "B"], crossfolds=3, patient_column='patient', site_column='site')
```

The resulting segregation of sites into crossfolds is printed as follows, with the appropriate assignment of patients to folds for cross validation appended to the dataframe.
```
Crossfold 1: A - 54 B - 251  Sites: ['Site 0', 'Site 7', 'Site 9', 'Site 10', 'Site 11', 'Site 13', 'Site 19', 'Site 28']
Crossfold 2: A - 54 B - 250  Sites: ['Site 1', 'Site 2', 'Site 4', 'Site 5', 'Site 8', 'Site 14', 'Site 15', 'Site 18', 'Site 20', 'Site 21', 'Site 22', 'Site 23', 'Site 25', 'Site 26', 'Site 30', 'Site 32', 'Site 34', 'Site 36']
Crossfold 3: A - 54 B - 250  Sites: ['Site 3', 'Site 6', 'Site 12', 'Site 16', 'Site 17', 'Site 24', 'Site 27', 'Site 29', 'Site 31', 'Site 33', 'Site 35', 'Site 37']
```

<b>Note:</b> <a href="https://www.ibm.com/support/pages/note-reproducibility-cplex-runs">due to the nature of how CPLEX arrives at solutions</a>, slightly different results are expected on different hardware than what is specified above. Unfortunately, testing suggested CPLEX reproducibility parameters such as feasibility tolerance, optimilty tolernace, and Markowitz tolerance still does not ensure identical results on different hardware. Additionally, CPLEX will continue to search for solutions until either it arrives at an optimal solution or a pre-specified time limit is exhausted. This can be set with the timelimit parameter of the generate function above. While in our tests, optimal solutions were achieved in <1 second, it is possible for example code to run for up to 100s by default searching for optimal solutions, depending on the specific hardware used.

## Reproduction
To recreate our cross validation setup for <a href="https://www.nature.com/articles/s41467-021-24698-1">our work describing batch effect in TCGA</a>, clinical annotations should be downloaded from https://www.cbioportal.org/ for TCGA datasets for the six cancers of interest. Immune subtype annotations were obtained from the work of <a href="https://pubmed.ncbi.nlm.nih.gov/29628290/">Thorsson et al</a>, and genomic ancestry annotations were obtained from <a href="https://www.cell.com/cancer-cell/pdfExtended/S1535-6108(20)30211-7">Carrot-Zhang et al</a>. The CSV files with annotations from cbioportal can then be loaded into a dataframe as per the above example to generate 3 groups of sites for preserved site cross validation.

To generate the folds used for cross validation, we ran our preserved site cross validation tool for each feature of interest:
```
data = cv.generate(data, "ER status", ["Positive", "Negative"], crossfolds=3, patient_column='patient', site_column='site')
data = cv.generate(data, "BRCA mutation", ["Present", "Absent"], crossfolds=3, patient_column='patient', site_column='site')
...
```
Given the non-deterministic nature of CPLEX, we have included the genereated splits used for our work to allow for replication in the Annotation Files folder.


