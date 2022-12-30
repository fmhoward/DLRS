from functools import reduce
import math
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import numpy as np
from scipy import stats
from math import exp
from sklearn import preprocessing
import numpy.random as rng
from scipy.stats import pearsonr, spearmanr, ttest_ind
from tableone import TableOne
from os.path import join
import slideflow as sf
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import csv

CUR_OUT_FILE = ""
RUN_FROM_OLD_STATS = True
SUFFIX = ""
PROJECT_ROOT = os.getcwd() + "/PROJECTS/"
correlation = "spearman"
exp_suffix = "Final_BRCAROI"
odx_train = "odx_train"
mp_train = "mp_train"

def LogReg():
    """Returns a standardized logistic regression model for use across training
    """
    return LogisticRegression()


def find_eval(project, label, outcome, epoch=1, kfold = None):
    """Finds matching eval directory.
    Parameters
    ----------
    project (slideflow.Project): Project.
    label (str): Experimental label.
    outcome (str, optional): Outcome name. If none, uses default
        (biscuit.utils.OUTCOME). Defaults to None.
    epoch (int, optional): Epoch number of saved model. Defaults to None.
    kfold (int, optional): K-fold iteration. Defaults to None.

    Returns
    -------
    str: path to eval directory
    """
    tail = '' if kfold is None else f'-kfold{kfold}'
    matching = [
        o for o in os.listdir(project.eval_dir)
        if o[11:] == f'{outcome}-{label}-HP0{tail}_epoch{epoch}'
    ]
    if len(matching) > 1:
        msg = f"Multiple matching eval experiments found for label {label}"
        raise Exception(msg)
    elif not len(matching):
        raise Exception(f"No matching eval found for label {label}")
    else:
        return join(project.eval_dir, matching[0])

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Parameters
    ----------
    x - a 1D numpy array
    
    Returns
    -------
    array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks, weighted.
    Parameters
    ----------
    x - a 1D numpy array
    
    Returns
    -------
    array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC
    
    Adapted from the following work:
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
       
    Parameters
    ----------
    predictions_sorted_transposed - a 2D numpy.array[n_classifiers, n_examples] sorted such as the examples with label "1" are first
    label_1_count - number of examples with label "1"
    sample_weight - weight for each sample
    
    Returns
    -------
    AUROC, covariance
    
    """
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)


def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC, weighted
    
    Adapted from the following work:
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
       
    Parameters
    ----------
    predictions_sorted_transposed - a 2D numpy.array[n_classifiers, n_examples] sorted such as the examples with label "1" are first
    label_1_count - number of examples with label "1"
    sample_weight - weight for each sample
    
    Returns
    -------
    AUROC, covariance

    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC
    
    Adapted from the following work:
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
       
    Parameters
    ----------
    predictions_sorted_transposed - a 2D numpy.array[n_classifiers, n_examples] sorted such as the examples with label "1" are first
    label_1_count - number of examples with label "1"
    
    Returns
    -------
    AUROC, covariance

    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """
    Adapted from the following work:
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
       
    Parameters
    ----------
    aucs - 1D array of AUCs
    sigma - AUC DeLong covariances
    Returns
    -------
    z-statistic, log10(pvalue)

    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return z, np.log10(2) + stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight):
    """
    Computes ground truht statistics
    Adapted from the following work:
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
       
    Parameters
    ----------
    ground_truth - array of labels for sample
    sample_weights  - weight for each sample
    
    Returns
    -------
    order - ordered version of ground truths
    label_1_count - number of examples with label "1"
    ordered_sample_weight - ordered version of sample weight

    """
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Adapted from the following work:
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
       
    Parameters
    ----------
    ground_truth - np.array of labels for sample (0 or 1)
    predictions - predictions: np.array of floats of the probability of being class 1
    
    Returns
    -------
    AUROC, covariance
    
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def r_to_z(r):
    """Convert from a Pearson's r distribution to normal distribution

    Parameters
    ----------
    r - valuein Pearson's r distribution

    Returns
    -------
    z - value in normal distribution
    """
    return math.log((1 + r) / (1 - r)) / 2.0

def z_to_r(z):
    """Convert from a normalized distribution to Pearson's r distribution to correlation

    Parameters
    ----------
    z - value in normal distribution

    Returns
    -------
    r - valuein Pearson's r distribution
    """
    e = math.exp(2 * z)
    return((e - 1) / (e + 1))

def r_confidence_interval(r, alpha, n):
    """Calculate confidence interval for a Pearson correlation coefficient

    Parameters
    ----------
    r - Correlation coefficient
    alpha - significance level
    n - number of patients

    Returns
    -------
    tuple of low and high range of confidence interval
    """
    z = r_to_z(r)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha/2)  # 2-tailed z critical value

    lo = z - z_crit * se
    hi = z + z_crit * se

    # Return a sequence
    return (z_to_r(lo), z_to_r(hi))


##Saved optimal features for clinical model generated using 10 fold cross validation
MP_vars = ['grade', 'tumor_size', 'PR', 'lymph_vasc_inv', 'ductal', 'mucinous', 'medullary', 'metaplasia', 'race_asian', 'race_black']
ODX_vars = ['grade', 'tumor_size', 'PR', 'ductal', 'lobular', 'ductlob', 'medullary', 'metaplasia', 'race_black']


def load_data(filename=None, savefile=None, loadfile=None, lower=True, rec_score = 'odx'):
    """Load data from NCDB and print # excluded at each step

    Parameters
    ----------
    filename - name of csv NCDB PUF file
    savefile - True if you want to save the resultant dataframe as CSV (saves as 'saved_' + filename)
    loadfile - True if you want to load a cleaned NCDB dataframe (from 'saved_' + filename)
    lower - True if the column headers are lowercase
    rec_score - 'odx' if you want to select patients with Oncotype testing, 'mp' to select patients with MammaPrint testing
    
    Returns
    -------
    dataframe with loaded NCDB dataset
    """
    
    if loadfile:
        try:
            df_ret = pd.read_csv(loadfile, dtype=str)
            return df_ret
        except:
            print("ERROR - UNABLE TO LOAD SAVED NCDB FILE - RELOADING FROM NCDB2017.csv")
    fields = ['AGE',  'RACE', 'spanish_hispanic_origin', 'insurance_status', 'cdcc_total_best', 'year_of_diagnosis', 'histology', 'behavior', 'GRADE', 'TUMOR_SIZE', 'REGIONAL_NODES_POSITIVE', 'tnm_clin_t', 'TNM_CLIN_M', 'tnm_path_t', 'TNM_PATH_M', 'analytic_stage_group', 'cs_mets_at_dx', 'lymph_vascular_invasion', 'CS_SITESPECIFIC_FACTOR_16', 'CS_SITESPECIFIC_FACTOR_22', 'CS_SITESPECIFIC_FACTOR_23', 'DX_LASTCONTACT_DEATH_MONTHS', 'PUF_VITAL_STATUS']
    fieldname = ['age',  'race', 'spanish', 'insurance', 'cdcc', 'year', 'histology', 'behavior', 'grade', 'tumor_size', 'regional_nodes_positive', 'tnm_clin_t', 'tnm_clin_m', 'tnm_path_t', 'tnm_path_m', 'analytic_stage_group', 'cs_mets_at_dx','lvi', 'receptors', 'recurrence_assay', 'recurrence_score', 'last_contact', 'alive']

    if lower:
        fields = [f.lower() for f in fields]
    df = pd.read_csv(filename, usecols = fields)

    df.columns = fieldname

    df = df[df.year.astype(int) >= 2010]
    df = df[df.age.astype(float) != 999]
    df = df[df.race.astype(float) != 99]
    df = df[df.spanish.astype(float) != 9]
    df = df[df.grade.astype(float).isin([1,2,3,4])]
    df.loc[df.grade == 4] = 3

    df = df[~(df.tumor_size.astype(float) > 995)]
    df = df[~(df.regional_nodes_positive.isin([98,99]))]
    if rec_score == 'odx':
        df = df[(df.recurrence_assay.astype(float) == 10)]
        df = df[(df.recurrence_score.astype(float) < 101)]
    elif rec_score == 'mp':
        df = df[(df.recurrence_assay.astype(float) == 20)]
        df = df[(df.recurrence_score.astype(float).isin([200, 205, 400]))]
        df.loc[df.recurrence_score == 205, 'recurrence_score'] = 400
    df = df[df.behavior.astype(float) == 3]
    df = df[~df.tnm_clin_m.astype(str).str.contains('1', regex=False)]
    df = df[~df.tnm_path_m.astype(str).str.contains('1', regex=False)]
    df = df[~df.cs_mets_at_dx.astype(float).isin([10,40,42,44,50,60])]
    df = df[~df.analytic_stage_group.astype(str).str.contains('4', regex=False)]
    df = df.drop(columns=['tnm_clin_m', 'tnm_path_m', 'cs_mets_at_dx', 'analytic_stage_group'])
    df = df[(df.receptors.astype(float).isin([100, 10, 110]))]
    #Parse race
    df['race_parse'] = 0 #non-hispanic white
    df.loc[((df.race == 2) & (df.spanish == 0)), 'race_parse'] = 1 #non-hispanic black
    df.loc[df.spanish > 0, 'race_parse'] = 2  # hispanic
    df.loc[((df.race == 98) & (df.spanish == 0)), 'race_parse'] = 3 #other
    df.loc[((df.race == 3) & (df.spanish == 0)), 'race_parse'] = 4 #Native american
    df.loc[((~df.race.isin([0,1,2,3,98,99])) & (df.spanish == 0)), 'race_parse'] = 5 #asian / pacific islander
    df['race_asian'] = 0
    df.loc[~df.race.isin([0,1,2,3,98,99]), 'race_asian'] = 1
    df['race_black'] = 0
    df.loc[df.race_parse == 1, 'race_black'] = 1
    df['race_hispanic'] = 0
    df.loc[df.race_parse == 2, 'race_hispanic'] = 1
    df['race_other'] = 0
    df.loc[df.race_parse > 2, 'race_other'] = 1
    #parse tumor size
    df.loc[df.tumor_size == 990, 'tumor_size'] = 1
    df.loc[df.tumor_size == 991, 'tumor_size'] = 5 # < 1 cm
    df.loc[df.tumor_size == 992, 'tumor_size'] = 15 # 1-2 cm
    df.loc[df.tumor_size == 993, 'tumor_size'] = 25 # 2-3 cm
    df.loc[df.tumor_size == 994, 'tumor_size'] = 35 # 3-4 cm
    df.loc[df.tumor_size == 995, 'tumor_size'] = 45 # 4-5 cm
    df['node_positive'] = 0
    df.loc[(df.regional_nodes_positive > 0) & (df.regional_nodes_positive < 98), 'node_positive'] = 1

    if savefile:
        df.to_csv(savefile, index=False)
    return df


def applybinMIP(data, category, values, name):
    """Uses mixed integer programming to solve for the best splits of sites within TCGA that maintain an
    even ratio of the outcome of interest. Presumes a column 'patient' with patient names and 'SITE' with sites

    Parameters
    ----------
    data - input dataframe to optimize
    category - column to optimize
    values - potential values for the column to optimize
    name - name for optimized column

    Returns
    -------
    dataframe with cross folds listed in column name
    """

    submitters = data['patient'].unique()
    newData = pd.DataFrame(submitters, columns=['patient'])
    newData2 = data[['patient', category, 'SITE']]
    newData3 = pd.merge(newData, newData2, on='patient', how='left')
    newData3.drop_duplicates(inplace=True)

    uniqueSites = newData2['SITE'].unique()

    n = len(uniqueSites)
    listSet = []
    for v in values:
        listOrder = []
        for s in uniqueSites:
            listOrder += [len(newData3[(newData3.SITE == s) & (newData3[category] == v)].index)]
        listSet += [listOrder]
    import cvxpy as cp
    g1 = cp.Variable(n, boolean=True)
    g2 = cp.Variable(n, boolean=True)
    g3 = cp.Variable(n, boolean=True)
    A = np.ones(n)
    constraints = [g1 + g2 + g3 == A]
    error = ""
    for v in range(len(values)):
        if v == 0:
            error = cp.square(cp.sum(3*cp.multiply(g1, listSet[0])) - sum(listSet[0])) + cp.square(cp.sum(3*cp.multiply(g2, listSet[0])) - sum(listSet[0])) + cp.square(cp.sum(3*cp.multiply(g3, listSet[0])) - sum(listSet[0]))
        else:
            error += cp.square(cp.sum(3 * cp.multiply(g1, listSet[v])) - sum(listSet[v])) + cp.square(
                cp.sum(3 * cp.multiply(g2, listSet[v])) - sum(listSet[v])) + cp.square(
                cp.sum(3 * cp.multiply(g3, listSet[v])) - sum(listSet[v]))
    prob = cp.Problem(cp.Minimize(error), constraints)
    import cplex
    prob.solve(solver='CPLEX', cplex_params={"timelimit": 100})
    g1gs = []
    g2gs = []
    g3gs = []
    for i in range(n):
        if g1.value[i] > 0.5:
            g1gs += [uniqueSites[i]]
        if g2.value[i] > 0.5:
            g2gs += [uniqueSites[i]]
        if g3.value[i] > 0.5:
            g3gs += [uniqueSites[i]]
    bins = pd.DataFrame()
    ds1 = newData3[newData3.SITE.isin(g1gs)]
    ds1[name] = str(1)
    ds2 = newData3[newData3.SITE.isin(g2gs)]
    ds2[name] = str(2)
    ds3 = newData3[newData3.SITE.isin(g3gs)]
    ds3[name] = str(3)
    bins = bins.append(ds1)
    bins = bins.append(ds2)
    bins = bins.append(ds3)
    return pd.merge(data, bins[['patient', name]], on='patient', how='left')


def createCrossfoldsRS(df):
    """Creates cross folds and categorical cutoffs for Oncotype and MammaPrint assays based on NCDB / MINDACT trial
    Presumses RNA expression based OncotypeDx assay is stored in the global variable odx_train, and
    RNA expression based MammaPrint assay is stored in the global variable mp_train

    Parameters
    ----------
    df - input dataframe with Oncotype / MammaPrint assay values

    Returns
    -------
    dataframe with the following additional columns:
    odx85 - an indicator column for high risk Oncotype result, assuming cutoff of 85th percentile of HR+/HER2- patients
    mphr - an indicator column for high risk MammaPrint result, assuming cutoff of bottom 36th percentile
    mpuhr - an indicator column for ultra high risk MammaPrint result, assuming cutoff of bottom 18th percentile
    mpulr - an indicator column for ultra low risk MammaPrint result, assuming cutoff of top 85th percentile
    CV3_odx85_mip, CV3_mphr_mip, CV3_mpuhr_mip, CV3_mpulr_mip - splits of patients to use for site preserved cross validation
    """
    global odx_train, mp_train
    #per mindact
    pcorrULR = df[mp_train].quantile(0.85)
    #per mindact
    pcorrHR = df[mp_train].quantile(0.36)
    #cutpoint used to define UHR in trials was half of high risk patients
    pcorrUHR = df[mp_train].quantile(0.18)

    dfsubset = df[(df.ER_Status_By_IHC == 'positive') | (df.PR == 1)]
    dfsubset = dfsubset[dfsubset.HER2Calc == 'negative']
    #per NCDB
    rsavg = dfsubset[odx_train].quantile(0.85)

    df['odx85'] = 0
    df.loc[df[odx_train].astype(float) > rsavg, 'odx85'] = 1
    df = applybinMIP(df, 'odx85', [0, 1], 'CV3_odx85_mip')
    df['mpulr'] = 0
    df.loc[df[mp_train].astype(float) > pcorrULR, 'mpulr'] = 1
    df['mphr'] = 0
    df.loc[df[mp_train].astype(float) < pcorrHR, 'mphr'] = 1
    df['mpuhr'] = 0
    df.loc[df[mp_train].astype(float) < pcorrUHR, 'mpuhr'] = 1
    df = applybinMIP(df, 'mpulr', [0, 1], 'CV3_mpulr_mip')
    df = applybinMIP(df, 'mphr', [0, 1], 'CV3_mphr_mip')
    df = applybinMIP(df, 'mpuhr', [0, 1], 'CV3_mpuhr_mip')
    return df

def plot_features(dict):
    """Provides a visual plot of sequential forward feature selection results

    Parameters
    ----------
    dict - dict providing a result of sequential forward feature selection
    """
    fig, ax = plt.subplots()
    old_feats = []
    label_rename = {'grade':'Grade', 'PR':'PR', 'ductal':'Ductal','tumor_size':'Tumor Size','lobular':'Lobular','race_black':'Black','medullary':'Medullary','metaplasia':'Metaplastic','ductlob':'Ductal & Lobular', 'lymph_vasc_inv':'LVI','race_asian':'Asian','chest_wall':'Chest Wall Disease','race_hispanic':'Hispanic','skin_changes':'Skin Changes','tubular':'Tubular','papillary':'Papillary','inflammatory':'Inflammatory','regional_nodes_positive':'Nodes Positive','paget':'Paget','mucinous':'Mucinous','race_other':'Other Race','age':'Age'}
    label_list = []
    allvals =[]
    for i in dict:
        this_dict = dict[i]
        allvals += [this_dict['avg_score']]
    norm = mpl.colors.Normalize(vmin=np.min(allvals)-0.05, vmax=np.max(allvals))
    for i in dict:
        this_dict = dict[i]
        cur_feats = this_dict['feature_names']
        dif_feats = [c for c in cur_feats if c not in old_feats]
        label_list += [label_rename[dif_feats[0]]]
        old_feats = cur_feats
        min_err = this_dict['avg_score'] - np.min(this_dict['cv_scores'])
        max_err = np.max(this_dict['cv_scores']) - this_dict['avg_score']
        cmap = mpl.cm.Blues
        ax.errorbar(i, this_dict['avg_score'], yerr = [[min_err], [max_err]], color=cmap(norm(this_dict['avg_score'])), fmt='.', capsize=2, elinewidth=1, markeredgewidth=1)
    plt.ylabel("AUROC")
    plt.xticks(range(len(dict)+1), [""] + label_list, rotation=90)
    plt.savefig("Feature Plot.png")  
    plt.clf()

def feature_selection(X, Y, model, k_max):
    """Performs sequential forward feature selection to identify parameters with optimal ROC over 10 cross fold validation

    Parameters
    ----------
    X - model inputs
    Y - target output
    model - model to optimize
    k_max - maximum number of features to select
    """

    from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    feat = SFS(model, k_features=k_max, forward=True, floating=False, verbose=2, scoring='roc_auc', cv=10)
    feat.fit(X, Y)
    print(feat.subsets_)
    print(feat.get_support())
    plot_features(feat.subsets_)


def prepare_plot_hp_search(ax, df, param = "loss"):
    """Plots a figure for the trend in tile auc from bayesian hyperparameter optimizaiton

    Parameters
    ----------
    ax - Matplotlib axis
    df - dataframe of optimization steps
    param - hyperparamater to include as color of individual points on plot
    """
    x_list = []
    y_list = []
    for index, row in df.iterrows():
        x_list += [index + 1]
        y_list += [float(row["avg_tile_auc"])]
    ax.plot(x_list, y_list)
    if param == "loss":
        loss = ["mean_squared_error", "mean_absolute_error", "sparse_categorical_crossentropy"]
        loss_labels={"mean_squared_error":"Mean Squared Error", "mean_absolute_error":"Mean Absolute Error", "sparse_categorical_crossentropy":"Sparse Categorical Crossentropy"}
        s= None
        for l in loss:
            x_list = []
            y_list = []
            for index, row in df.iterrows():
                if row["loss"] == l:
                    x_list += [index + 1]
                    y_list += [float(row["avg_tile_auc"])]
            s = ax.scatter(x_list, y_list, label=loss_labels[l])
        ax.legend(loc = "lower right")
        cb = plt.colorbar(s, ax=ax, label = "remove")
    label_dict = {"batch_size":"Batch Size", "dropout":"Dropout", "hidden_layers":"Hidden Layers", "learning_rate":"Learning Rate"}
    if param in label_dict:
        x_list = []
        y_list = []
        params = []
        for index, row in df.iterrows():
            x_list += [index + 1]
            y_list += [float(row["avg_tile_auc"])]
            params += [float(row[param])]
        s = ax.scatter(x_list, y_list, c=params, cmap=cm.inferno)
        plt.colorbar(s, ax=ax, label = label_dict[param])
    ax.set_xlabel("Iteration")


def plot_hp_search(f = "hpopt.csv"):
    """Plots a figure with four subplots illustrating the hyperparameter search

    Parameters
    ----------
    f - file to with iterations from hyperparameter search
    """
    fig, ax = plt.subplots(2,2, dpi=300, figsize=(15,10), sharey=True)
    df = pd.read_csv(f)
    prepare_plot_hp_search(ax[0][0], df, "loss")
    prepare_plot_hp_search(ax[0][1], df, "batch_size")
    prepare_plot_hp_search(ax[1][0], df, "dropout")
    prepare_plot_hp_search(ax[1][1], df, "learning_rate")
    ax[0][0].set_ylabel("Average Tile-Level AUROC")
    ax[1][0].set_ylabel("Average Tile-Level AUROC")
    x_off = -4
    y_off = 0.77
    ax[0][0].annotate("a", (x_off, y_off), annotation_clip = False, fontsize=12, weight='bold')
    ax[1][0].annotate("b", (x_off, y_off), annotation_clip = False, fontsize=12, weight='bold')
    ax[0][1].annotate("c", (x_off, y_off), annotation_clip = False, fontsize=12, weight='bold')
    ax[1][1].annotate("d", (x_off, y_off), annotation_clip = False, fontsize=12, weight='bold')
    plt.savefig("HP Search.png")
    plt.clf()

def getNCDBClassifications(ncdbfile = "NCDB2017.csv", loadfile = False, outcome = 'odx'):
    """Gets a copy of the NCDB database with appropriate column annotations for feature selection

    Parameters
    ----------
    ncdbfile - location of the NCDB csv file
    loadfile - False if the short version of the NCDB file needs to be created; True if the short version is already
    created and can be loaded
    outcome - 'odx' for Oncotype and 'mp' for MammaPrint

    Returns
    -------
    dataframe with NCDB column annotations needed for feature selection / classification
    """
    if not loadfile:
        if outcome == 'odx':
            df = load_data(filename=ncdbfile, savefile=r"NCDB2017_saved2.csv", rec_score = outcome)
        elif outcome == 'mp':
            df = load_data(filenae=ncdbfile, savefile=r"NCDB2017_saved_mp.csv", rec_score = outcome)
    else:
        if outcome == 'odx':
            df = load_data(loadfile="NCDB2017_saved2.csv")
        elif outcome == 'mp':
            df = load_data(loadfile="NCDB2017_saved_mp.csv")
    df['PR'] = 1
    df.loc[df.receptors.astype(float) == 100, 'PR'] = 0
    df['ductal'] = 0
    df.loc[df.histology.astype(float).isin([8500, 8501, 8502, 8503, 8505, 8506, 8507, 8508, 8523, 8230]), 'ductal'] = 1
    df['lobular'] = 0
    df.loc[df.histology.astype(float).isin([8520, 8521, 8523, 8524, 8524, 8525]), 'lobular'] = 1
    df['ductlob'] = 0
    df.loc[df.histology.astype(float).isin([8522]), 'ductlob'] = 1
    df['mucinous'] = 0
    df.loc[df.histology.astype(float).isin([8480, 8481]), 'mucinous'] = 1
    df['papillary'] = 0
    df.loc[df.histology.astype(float).isin([8503, 8504, 8260, 8050, 8051, 8052]), 'papillary'] = 1
    df['tubular'] = 0
    df.loc[df.histology.astype(float).isin([8211, 8210]), 'tubular'] = 1
    df['inflammatory'] = 0
    df.loc[df.histology.astype(float).isin([8530]), 'inflammatory'] = 1
    df['medullary'] = 0
    df.loc[df.histology.astype(float).isin([8510, 8512, 8513, 8514]), 'medullary'] = 1
    df['metaplasia'] = 0
    df.loc[df.histology.astype(float).isin([8570, 8571, 8572, 8573, 8574, 8575, 8576]), 'metaplasia'] = 1
    df['paget'] = 0
    df.loc[df.histology.astype(float).isin([8540, 8541, 8542, 8543]), 'paget'] = 1
    df['lymph_vasc_inv'] = 0.5
    df.loc[df.lvi == 0, 'lymph_vasc_inv'] = 0
    df.loc[df.lvi == 1, 'lymph_vasc_inv'] = 1
    df['chest_wall'] = 0
    df['skin_changes'] = 0
    df.loc[df.tnm_clin_t.astype(str).str.lower().str.contains('4a', regex=False), 'chest_wall'] = 1
    df.loc[df.tnm_clin_t.astype(str).str.lower().str.contains('4b', regex=False), 'skin_changes'] = 1
    df.loc[df.tnm_clin_t.astype(str).str.lower().str.contains('4c', regex=False), 'chest_wall'] = 1
    df.loc[df.tnm_clin_t.astype(str).str.lower().str.contains('4c', regex=False), 'skin_changes'] = 1
    df.loc[df.tnm_clin_t.astype(str).str.lower().str.contains('4d', regex=False), 'inflammatory'] = 1
    df.loc[df.tnm_path_t.astype(str).str.lower().str.contains('4a', regex=False), 'chest_wall'] = 1
    df.loc[df.tnm_path_t.astype(str).str.lower().str.contains('4b', regex=False), 'skin_changes'] = 1
    df.loc[df.tnm_path_t.astype(str).str.lower().str.contains('4c', regex=False), 'chest_wall'] = 1
    df.loc[df.tnm_path_t.astype(str).str.lower().str.contains('4c', regex=False), 'skin_changes'] = 1
    df.loc[df.tnm_path_t.astype(str).str.lower().str.contains('4d', regex=False), 'inflammatory'] = 1
    if outcome == 'odx':
        df['high_odx'] = 0
        df.loc[df.recurrence_score.astype(float) > 25, 'high_odx'] = 1
    elif outcome == 'mp':
        df['high_mp'] = 0
        df.loc[df.recurrence_score.astype(float) == 400, 'high_mp'] = 1
    return df

def compareClassifiersNCDB(ncdbfile = "NCDB2017.csv", loadfile = False, outcome = 'odx'):
    """Performs sequential forward feature selection to identify optimal input variables to include in subsequent models
    Output is printed to console
    Parameters
    ----------
    ncdbfile - location of the NCDB csv file
    loadfile - False if the short version of the NCDB file needs to be created; True if the short version is already
    created and can be loaded
    outcome - 'odx' for Oncotype and 'mp' for MammaPrint
    """
    df = getNCDBClassifications(ncdbfile, loadfile, outcome)
    variable_list = ['age', 'grade', 'tumor_size', 'PR', 'regional_nodes_positive', 'chest_wall', 'skin_changes', 'inflammatory', 'lymph_vasc_inv', 'ductal', 'lobular', 'ductlob', 'mucinous', 'papillary', 'tubular', 'medullary', 'metaplasia', 'paget', 'race_asian', 'race_black', 'race_hispanic', 'race_other']
    feature_selection(df[variable_list], df[['high_' + outcome]].values.ravel(), LogisticRegression(), len(variable_list))

def getNCDBClassifier(ncdbfile = "NCDB2017.csv", loadfile = False, outcome = 'odx'):
    """Gets NCDB classifier using saved
    Parameters
    ----------
    ncdbfile - location of the NCDB csv file
    loadfile - False if the short version of the NCDB file needs to be created; True if the short version is already
    created and can be loaded
    outcome - 'odx' for Oncotype and 'mp' for MammaPrint
    """

    df = getNCDBClassifications(ncdbfile, loadfile, outcome)
    variable_list = None
    if outcome == 'odx':
        variable_list = ODX_vars
    elif outcome == 'mp':
        variable_list = MP_vars
    return LogisticRegression(max_iter = 1000).fit(df[variable_list], np.ravel(df[['high_' + outcome]]))

def applySizeCutoffs(df):
    """Applies size cutoffs to TCGA dataset from T stage
    Parameters
    ----------
    df - dataframe with TCGA dataset

    Returns
    ----------
    df - dataframe with size cutoffs applied
    """
    df.loc[df.AJCC_T == 't1', 'tumor_size'] = 10
    df.loc[df.AJCC_T == 't1a', 'tumor_size'] = 2.5
    df.loc[df.AJCC_T == 't1b', 'tumor_size'] = 7.5
    df.loc[df.AJCC_T == 't1c', 'tumor_size'] = 15
    df.loc[df.AJCC_T == 't2', 'tumor_size'] = 35
    df.loc[df.AJCC_T == 't2a', 'tumor_size'] = 35
    df.loc[df.AJCC_T == 't2b', 'tumor_size'] = 35
    df.loc[df.AJCC_T == 't3', 'tumor_size'] = 50
    df.loc[df.AJCC_T == 't3a', 'tumor_size'] = 50
    df.loc[df.AJCC_T == 't4', 'tumor_size'] = 50
    df.loc[df.AJCC_T == 't4b', 'tumor_size'] = 50
    df.loc[df.AJCC_T == 't4d', 'tumor_size'] = 50
    df.loc[df.AJCC_T == 'tx', 'tumor_size'] = 35
    return df

def roundp(p, d):
    if round(p, d) == 0:
        return p
    else:
        return round(p, d)

def prognosticPlot(df, column, round_hazard = 2):
    """Generates a string with prognostic accuracy of variable in column

    Parameters
    ----------
    df - dataframe with recurrence data ('year_recur', and 'recur') and column to assess prognostic accuracy
    column - column to assess for prognostic accuracy

    Returns
    ----------
    String with hazard ratio, confidence interval, and p-value; normalized hazard ratio, confidence interval, and p-value; and concordance index
    """
    cph = CoxPHFitter()
    cph2 = CoxPHFitter()
    if type(column) != list:
        column = [column]
    dfcph = df[['year_recur', 'recur'] + column].dropna()
    if len(dfcph[dfcph.recur == 1].index) <= 1:
        return "Two few events to estimate"
    cph.fit(dfcph, duration_col='year_recur', event_col='recur')

    
    #if column[0] == 'comb': #no longer log scaling the combined model
        #dfcph[column[0]] = np.log(dfcph[column[0]])
        
        
    dfcph[column[0]] = dfcph[column[0]] / dfcph[column[0]].std()
    cph2.fit(dfcph, duration_col='year_recur', event_col='recur')
    try:
        ci1 = round(exp(cph.confidence_intervals_["95% lower-bound"][0]), round_hazard)
    except OverflowError:
        ci1 = float('inf')
    try:
        ci2 = round(exp(cph.confidence_intervals_["95% upper-bound"][0]), round_hazard)
    except OverflowError:
        ci2 = float('inf')

    try:
        ci3 = round(exp(cph2.confidence_intervals_["95% lower-bound"][0]), round_hazard)
    except OverflowError:
        ci3 = float('inf')
    try:
        ci4 = round(exp(cph2.confidence_intervals_["95% upper-bound"][0]), round_hazard)
    except OverflowError:
        ci4 = float('inf')

    return column[0] + "," + str(round(cph.hazard_ratios_[0], round_hazard)) + " (" + str(ci1) + " - " + str(ci2) + "), " + str(roundp(cph.summary['p'][0], 3)) + "," + str(round(cph2.hazard_ratios_[0], round_hazard)) + " (" + str(ci3) + " - " + str(ci4) + "), " + str(roundp(cph2.summary['z'][0], 2)) + ", " + str(roundp(cph2.summary['p'][0], 3)) + "," + str(round(cph.concordance_index_, 3))

def prognosticPlotThreshold(df, ax = None, column = None, limits = "UCMC"):
    """Generates a plot of comparative prognostic accuracy for two groups identified by a threshold column

    Parameters
    ----------
    df - dataframe with recurrence data ('year_recur', and 'recur') and threshold column
    ax - Matplotlib axis for plotting
    column - threshold column to assess for prognostic accuracy
    limits - set to UCH to set limits for plotting UCMC dataset; otherwise will plot limits for TCGA dataset

    Returns
    ----------
    String with hazard ratio and confidence interval for Cox model fit with the target threshold column
    """
    cph = CoxPHFitter()
    dfcph = df[['year_recur', 'recur'] + [column]].dropna()
    if len(dfcph[dfcph.recur == 1].index) <= 1:
        return "Too Few Events"
    g1 = dfcph[dfcph[column] == 0]
    g2 = dfcph[dfcph[column] == 1]
    cph.fit(dfcph, duration_col='year_recur', event_col='recur')
    if ax:
        kmf1 = KaplanMeierFitter()
        kmf2 = KaplanMeierFitter()
        kmf1.fit(g1['year_recur'], g1['recur'], label='Predicted Low Risk').plot_survival_function(ax=ax, ci_show=False)
        kmf2.fit(g2['year_recur'], g2['recur'], label='Predicted High Risk').plot_survival_function(ax=ax, ci_show=False)
        add_at_risk_counts(kmf1, kmf2, ax=ax)
        ax.set_xlabel("Years")
        if limits == "UCMC":
            ax.set_ylim([0.8999, 1.001])
            ax.legend(loc = 'lower left')
            ax.annotate("HR " + str(round(cph.hazard_ratios_[0], 2)) + " (" + str(round(exp(cph.confidence_intervals_["95% lower-bound"][0]), 2)) + " - " + str(round(exp(cph.confidence_intervals_["95% upper-bound"][0]), 2)) + "), p = " + str(round(cph.summary['p'][0], 3)), (1.5, 0.93), annotation_clip=False, fontsize=12)

        else:
            ax.set_ylim([0.3, 1.001])
            ax.legend(loc = 'lower left')
            ax.annotate("HR " + str(round(cph.hazard_ratios_[0], 2)) + " (" + str(round(exp(cph.confidence_intervals_["95% lower-bound"][0]), 2)) + " - " + str(round(exp(cph.confidence_intervals_["95% upper-bound"][0]), 2)) + "), p = " + str(round(cph.summary['p'][0], 3)), (8, 0.9), annotation_clip=False, fontsize=12)
        ax.set_ylabel("Recurrence-Free Interval")
    if len(g1[g1.recur == 1]) == 0:
        return str("N/A (No events)")

    return str(round(cph.hazard_ratios_[0], 3)) + " (" + str(round(exp(cph.confidence_intervals_["95% lower-bound"][0]), 3)) + " - " + str(round(exp(cph.confidence_intervals_["95% upper-bound"][0]), 3)) + ")"


def plotAUROC(ax, df, outcome, predictor, n_bootstraps = None, predictor_name = None, plot_stdev = False):
    """Plots receiver operating characteristic curve for predictor for outcome in dataframe

    Parameters
    ----------
    ax - Matplotlib axis for plotting
    df - Dataframe with outcome and predictor information
    outcome - outcome column in dataframe
    predictor - predictor column in dataframe
    n_bootstraps - None or number. If None, will calculate AUROC and variance with Delong's method. If number, will calculate confidence intervals with bootstrapping
    predictor_name - rename predictor for plot
    plot_stdev - True if plotting standard deviation of ROC

    Returns
    ----------
    String with AUROC and confidence intervals
    """
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    aucs = []
    prcs = []
    if n_bootstraps:
        X = df[predictor].to_numpy()
        Y = df[outcome].to_numpy()
        for i in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = rng.randint(0, len(Y), len(Y))
            if len(np.unique(Y[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue
            fpr, tpr, _ = roc_curve(Y[indices], X[indices])
            aucs += [auc(fpr, tpr)]            
            precision, recall, _ = precision_recall_curve(Y[indices], X[indices])
            prcs += [auc(recall, precision)]
            tpr = np.interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)
    else:
        fpr, tpr, _ = roc_curve(df[outcome], df[predictor])
        #precision, recall, _ = precision_recall_curve(df[outcome], df[predictor])
        aucs += [auc(fpr, tpr)]
        #prcs += [auc(recall, precision)]
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
        
        #Calculate prcs:
        X = df[predictor].to_numpy()
        Y = df[outcome].to_numpy()
        for i in range(1000):
            # bootstrap by sampling with replacement on the prediction indices
            indices = rng.randint(0, len(Y), len(Y))
            if len(np.unique(Y[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue
            precision, recall, _ = precision_recall_curve(Y[indices], X[indices])
            prcs += [auc(recall, precision)]
            
    auc_delong, auc_cov_delong = delong_roc_variance(df[outcome].to_numpy(), df[predictor].to_numpy())
    auc_std_delong = np.sqrt(auc_cov_delong)
    alpha = 0.95
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(lower_upper_q, loc=auc_delong, scale=auc_std_delong)
    ci[ci > 1] = 1
    if not n_bootstraps:
        lb = ci[0]
        ub = ci[1]
    else:
        lb = np.percentile(aucs, 2.5)
        ub = np.percentile(aucs, 97.5)
    aucs = sum(aucs)/len(aucs)
    
    lb_prc = np.percentile(prcs, 2.5)
    ub_prc = np.percentile(prcs, 97.5)
    prcs = sum(prcs)/len(prcs)
    
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    ax1, ax2 = tprs.shape
    std = 1.95 * tprs.std(axis=0) / math.sqrt(ax1)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    if predictor_name is None:
        predictor_name = predictor
    if ax:
        ax.plot(base_fpr, mean_tprs, label = predictor_name + " AUC {:.3f}".format(aucs) + " [{:.3f} - ".format(lb) + "{:.3f}]".format(ub))
        if plot_stdev:
            ax.fill_between(base_fpr, tprs_lower, tprs_upper, alpha=0.3)
    return "{:.3f}".format(aucs) + " ({:.3f} - ".format(lb) + "{:.3f})".format(ub) + "," + "{:.3f}".format(prcs) + " ({:.3f} - ".format(lb_prc) + "{:.3f})".format(ub_prc) + "," + "{:.3f}".format(df[outcome].mean())

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count, _ = compute_ground_truth_statistics(ground_truth, None)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, None)
    return calc_pvalue(aucs, delongcov)

def plotAUROCODX(df, ax, columns, names, outcome):
    """Plots receiver operating characteristic curves for predictor of outcome of interest

    Parameters
    ----------
    df - Dataframe with outcome and predictor information
    ax - Matplotlib axis for plotting
    columns - columns containing predictors to plot
    names - names of predictors to plot
    outcome - outcome of interest ('odx_cat' for Oncotype, 'mp_cat' for MammaPrint)

    Returns
    ----------
    Dictionary containing AUC and confidence interval for each predictor
    """
    ret_dict = {}
    for c,n in zip(columns, names):
        res = plotAUROC(ax, df, outcome = outcome, predictor = c, predictor_name = n, n_bootstraps = None)
        ret_dict[c] = res + ","
    z, p = delong_roc_test(df[outcome], df[columns[1]], df['comb'])
    ret_dict[columns[1]] += str(round(z[0][0],3)) + "," + str(round(10 ** p[0][0], 4))
    z, p = delong_roc_test(df[outcome], df['percent_tiles_positive0'], df['comb'])
    ret_dict['percent_tiles_positive0'] += str(round(z[0][0],3)) + "," + str(round(10 ** p[0][0], 4))
    ret_dict['comb'] += "-"
    if ax:
        ax.legend()
        ax.plot([0, 1], [0, 1],'k--')
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        ax.set_title("UCMC (Validation)")
        #ax.annotate("b", (-0.05, 1.05), annotation_clip = False, fontsize=12, weight='bold')
    return ret_dict

def calcSensSpec(df, columns, outcome):
    """Calculates the sensitivty and specificty of predictor for outcome, computed at specific threshold

    Parameters
    ----------
    df - Dataframe with outcome and predictor information
    columns - columns to calculate sensitivty / specificity
    outcome - outcome of interest for prediction

    Returns
    ----------
    Dictionary containing sensitiivty and specificity for each thresholded predictor
    """
    thresh_dict = {}
    for c in columns:
        thresh_dict[c] = {}
        sens = len(df[(df[c + "_thresh"] == 1) & (df[outcome] == 1)].index) / len(df[(df[outcome] == 1)].index)
        spec = len(df[(df[c + "_thresh"] == 0) & (df[outcome] == 0)].index) / len(df[(df[outcome] == 0)].index)
        ppv = len(df[(df[c + "_thresh"] == 1) & (df[outcome] == 1)].index) / len(df[(df[c + "_thresh"] == 1)].index)
        npv = len(df[(df[c + "_thresh"] == 0) & (df[outcome] == 0)].index) / len(df[(df[c + "_thresh"] == 0)].index)

        thresh_dict[c]['Sen'] = sens
        thresh_dict[c]['Spec'] = spec
        thresh_dict[c]['PPV'] = ppv
        thresh_dict[c]['NPV'] = npv
    return thresh_dict

def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    """Return an axes of confidence bands using a simple approach.

    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

    """
    if ax is None:
        ax = plt.gca()

    ci = t * s_err * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color=["#b9cfe7"], label = "95% Confidence Interval")

    return ax

def plotCorrelation(df, ax, outcome, predictor, predictor_name):
    """Plots correlation between predictor and numerical outcome

    Parameters
    ----------
    df - Dataframe with outcome and predictor information
    ax - Matplotlib axis for plot
    outcome - numerical outcome of interest for prediction
    predictor - column for predictor variable in dataframe
    predictor_name - string predictor name for plotting

    Returns
    ----------
    String with correlation coefficient and confidence interval

    """
    global odx_train, mp_train
    if predictor == 'comb':
        #df[predictor] = np.log(df[predictor]) #- no longer log scaling the combined model
        predictor_name = 'Combined Model'
        #ax.set_xscale("log") 
    if predictor == 'pred_score':
        predictor_name = 'Clinical Model'
    x = df[predictor].values
    y = df[outcome].values
    if outcome == 'RS' or outcome == odx_train:
        df_low = df[df["RS_cat"] == 0]
        df_high = df[df["RS_cat"] == 1]
    elif outcome == 'mpscore' or outcome == mp_train:
        df_low = df[df["mp_cat"] == 0]
        df_high = df[df["mp_cat"] == 1]
    x_low = df_low[predictor].values
    y_low = df_low[outcome].values
    x_high = df_high[predictor].values
    y_high = df_high[outcome].values
    def equation(a, b):
        """Return a 1D polynomial."""
        return np.polyval(a, b)

    p, cov = np.polyfit(x, y, 1, cov=True)  # parameters and covariance from of the fit of 1-D polynom.
    y_model = equation(p, x)  # model using the fit parameters; NOTE: parameters here are coefficients

    # Statistics
    n = len(x)  # number of observations
    m = p.size  # number of parameters
    dof = n - m  # degrees of freedom
    t = stats.t.ppf(0.975, n - m)  # used for CI and PI bands

    # Estimates of Error in Data/Model
    resid = y - y_model
    chi2 = np.sum((resid / y_model) ** 2)  # chi-squared; estimates error in data
    chi2_red = chi2 / dof  # reduced chi-squared; measures goodness of fit
    s_err = np.sqrt(np.sum(resid ** 2) / dof)  # standard deviation of the error
    global correlation    
    pearson_r, pearson_p = pearsonr(x, y)
    corr_r, corr_p = spearmanr(x, y)

    # Plotting --------------------------------------------------------------------

    ax.plot(
        x_low, y_low, linestyle = 'None', marker='.', label = 'Low Risk'
    )

    ax.plot(
        x_high, y_high, linestyle = 'None', marker='.', label = 'High Risk'
    )
    # Fit
    corr_string = "r"
    if correlation == "spearman":
        corr_string = "ρ"
    ax.plot(x, y_model, "-", color="0.1", linewidth=1.5, alpha=0.5, label="r = " + str(round(pearson_r,2)) + ", ρ = " + str(round(corr_r,2)))

    x2 = np.linspace(np.min(x), np.max(x), 100)
    y2 = equation(p, x2)

    # Confidence Interval (select one)
    plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax)

    # Prediction Interval
    pi = t * s_err * np.sqrt(1 + 1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
    ax.plot(x2, y2 - pi, "--", color="0.5", label="95% Prediction Limits")
    ax.plot(x2, y2 + pi, "--", color="0.5")
    ax.legend()
    ax.set_xlabel(predictor_name) 
    lb, ub = r_confidence_interval(pearson_r, 0.05, len(df.index))
    lb_p, ub_p = r_confidence_interval(corr_r, 0.05, len(df.index))
    return "{:.3f}".format(pearson_r) + " ({:.3f} - ".format(lb) + "{:.3f})".format(ub) + "," + "{:.3f}".format(corr_r) + " ({:.3f} - ".format(lb_p) + "{:.3f})".format(ub_p)

def plotCorrelationODX(df, columns, names, outcome):
    """Plots correlation between predictor and numerical outcome

    Parameters
    ----------
    df - Dataframe with outcome and predictor information
    columns - list of predictor variables in dataframe
    names - list of names for preidctor variables
    outcome - numerical outcome of interest for prediction

    Returns
    ----------
    Dict with correlation coefficient and confidence interval for each predictor

    """
    ret_dict = {}
    fig, ax = plt.subplots(1, 3, dpi = 300, sharey=True, figsize=(12,6))
    for i, (c,n) in enumerate(zip(columns, names)):
        ret_dict[c] = plotCorrelation(df, ax = ax[i], outcome = outcome, predictor = c, predictor_name = n)
        if i == 0:
            ax[i].set_ylabel('Recurrence Score')

    return ret_dict

def plotAUROCrecur(df, columns, plot = False):
    """Calculates the AUROC for prediction of recurrence as a binary variable

    Parameters
    ----------
    df - Dataframe with recurrence and predictor information
    columns - list of predictor variables in dataframe to compute AUROC for recurrence
    plot - True to show results as plot

    Returns
    ----------
    list of AUROCs for each predictor

    """
    if plot:
        fig, ax = plt.subplots()
    aucs = []
    for c in columns:
        fpr, tpr, _ = roc_curve(df['recur'], df[c])
        a = auc(fpr, tpr)
        aucs += [a]
        if plot:
            ax.plot(fpr, tpr, label = c + " auc: " + str(a))
    if plot:
        ax.legend()
        plt.clf()
    return aucs
    
def predictLogisticRegression(clf, input_cols):
    return clf.predict_proba(input_cols)[:, 1]

def plotROCCV(ax, df, cv_column, plot_stdev = False, outcome = "odx85", predictor = None, model = False, model_input = None, n_bootstraps = 0, predictor_name = None, alternate_train = True):
    """Plot ROC curves and calculate correlation with outcome, averaged over 3 cross folds

    Parameters
    ----------
    ax - axis for plotting
    df - Dataframe with outcome and predictor information
    cv_column - column that specifies crossfold
    plot_stdev - True to plot standard deviation on plot
    outcome - outcome variable within dataframe
    predictor - predictor variable column
    model - if True, will compute the predictor as a logistic regression from multiple columns
    model_input - input variables for logistic regression, if model is True
    n_bootstraps - None or number. If none, will simply plot the average AUROC over 3 cross folds. Otherwise, will include bootstraps for each cross fold prediction
    predictor_name - Desired string name of predictor for display in plot
    alternate_train - Specifies to fit logistic regression using data from other two crossfolds
    
    Returns
    ----------
    Correlation coefficient (and confidence interval) and AUROC (and confidence interval) for predictor and outcome variable

    """
    global odx_train, mp_train
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    aucs = []
    prcs = []
    rs = []
    ps = []
    for i in [1,2,3]:
        df_temp = df[df[cv_column] == float(i)].copy()
        if model:
            clf = None
            if alternate_train:
                df_train = df[df.cv_linear != float(i)].copy()
                clf = LogReg().fit(df_train[model_input], df_train[outcome])                
            else:
                clf = LogReg().fit(df_temp[model_input], df_temp[outcome])
            df_temp["comb"] = predictLogisticRegression(clf, df_temp[model_input])
            predictor = "comb"
            #z, p = delong_roc_test(df_temp[outcome], df_temp['ten_score'], df_temp['comb'])
            #print(round(z[0][0],3))
            #print(round(10 ** p[0][0], 3))
            ##z, p = delong_roc_test(df_temp[outcome], df_temp['comb'], df_temp['percent_tiles_positive0'])
            #print(round(z[0][0],3))
            #print(round(10 ** p[0][0], 3))
            
        r = 0
        p = 0
        global correlation
        if outcome == 'odx85_cat' and SUFFIX != '_REV':
            r, _ = pearsonr(df_temp[predictor], df_temp[odx_train])
            p, _ = spearmanr(df_temp[predictor], df_temp[odx_train])

        elif outcome == 'mp_cat' and SUFFIX != '_REV':
            r, _ = pearsonr(df_temp[predictor], df_temp[mp_train])
            p, _ = spearmanr(df_temp[predictor], df_temp[mp_train])

        rs += [r]
        ps += [p]
        aucs_sub = []
        if n_bootstraps:
            X = df_temp[predictor].to_numpy()
            Y = df_temp[outcome].to_numpy()
            for i in range(n_bootstraps):
                # bootstrap by sampling with replacement on the prediction indices
                indices = rng.randint(0, len(Y), len(Y))
                if len(np.unique(Y[indices])) < 2:
                    # We need at least one positive and one negative sample for ROC AUC
                    # to be defined: reject the sample
                    continue
                fpr, tpr, _ = roc_curve(Y[indices], X[indices])
                aucs += [auc(fpr, tpr)]
                aucs_sub += [auc(fpr, tpr)]
                precision, recall, _ = precision_recall_curve(Y[indices], X[indices])
                prcs += [auc(recall, precision)]
                tpr = np.interp(base_fpr, fpr, tpr)
                tpr[0] = 0.0
                tprs.append(tpr)
            lb_sub = np.percentile(aucs_sub, 2.5)
            ub_sub = np.percentile(aucs_sub, 97.5)
            aucs_sub = sum(aucs_sub)/len(aucs_sub)
            #print("{:.3f}".format(aucs_sub) + " ({:.3f} - ".format(lb_sub) + "{:.3f})".format(ub_sub))
        else:
            fpr, tpr, _ = roc_curve(df[outcome], df[predictor])
            aucs += [auc(fpr, tpr)]
            precision, recall, _ = precision_recall_curve(df[outcome], df[predictor])
            prcs += [auc(recall, precision)]
            tpr = np.interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)
            
    lb = np.percentile(aucs, 2.5)
    ub = np.percentile(aucs, 97.5)
    aucs = sum(aucs)/len(aucs)
    lb_prc = np.percentile(prcs, 2.5)
    ub_prc = np.percentile(prcs, 97.5)
    prcs = sum(prcs)/len(prcs)
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    ax1, ax2 = tprs.shape
    std = 1.95 * tprs.std(axis=0) / math.sqrt(ax1)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    if predictor_name is None:
        predictor_name = predictor
    ax.plot(base_fpr, mean_tprs, label = predictor_name + " AUC {:.3f}".format(aucs) + " [{:.3f} - ".format(lb) + "{:.3f}]".format(ub))
    if plot_stdev:
        ax.fill_between(base_fpr, tprs_lower, tprs_upper, alpha=0.3)
    avg_r = sum(rs)/len(rs)
    avg_p = sum(ps)/len(ps)
    p_lb, p_ub = r_confidence_interval(avg_p, 0.05, len(df.index))
    r_lb, r_ub = r_confidence_interval(avg_r, 0.05, len(df.index))
    return "{:.3f}".format(avg_r) + " ({:.3f} - ".format(r_lb) + "{:.3f})".format(r_ub) +"," + "{:.3f}".format(avg_p) + " ({:.3f} - ".format(p_lb) + "{:.3f})".format(p_ub) +"," + "{:.3f}".format(aucs) + " ({:.3f} - ".format(lb) + "{:.3f})".format(ub) +"," + "{:.3f}".format(prcs) + " ({:.3f} - ".format(lb_prc) + "{:.3f})".format(ub_prc) + "," + "{:.3f}".format(df[outcome].mean())

def plotROC(df1, ax, outcome, clin_predictor, avg_coefs):
    """Plot ROC curve for predictors averaged over 3 cross folds

    Parameters
    ----------
    df1 - dataframe with predictors and outcomes
    ax - axis for plotting
    outcome - outcome variable within dataframe
    clin_predictor - column name for clinical model
    avg_coefs - if True, will average coefficients for fit logisitc regression over 3 cross folds. Otherwise will pool predictors and generate a single logisitc regression

    Returns
    ----------
    Dict containing Correlation coefficient (and confidence interval) and AUROC (and confidence interval) for each predictor

    """
    df = df1.copy()
    ret_vals = {}
    if avg_coefs:
        clin_pred_str = "Clinical Model"
        if clin_predictor == "ten_score":
            clin_pred_str = "Clinical Nomogram"
        ret_vals["percent_tiles_positive0"] = plotROCCV(ax, df, "cv_linear", plot_stdev=True, outcome = outcome, predictor = "percent_tiles_positive0", model = None, model_input = None, predictor_name="Pathologic Model", n_bootstraps = 1000)
        ret_vals[clin_predictor] = plotROCCV(ax, df, "cv_linear", plot_stdev=True, outcome = outcome, predictor = clin_predictor, model = None, model_input = None, predictor_name=clin_pred_str, n_bootstraps = 1000)
        ret_vals["comb"] = plotROCCV(ax, df, "cv_linear", plot_stdev=True, outcome = outcome, predictor = None, model = True, model_input = ["percent_tiles_positive0", clin_predictor], predictor_name="Combined Model", n_bootstraps = 1000)
    else:
        fpr, tpr, _ = roc_curve(df[outcome], df["percent_tiles_positive0"])
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, label="auc DL: " + str(a))

        fpr, tpr, _ = roc_curve(df[outcome], df["ten_score"])
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, label="auc ten_score: " + str(a))

        fpr, tpr, _ = roc_curve(df[outcome], df["comb"])
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, label="auc comb: " + str(a))

    ax.legend()
    ax.plot([0, 1], [0, 1],'k--')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.set_title("TCGA (Training)")
    #ax.annotate("a", (-0.05, 1.05), annotation_clip = False, fontsize=12, weight='bold')
    return ret_vals

def calcClinModelTCGA(df1):
    """Prepares all clinical model variables for calculation for the TCGA dataset
    Preperation performed in place on input dataframe

    Parameters
    ----------
    df1 - input dataframe
    """
    df1['ten_score'] = 0
    df1['ten_score_age'] = df1['age_at_diagnosis']
    df1.loc[df1.ten_score_age < 19, 'ten_score_age'] = 19
    df1.loc[df1.ten_score_age > 90, 'ten_score_age'] = 90
    df1['ten_score_age'] = 9 - 9*(df1['ten_score_age'] - 19)/71
    df1['ten_score_size'] = df1['tumor_size']
    df1.loc[df1.ten_score_size < 6, 'ten_score_size'] = 6
    df1.loc[df1.ten_score_size > 50, 'ten_score_size'] = 50
    df1['ten_score_size'] = 31*(df1['ten_score_size'] - 6)/44
    df1['ten_grade'] = df1['Grade']
    df1.loc[df1.ten_grade == 1, 'ten_grade'] = 0
    df1.loc[df1.ten_grade == 2, 'ten_grade'] = 38
    df1.loc[df1.ten_grade == 3, 'ten_grade'] = 100
    df1['ten_pr'] = df1['PR']
    df1.loc[df1.PR == 1, 'ten_pr'] = 0
    df1.loc[df1.PR == 0, 'ten_pr'] = 70
    df1['ten_type'] = 25
    df1.loc[df1['2016 Histology Annotations'] == 'Invasive ductal carcinoma', 'ten_type'] = 31
    df1.loc[df1['2016 Histology Annotations'] == 'Invasive lobular carcinoma', 'ten_type'] = 0
    df1.loc[df1['2016 Histology Annotations'] == 'Mixed', 'ten_type'] = 10
    df1['ten_score_age'] = pd.to_numeric(df1['ten_score_age'],errors='coerce').fillna(value=df1['ten_score_age'].mean())
    df1['ten_pr'] = pd.to_numeric(df1['ten_pr'],errors='coerce')
    df1['PR'] = df1['PR'].fillna(value=df1['PR'].mean())
    df1['ten_pr'] = df1['ten_pr'].fillna(value=df1['ten_pr'].mean())
    df1['ten_score_size'] = pd.to_numeric(df1['ten_score_size'],errors='coerce').fillna(value=df1['ten_score_size'].mean())
    df1['ten_grade'] = pd.to_numeric(df1['ten_grade'], errors='coerce').fillna(value=df1['ten_grade'].mean())
    df1['ten_type'] = pd.to_numeric(df1['ten_type'],errors='coerce').fillna(value=df1['ten_type'].mean())
    df1['ten_score'] = df1['ten_score_age'] + df1['ten_score_size'] + df1['ten_grade'] + df1['ten_pr'] + df1['ten_type']
    df1['ten_score'] = df1['ten_score'].fillna(value=df1['ten_score'].mean())

    #Also calculate variables as needed for the clinical predictive model from NCDB
    df1['age'] = df1['age_at_diagnosis']
    df1['inflammatory'] = 0
    df1.loc[df1.AJCC_T == 't4d', 'inflammatory'] = 1
    df1['lymph_vasc_inv'] = 0
    df1.loc[df1['Lymphovascular Invasion (LVI)'] == 'Present', 'lymph_vasc_inv'] = 1
    df1['grade'] = df1['Grade']
    df1['ductal'] = 0
    df1['lobular'] = 0
    df1['medullary'] = 0
    df1['metaplasia'] = 0
    df1['paget'] = 0
    df1['ductlob'] = 0
    df1['race_black'] = 0
    df1['race_asian'] = 0
    df1['mucinous'] = 0
    df1.loc[df1['2016 Histology Annotations'] == 'Invasive ductal carcinoma', 'ductal'] = 1
    df1.loc[df1['2016 Histology Annotations'] == 'Invasive lobular carcinoma', 'lobular'] = 1
    df1.loc[df1['2016 Histology Annotations'] == 'Invasive carcinoma with medullary features', 'medullary'] = 1
    df1.loc[df1['2016 Histology Annotations'] == 'Metaplastic carcinoma', 'metaplasia'] = 1
    df1.loc[df1['2016 Histology Annotations'] == 'Mixed', 'ductlob'] = 1
    df1.loc[df1['2016 Histology Annotations'] == 'Mucinous carcinoma', 'mucinous'] = 1
    df1.loc[df1['Histology_Class_ICD'] == 'Paget dis. & infil. duct carcinoma', 'paget'] = 1
    df1.loc[df1['RACE_repo'] == 'Black', 'race_black'] = 1
    df1.loc[df1['RACE_repo'] == 'Asian', 'race_asian'] = 1
    df1['grade'] = pd.to_numeric(df1['grade'], errors = 'coerce').fillna(value = df1['grade'].mean())

def calcClinModelUCMC(df1):
    """Prepares all clinical model variables for calculation for the University of Chicago dataset
    Preperation performed in place on input dataframe

    Parameters
    ----------
    df1 - input dataframe
    """

    df1['ten_score'] = 0
    df1['ten_score_age'] = df1['Age']
    df1.loc[df1.ten_score_age < 19, 'ten_score_age'] = 19
    df1.loc[df1.ten_score_age > 90, 'ten_score_age'] = 90
    df1['ten_score_age'] = 9 - 9*(df1['ten_score_age'] - 19)/71
    df1['ten_score_size'] = df1['tumor_size']
    df1.loc[df1.ten_score_size < 6, 'ten_score_size'] = 6
    df1.loc[df1.ten_score_size > 50, 'ten_score_size'] = 50
    df1['ten_score_size'] = 31*(df1['ten_score_size'] - 6)/44
    df1['ten_grade'] = df1['grade']
    df1.loc[df1.ten_grade == 1, 'ten_grade'] = 0
    df1.loc[df1.ten_grade == 2, 'ten_grade'] = 38
    df1.loc[df1.ten_grade == 3, 'ten_grade'] = 100
    df1['ten_pr'] = df1['PR']
    df1.loc[df1.ten_pr == 'Pos', 'ten_pr'] = 0
    df1.loc[df1.ten_pr == 'Neg', 'ten_pr'] = 70
    df1['ten_type'] = 25
    df1.loc[df1.hist_type == 'Ductal', 'ten_type'] = 31
    df1.loc[df1.hist_type == 'Lobular', 'ten_type'] = 0
    df1.loc[df1.hist_type == 'D&L', 'ten_type'] = 10
    df1['ten_score_age'] = pd.to_numeric(df1['ten_score_age'],errors='coerce').fillna(value=df1['ten_score_age'].mean())
    df1['ten_grade'] = pd.to_numeric(df1['ten_grade'],errors='coerce').fillna(value=df1['ten_grade'].mean())
    df1['ten_pr'] = pd.to_numeric(df1['ten_pr'],errors='coerce')
    df1['ten_pr'] = df1['ten_pr'].fillna(value=df1['ten_pr'].mean())
    df1['ten_score_size'] = pd.to_numeric(df1['ten_score_size'],errors='coerce').fillna(value=df1['ten_score_size'].mean())
    df1['ten_type'] = pd.to_numeric(df1['ten_type'],errors='coerce').fillna(value=df1['ten_type'].mean())
    df1['ten_score'] = df1['ten_score_age'] + df1['ten_score_size'] + df1['ten_grade'] + df1['ten_pr'] + df1['ten_type']
    df1['ten_score'] = df1['ten_score'].fillna(value=df1['ten_score'].mean())
    df1['inflammatory'] = 0
    df1['grade'] = pd.to_numeric(df1['grade'], errors='coerce').fillna(value=df1['grade'].mean())
    df1['tumor_size'] = pd.to_numeric(df1['tumor_size'], errors='coerce').fillna(value=df1['tumor_size'].mean())
    df1['age'] = df1['Age']
    df1.loc[df1.PR == 'Pos', 'PR'] = 1
    df1.loc[df1.PR == 'Neg', 'PR'] = 0
    df1['PR'] = pd.to_numeric(df1['PR'], errors='coerce')
    df1['PR'] = df1['PR'].fillna(value=df1['PR'].mean())
    df1['lymph_vasc_inv'] = 0.5
    df1.loc[df1['lymphvascularinvasion'] == 1, 'lymph_vasc_inv'] = 1
    df1.loc[df1['lymphvascularinvasion'] == 0, 'lymph_vasc_inv'] = 0
    df1['ductal'] = 0
    df1.loc[df1.hist_type == 'Ductal', 'ductal'] = 1
    df1['lobular'] = 0
    df1.loc[df1.hist_type == 'Lobular', 'lobular'] = 1
    df1['ductlob'] = 0
    df1.loc[df1.hist_type == 'D&L', 'lobular'] = 1
    df1['medullary'] = 0
    df1['metaplasia'] = 0
    df1.loc[df1.hist_type == 'Metaplastia', 'metaplasia'] = 1
    df1['mucinous'] = 0
    df1.loc[df1.hist_type == 'Mucinous', 'mucinous'] = 1
    df1['paget'] = 0
    df1.loc[df1.hist_type == 'Paget disease', 'paget'] = 1
    df1['race_class'] = df1['race_1'].map({'01 White': 'White', '02 Black': 'Black',
                                     '04 Chinese': 'Asian', '05 Japanese': 'Asian', '06 Filipino': 'Asian',
                                     '08 Korean': 'Asian', '10 Vietnamese': 'Asian',
                                     '15 Asian Indian or Pakistani, NOS': 'Asian',
                                     '16 Asian Indian': 'Asian',
                                     '96 Other Asian, including Asian/Oriental, NOS': 'Asian',
                                     '03 American Indian, Aleutian, or Eskimo': 'Other', '98 Other': 'Other',
                                     '99 Unknown': np.nan})
    df1['race_black'] = 0
    df1.loc[df1.race_class == 'Black', 'race_black'] = 1
    df1['race_asian'] = 0
    df1.loc[df1.race_class == 'Asian', 'race_asian'] = 1


def getThresh(df, pred_path, pred_clin, outcome, avg_coefs = True, cutoff = 0.95, t_cutoff = True, interpolate = True, alternate_train = True):
    """Gets the threshold for models highly sensitivty for high recurrence score, computed in TCGA
    Parameters
    ----------
    df - input dataframe
    pred_path - pathologic predictive model name
    pred_clin - clinical predictive model name
    outcome - target outcome for prediction
    avg_coefs - if True, will average thresholds over each of three crossfolds
    cutoff - sensitivity / specificty cutoff
    t_cutoff - if true, will return a TPR (sensitivty) threshold using cutoff for target sensitivity
               if false will return a specifiicty threshold using cutoff for target specificity
    interpolate - Thresholds are interpolated between two nearest points (versus choosing first point exceeding cutoff)
    alternate_train - Specifies to fit logistic regression using data from other two crossfolds
    """

    df1 = df.copy()
    if avg_coefs:
        perform_met = {}
        perform_met[pred_path] = {'Sen':0, 'Spec':0, 'PPV':0, 'NPV':0, 'Thresh':0, 'ThreshList':[]}
        perform_met[pred_clin] = {'Sen':0, 'Spec':0, 'PPV':0, 'NPV':0, 'Thresh':0, 'ThreshList':[]}
        perform_met["comb"] = {'Sen':0, 'Spec':0, 'PPV':0, 'NPV':0, 'Thresh':0, 'ThreshList':[]}
        pred_cols = [pred_path, pred_clin]
        for i in [1, 2, 3]:
            df_temp = df1[df1.cv_linear == float(i)].copy()
            clf = None
            if alternate_train:
                df_train = df1[df1.cv_linear != float(i)].copy()
                clf = LogReg().fit(df_train[pred_cols], df_train[outcome])
            else:
                clf = LogReg().fit(df_temp[pred_cols], df_temp[outcome])
            #df_temp['comb'] = clf.predict_proba(df_temp[pred_cols])[:,1]
            df_temp['comb'] = predictLogisticRegression(clf, df_temp[pred_cols])
            for col in [pred_path, pred_clin, 'comb']:
                fpr, tpr, thresh = roc_curve(df_temp[outcome], df_temp[col])
                for ind, (f, t, th) in enumerate(zip(fpr, tpr, thresh)):
                    if t_cutoff and t >= cutoff:
                        if interpolate and t != cutoff:
                            alp = 1/(cutoff - tpr[ind-1])
                            bet = 1/(t - cutoff)
                            prev = len(df_temp[df_temp[outcome] == 1].index) / len(df_temp.index)
                            sen = (bet*t+alp*tpr[ind-1])/((alp+bet))
                            spec = (bet*(1-f)+alp*(1-fpr[ind-1]))/((alp+bet))
                            perform_met[col]['Sen'] += sen / 3
                            perform_met[col]['Spec'] += spec / 3
                            perform_met[col]['PPV'] += sen * prev / (3*((sen*prev) + (1-spec)*(1-prev)))
                            perform_met[col]['NPV'] += spec * (1 - prev) / (3 * ((spec)*(1 - prev) + (1 - sen)*(prev)))
                            perform_met[col]['Thresh'] += (bet*th+alp*thresh[ind-1])/(3*(alp+bet))
                            perform_met[col]['ThreshList'] += [(bet*th+alp*thresh[ind-1])/((alp+bet))]
                        else:
                            prev = len(df_temp[df_temp[outcome] == 1].index) / len(df_temp.index)
                            sen = t
                            spec = 1 - f
                            perform_met[col]['Sen'] += t/3
                            perform_met[col]['Spec'] += (1-f)/3
                            perform_met[col]['PPV'] += sen * prev / (3*((sen*prev) + (1-spec)*(1-prev)))
                            perform_met[col]['NPV'] += spec * (1 - prev) / (3 * ((spec)*(1 - prev) + (1 - sen)*(prev)))
                            perform_met[col]['Thresh'] += th/3
                            perform_met[col]['ThreshList'] += [th]
                        break
                    elif not t_cutoff and (1-f) <= cutoff:
                        if interpolate and (1-f) != cutoff:
                            alp = 1 / (cutoff - (1 - fpr[ind - 1]))
                            bet = 1 / ((1-f) - cutoff)
                            prev = len(df_temp[df_temp[outcome] == 1].index) / len(df_temp.index)
                            sen = (bet * t + alp * tpr[ind - 1]) / ((alp + bet))
                            spec = (bet * (1 - f) + alp * (1 - fpr[ind - 1])) / ((alp + bet))
                            perform_met[col]['Sen'] += sen/3
                            perform_met[col]['Spec'] += spec/3
                            perform_met[col]['PPV'] += sen * prev / (3*((sen*prev) + (1-spec)*(1-prev)))
                            perform_met[col]['NPV'] += spec * (1 - prev) / (3 * ((spec)*(1 - prev) + (1 - sen)*(prev)))
                            perform_met[col]['Thresh'] += (bet * th + alp * thresh[ind - 1]) / (3 * (alp + bet))
                            perform_met[col]['ThreshList'] += [(bet * th + alp * thresh[ind - 1]) / ((alp + bet))]
                        else:
                            prev = len(df_temp[df_temp[outcome] == 1].index) / len(df_temp.index)
                            sen = t
                            spec = 1 - f
                            perform_met[col]['Sen'] += t/3
                            perform_met[col]['Spec'] += (1-f)/3
                            perform_met[col]['Thresh'] += th/3
                            perform_met[col]['ThreshList'] += [th]
                            perform_met[col]['PPV'] += sen * prev / (3*((sen*prev) + (1-spec)*(1-prev)))
                            perform_met[col]['NPV'] += spec * (1 - prev) / (3 * ((spec)*(1 - prev) + (1 - sen)*(prev)))
                        break
        #perform_met[col]['Thresh'] = np.log(perform_met[col]['Thresh'])
        return perform_met
    raise NotImplementedError

def getClassifier(df1, pred_cols, outcome, avg_coefs=True, alternate_train = True):
    """Gets a logistic regression classifier fit to TCGA

    Parameters
    ----------
    df1 - input dataframe
    pred_cols - columsn to include in the logistic regression
    outcome - outcome to fit classifier to
    avg_coefs - if True, will average thresholds over each of three crossfolds
    alternate_train - Specifies to fit logistic regression using data from other two crossfolds
    
    Returns
    ----------
    Fitted logistic regression predicting outcome from pred_cols
    """

    if avg_coefs:
        avg_auc = []
        coefs = None
        intercepts = None
        for i in [1, 2, 3]:
            df_temp = df1[df1.cv_linear == float(i)].copy()
            clf = None
            if alternate_train:
                df_train = df1[df1.cv_linear != float(i)].copy()
                clf = LogReg().fit(df_train[pred_cols], df_train[outcome])
            else:
                clf = LogReg().fit(df_temp[pred_cols], df_temp[outcome])
            df_temp['comb'] = predictLogisticRegression(clf, df_temp[pred_cols])
            if coefs is None:
                coefs = clf.coef_
                intercepts = clf.intercept_
            else:
                coefs += clf.coef_
                intercepts += clf.intercept_
            fpr, tpr, thresh = roc_curve(df_temp[outcome], df_temp["comb"])
            a = auc(fpr, tpr)
            avg_auc += [a]
        intercepts = intercepts / 3
        coefs = coefs / 3

        clf = LogReg().fit(df1[pred_cols], df1[outcome])
        if avg_coefs:
            clf.coef_ = coefs
            clf.intercept_ = intercepts
            #print(coefs)
        return clf
    else:
        clf = LogReg().fit(df1[pred_cols], df1[outcome])
        return clf

def fitMultivariate(ax = None, avg_coefs=True, outcome="odx85_cat", NCDB=False, prognostic_plots = "none"):
    """Fits a Multivariate Model on NCDB 

    Parameters
    ----------
    ax - leftmost axis for plotting a dual plot with ROC curves for prediction of outcome
    avg_coefs - if true, will average logistic regression coefficients over three cross folds
    outcome - outcome to predict
    NCDB - if False, will use the Tennesse clincal model; otherwise will use models fit from NCDB
    prognostic_plots - 'roc' - plot training / validation ROC curves. 'all' - also plot survival curves using threshold as cutpoint. 'none' - do not save any plots

    Returns
    ----------
    Tuple of the following -
    Logistic regression combined model to predict outcome from TCGA
    (If NCDB) logisitc regression classifier fit in NCDB to predict outcome from clinical features
    Thresholds computed to optimize sensitivtiy for high risk outcome
    Correlation coeffcient and AUROC for models predicting outcome in TCGA
    """
    global odx_train, mp_train, exp_suffix
    pred_cols = ['percent_tiles_positive0', 'ten_score']
    if NCDB:
        pred_cols = ['percent_tiles_positive0', 'pred_score']
    df1 = pd.read_csv(join(PROJECT_ROOT, "UCH_RS", "tcga_brca_complete.csv"))
    if SUFFIX != "_REV":
        df1 = df1[(df1.ER_Status_By_IHC == 'positive') | (df1.PR == 1) ] 
        df1 = df1[df1.HER2Calc == 'negative'] 
    else:
        df1 = pd.read_csv(join(PROJECT_ROOT, "UCH_RS", "uch_brca_complete.csv"))
    df1 = df1.drop_duplicates(subset='patient')
    dfs = []
    for i in [1,2,3]:
        if RUN_FROM_OLD_STATS:
            dfa = pd.read_csv(join(PROJECT_ROOT, "saved_results", outcome + str(i)+ ".csv"))
        else:
            SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS"))
            if outcome == "odx85_cat":
                if SUFFIX != "_REV":
                    dfa =pd.read_csv(join(find_eval(SFP, "ODX_" + exp_suffix + SUFFIX, odx_train, 1, i), "patient_predictions_odx85_eval.csv"))
                else:
                    dfa = pd.read_csv(join(find_eval(SFP, "ODX_" + exp_suffix + SUFFIX, "RS", 1, i), "patient_predictions_RSHigh_eval.csv"))
            else:
                if SUFFIX != "_REV":
                    dfa = pd.read_csv(join(find_eval(SFP, "MP_" + exp_suffix + SUFFIX, mp_train, 1, i), "patient_predictions_mphr_eval.csv"))
                else:
                    dfa = pd.read_csv(join(find_eval(SFP, "MP_" + exp_suffix + SUFFIX, "mpscore", 1, i), "patient_predictions_MPHigh_eval.csv"))
        dfa['cv_linear'] = i
        dfs += [dfa]
    df1 = df1.merge(pd.concat(dfs), left_on="patient", right_on="patient", how="left")
    df1['odx85_cat'] = 0    
    if SUFFIX != "_REV":
        df1.loc[df1.odx85 == 'H', 'odx85_cat'] = 1
    else:
        df1.loc[df1.RSHigh == 'H', 'odx85_cat'] = 1

    df1['mp_cat'] = 0
    if SUFFIX != "_REV":
        df1.loc[df1.mphr == 'H', 'mp_cat'] = 1
    else:
        df1.loc[df1.MPHigh == 'H', 'mp_cat'] = 1

    df1 = df1.dropna(subset=[outcome, 'percent_tiles_positive0'])
    if outcome == "mp_cat":
        df1['percent_tiles_positive0'] = -1 * df1['percent_tiles_positive0']
    if SUFFIX != "_REV":
        calcClinModelTCGA(df1)
    else:
        calcClinModelUCMC(df1)
    if NCDB:
        if outcome == "odx85_cat":
            clf_ncdb = getNCDBClassifier(loadfile = True, outcome = 'odx')
            df1['pred_score'] = clf_ncdb.predict_proba(df1[ODX_vars])[:, 1]
        elif outcome == "mp_cat":
            clf_ncdb = getNCDBClassifier(loadfile = True, outcome = 'mp')
            df1['pred_score'] = clf_ncdb.predict_proba(df1[MP_vars])[:, 1]

    df1['cv_linear'] = df1['cv_linear'].astype(int)
    clf = getClassifier(df1, pred_cols, outcome, avg_coefs)
    df1['comb'] = predictLogisticRegression(clf, df1[pred_cols])


    thresh = getThresh(df1, 'percent_tiles_positive0', pred_cols[1], outcome, avg_coefs)
    if NCDB:
        thresh = getThresh(df1, 'percent_tiles_positive0', pred_cols[1], outcome, avg_coefs)
    if (prognostic_plots == 'tcga' or prognostic_plots == 'all') and SUFFIX != "_REV":
        for i in [1, 2, 3]:
            df_temp = df1[df1.cv_linear == float(i)].copy()
            df_train = df1[df1.cv_linear != float(i)].copy()
            clf_sub = LogReg().fit(df_train[pred_cols], df_train[outcome])
            df1.loc[df1.cv_linear == float(i), 'comb'] = predictLogisticRegression(clf_sub, df1.loc[df1.cv_linear == float(i),pred_cols])
            
        temp_dfs = []
        clin_model = pred_cols[1]
        dfprog = df1.copy() 
        global CUR_OUT_FILE
        f = open(CUR_OUT_FILE, 'a')
        writer = csv.writer(f)

        
        dfprog['recur'] = dfprog['Disease_Free_Status'].map({'censored':0,'progressed':1})
        dfprog['year_recur'] = dfprog['Disease_Free_Months'] / 12
        dfprog = dfprog[['percent_tiles_positive0', 'recur', 'year_recur', 'comb'] + [clin_model] + [outcome]].dropna().reset_index()
        writer.writerow(["**Disease Free Interval - TCGA**"])
        writer.writerow(["n = " + str(len(dfprog.index))])
        writer.writerow("Variable,HR (95% CI), p-value, HR (95% CI) scaled, z-statistic, p-value, C-Index, AUROC".split(","))        
        recur_plot_cph(dfprog, writer, ['percent_tiles_positive0'] + [clin_model] + ['comb'] + [outcome])
        writer.writerow("")
        dfprog = df1.copy()
        dfprog['recur'] = dfprog['Disease_Free_Status'].map({'censored':0,'progressed':1})
        dfprog['year_recur'] = dfprog['Disease_Free_Months'] / 12
        dfprog.loc[dfprog['Overall_Survival_Status'] == 'dead', 'recur'] = 1
        dfprog = dfprog[['percent_tiles_positive0', 'recur', 'year_recur', 'comb'] + [clin_model] + [outcome]].dropna().reset_index()
        writer.writerow(["**Disease Free Survival - TCGA**"])
        writer.writerow(["n = " + str(len(dfprog.index))])
        writer.writerow("Variable,HR (95% CI), p-value, HR (95% CI) scaled, z-statistic, p-value, C-Index, AUROC".split(","))        
        recur_plot_cph(dfprog, writer, ['percent_tiles_positive0'] + [clin_model] + ['comb'] + [outcome])
        writer.writerow([""])
        writer.writerow(["**Overall Survival - TCGA**"])
        dfprog = df1.copy()
        dfprog['recur'] = dfprog['Overall_Survival_Status'].map({'alive':0,'dead':1})
        dfprog['year_recur'] = dfprog['Overall_Survival_Months'] / 12        
        dfprog.loc[dfprog['Overall_Survival_Status'] == 'dead', 'recur'] = 1
        dfprog = dfprog[['percent_tiles_positive0', 'recur', 'year_recur', 'comb'] + [clin_model] + [outcome]].dropna().reset_index()
        writer.writerow(["n = " + str(len(dfprog.index))])
        writer.writerow("Variable,HR (95% CI), p-value, HR (95% CI) scaled, z-statistic, p-value, C-Index, AUROC".split(","))      
        recur_plot_cph(dfprog, writer, ['percent_tiles_positive0'] + [clin_model] + ['comb'] + [outcome])
        writer.writerow([""])

        roc_fig = plt.gcf()
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=300, sharex = True, num = 99)
        
        dfprog = df1.copy()
        dfprog['recur'] = dfprog['Disease_Free_Status'].map({'censored':0,'progressed':1})
        dfprog['year_recur'] = dfprog['Disease_Free_Months'] / 12
        dfprog = applyThresh(dfprog, clin_model, thresh, cv = True)   
        dfprog = dfprog[['recur', 'year_recur', 'comb_thresh', 'percent_tiles_positive0', 'percent_tiles_positive0_thresh'] + ['comb', clin_model, clin_model + '_thresh'] + [outcome]].dropna()
        prognosticPlotThreshold(dfprog, axs[0], clin_model + '_thresh', limits = "TCGA")
        prognosticPlotThreshold(dfprog, axs[1], 'comb_thresh', limits = "TCGA")
        
        axs[0].set_title("Clinical Nomogram (Target 95% Sensitivity for High Risk)")
        axs[0].annotate("a", (-5, 1.01), annotation_clip = False, fontsize=12, weight='bold')
        axs[1].set_title("Combined Model (Target 95% Sensitivity for High Risk)")
        axs[1].annotate("b", (-5, 1.01), annotation_clip = False, fontsize=12, weight='bold')
        f.close()
        plt.tight_layout()
        plt.savefig("Prognostic Plots TCGA " + outcome + ".png")
        plt.clf()
        plt.figure(roc_fig)
    ret = plotROC(df1, ax, outcome, pred_cols[1], avg_coefs)
    
    if NCDB:
        return clf, clf_ncdb, thresh, ret
    return clf, thresh, ret
    
def eval_model(df1, ax, clin_model, outcome, prognostic_plots = 'none'):
    """Evaluates a model trained on TCGA in the University of Chicago cohort

    Parameters
    ----------
    df1 - dataframe containing predictions
    ax - rightmost axis for plotting dual plot with ROC curves for prediction of outcome
    clin_model - name of column of clinical predictive model
    outcome - outcome to predict
    prognostic_plots - 'roc' - plot training / validation ROC curves. 'all' - also plot survival curves using threshold as cutpoint. 'none' - do not save any plots

    Returns
    ----------
    Dict with correlation coefficient, AUROC, and p-value for prediction of outcome for each predictive variable
    """
    global odx_train, mp_train
    df1['chemo'] = 0
    if SUFFIX != "_REV":
        df1.loc[df1.dup_chemo == 1, 'chemo'] = 1
    else:
        if outcome == "RS":
            outcome = odx_train
        elif outcome == "mpscore":
            outcome = mp_train
    dfout = df1.copy()
    if SUFFIX != "_REV":
        dfout = dfout[['percent_tiles_positive0', 'comb'] + [clin_model] + ['percent_tiles_positive0_thresh', 'comb_thresh'] + [clin_model + '_thresh'] + [outcome]].dropna()
    else:
        dfout = dfout[['percent_tiles_positive0', 'comb'] + [clin_model] + ['percent_tiles_positive0_thresh', 'comb_thresh'] + [clin_model + '_thresh'] + [outcome, "odx85", "mphr"]].dropna()        
    outcome_cat = 'RS_cat'
    if outcome == 'RS' or outcome == odx_train:
        dfout['RS_cat'] = 0
        if SUFFIX != "_REV":
            dfout['RS'] = pd.to_numeric(dfout['RS'])
            dfout.loc[dfout.RS > 25, 'RS_cat'] = 1
        else:
            dfout.loc[dfout.odx85 == 'H', 'RS_cat'] = 1
    if outcome == 'mpscore' or outcome == mp_train:
        dfout['mp_cat'] = 0
        outcome_cat = 'mp_cat'

        if SUFFIX != "_REV":
            dfout['mpscore'] = pd.to_numeric(dfout['mpscore'])
            dfout.loc[dfout.mpscore < 0, 'mp_cat'] = 1
        else:
            dfout.loc[dfout.mphr == 'H', 'mp_cat'] = 1

    clin_name = "Clinical Nomogram"
    if clin_model != "ten_score":
        clin_name = "Clinical Model"

    if prognostic_plots == 'all':
        r1 = plotAUROCODX(dfout, ax[1], ['percent_tiles_positive0', clin_model, 'comb'], ['Pathologic Model', clin_name, 'Combined Model'], outcome_cat)
    else:
        r1 = plotAUROCODX(dfout, ax[1], ['percent_tiles_positive0', clin_model, 'comb'],
                          ['Pathologic Model', clin_name, 'Combined Model'], outcome_cat)
    if prognostic_plots == 'roc' or prognostic_plots == 'all':
        plt.tight_layout()
        plt.savefig("ROC curves " + outcome + ".png")
        plt.clf()
    thresh = None
    if prognostic_plots == 'all' and SUFFIX != "_REV":
        thresh = calcSensSpec(dfout, ['percent_tiles_positive0', clin_model, 'comb'], outcome_cat)
        dfprog = df1.copy()
        dfprog = dfprog[['chemo', 'percent_tiles_positive0', 'recur', 'year_recur', 'dead', 'year_FU', 'comb', 'tumor_size'] + [clin_model] + [outcome]].dropna().reset_index()
        global CUR_OUT_FILE
        f = open(CUR_OUT_FILE, 'a')
        writer = csv.writer(f)
        
        writer.writerow(["**Recurrence Free Interval - Validation**"])
        writer.writerow("Variable,HR (95% CI), p-value, HR (95% CI) scaled, z-statistic, p-value, C-Index, AUROC".split(","))        
        writer.writerow(["No Chemo - n = " + str(len(dfprog[dfprog.chemo == 0].index))])
        recur_plot_cph(dfprog[dfprog.chemo == 0], writer, ['percent_tiles_positive0'] + [clin_model] + ['comb'] + [outcome])
        writer.writerow(["Chemo  - n = " + str(len(dfprog[dfprog.chemo == 1].index))])
        recur_plot_cph(dfprog[dfprog.chemo == 1], writer, ['percent_tiles_positive0'] + [clin_model] + ['comb'] + [outcome])
        writer.writerow(["Overall - n = " + str(len(dfprog.index))])
        recur_plot_cph(dfprog, writer, ['percent_tiles_positive0'] + [clin_model] + ['comb'] + [outcome])
        writer.writerow([""])

        writer.writerow(["**Recurrence Free Survival - Validation**"])
        dfprog.loc[dfprog.dead == 1, 'recur'] = 1
        writer.writerow("Variable,HR (95% CI), p-value, HR (95% CI) scaled, z-statistic, p-value, C-Index, AUROC".split(","))        
        writer.writerow(["No Chemo - n = " + str(len(dfprog[dfprog.chemo == 0].index))])
        recur_plot_cph(dfprog[dfprog.chemo == 0], writer, ['percent_tiles_positive0'] + [clin_model] + ['comb'] + [outcome])
        writer.writerow(["Chemo  - n = " + str(len(dfprog[dfprog.chemo == 1].index))])
        recur_plot_cph(dfprog[dfprog.chemo == 1], writer, ['percent_tiles_positive0'] + [clin_model] + ['comb'] + [outcome])
        writer.writerow(["Overall - n = " + str(len(dfprog.index))])
        recur_plot_cph(dfprog, writer, ['percent_tiles_positive0'] + [clin_model] + ['comb'] + [outcome])
        writer.writerow([""])
        
        writer.writerow(["**Overall Survival - Validation**"])
        dfprog['recur'] = dfprog['dead']
        dfprog['year_recur'] = dfprog['year_FU']
        writer.writerow("Variable,HR (95% CI), p-value, HR (95% CI) scaled, z-statistic, p-value, C-Index, AUROC".split(","))        
        writer.writerow(["No Chemo - n = " + str(len(dfprog[dfprog.chemo == 0].index))])
        recur_plot_cph(dfprog[dfprog.chemo == 0], writer, ['percent_tiles_positive0'] + [clin_model] + ['comb'] + [outcome])
        writer.writerow(["Chemo  - n = " + str(len(dfprog[dfprog.chemo == 1].index))])
        recur_plot_cph(dfprog[dfprog.chemo == 1], writer, ['percent_tiles_positive0'] + [clin_model] + ['comb'] + [outcome])
        writer.writerow(["Overall - n = " + str(len(dfprog.index))])
        recur_plot_cph(dfprog, writer, ['percent_tiles_positive0'] + [clin_model] + ['comb'] + [outcome])
        writer.writerow([""])
        
        f.close()

        dfprog = df1.copy()
        dfprog = dfprog[['chemo', 'recur', 'year_recur', 'dead', 'comb_thresh', 'percent_tiles_positive0', 'percent_tiles_positive0_thresh'] + [clin_model + '_thresh'] + [outcome]].dropna()
        fig, ax = plt.subplots(1,2, figsize=(10,5), dpi=300, sharey = 'row')

        prognosticPlotThreshold(dfprog, ax[0], clin_model + '_thresh')
        prognosticPlotThreshold(dfprog, ax[1], 'comb_thresh')
        ax[0].set_ylabel("Recurrence Free Interval")
        ax[0].set_title("Clinical Nomogram (High Sensitivity Threshold)")
        ax[0].set_xlim([-0.1, 10.1])
        ax[1].set_xlim([-0.1, 10.1])
        #ax[0].annotate("c", (-0.5, 1.005), annotation_clip = False, fontsize=12, weight='bold')
        ax[1].set_title("Combined Model (High Sensitivity Threshold)")
        #ax[1].annotate("d", (-0.5, 1.005), annotation_clip = False, fontsize=12, weight='bold')

        plt.tight_layout()
        plt.savefig("Prognostic Plots " + outcome + ".png")  
        plt.clf()
        fig, axs = plt.subplots(3, 1, figsize = (7.5, 12), dpi = 300)
        thresh['comb']['HR'] = prognosticPlotThreshold(dfprog,None, 'comb_thresh')
        thresh[clin_model]['HR'] = prognosticPlotThreshold(dfprog, None, clin_model + '_thresh')
        thresh['percent_tiles_positive0']['HR'] = prognosticPlotThreshold(dfprog, None, 'percent_tiles_positive0_thresh')

        thresh['comb']['HRendo'] = prognosticPlotThreshold(dfprog[dfprog.chemo == 0], axs[2], 'comb_thresh')
        thresh[clin_model]['HRendo'] = prognosticPlotThreshold(dfprog[dfprog.chemo == 0], axs[1], clin_model + '_thresh')
        thresh['percent_tiles_positive0']['HRendo'] = prognosticPlotThreshold(dfprog[dfprog.chemo == 0], axs[0], 'percent_tiles_positive0_thresh')
        axs[0].set_title("Pathologic Model")
        axs[1].set_title("Clinical Model")
        axs[2].set_title("Combined Model")
        plt.savefig("Prognostic Plots (all models) " + outcome + ".png")  
        plt.tight_layout()
        plt.clf()

        dfprog.loc[dfprog.dead == 1, 'recur'] = 1
        thresh['comb']['HR_RFS'] = prognosticPlotThreshold(dfprog, None, 'comb_thresh')
        thresh[clin_model]['HR_RFS'] = prognosticPlotThreshold(dfprog, None, clin_model + '_thresh')
        thresh['percent_tiles_positive0']['HR_RFS'] = prognosticPlotThreshold(dfprog, None, 'percent_tiles_positive0_thresh')

        thresh['comb']['HRendo_RFS'] = prognosticPlotThreshold(dfprog[dfprog.chemo == 0], None, 'comb_thresh')
        thresh[clin_model]['HRendo_RFS'] = prognosticPlotThreshold(dfprog[dfprog.chemo == 0], None, clin_model + '_thresh')
        thresh['percent_tiles_positive0']['HRendo_RFS'] = prognosticPlotThreshold(dfprog[dfprog.chemo == 0], None, 'percent_tiles_positive0_thresh')


    #plt.show()

    r2 = plotCorrelationODX(dfout, ['percent_tiles_positive0', clin_model, 'comb'],
                            ['Pathologic Model', clin_name, 'Combined Model'], outcome)
    if prognostic_plots != "none":
        plt.tight_layout()
        plt.savefig("Correlation " + outcome + ".png")
        plt.clf()
    for c in ['percent_tiles_positive0', clin_model, 'comb']:
        r2[c] = r2[c] + "," + r1[c]

    return r2, thresh


def recur_plot_cph(df, writer, vars):
    """Prints string to console containing hazard ratio / Cindex for cox model fit to each predictor

    Parameters
    ----------
    df - dataframe containing recurrence data and predictors
    vars - predictive variables to plot
    """
    if len(df[df.recur == 1].index) == 0:
        writer.writerow(["N/A - no events"])
        return
    aucs = plotAUROCrecur(df, vars, plot=False)
    for a, v in zip(aucs, vars):
        writer.writerow((prognosticPlot(df, v) + "," + str(round(a,3))).split(","))

def applyThresh(df, clin_name, thresh, cv = False):
    """Generates a dummy variable to identify patients where each model exceeds the specified threshold

    Parameters
    ----------
    df - dataframe containing predictors
    clin_name - name of the clinical predictive model
    thresh - dictionary with thresholds to apply for each model
    cv - use to apply the threshold as appropriate for each crossfold
    """

    df['percent_tiles_positive0_thresh'] = 0
    df[clin_name + '_thresh'] = 0
    df['comb_thresh'] = 0
    if cv:
        for i in [1,2,3]:
            df.loc[((df['percent_tiles_positive0'] > thresh['percent_tiles_positive0']['ThreshList'][i-1]) & (df.cv_linear == i)), 'percent_tiles_positive0_thresh'] = 1
            df.loc[((df[clin_name] > thresh[clin_name]['ThreshList'][i-1]) & (df.cv_linear == i)), clin_name + '_thresh'] = 1
            df.loc[((df['comb'] > thresh['comb']['ThreshList'][i-1]) & (df.cv_linear == i)), 'comb_thresh'] = 1
    else:        
        df.loc[df['percent_tiles_positive0'] > thresh['percent_tiles_positive0']['Thresh'], 'percent_tiles_positive0_thresh'] = 1
        df.loc[df[clin_name] > thresh[clin_name]['Thresh'], clin_name + '_thresh'] = 1
        df.loc[df['comb'] > thresh['comb']['Thresh'], 'comb_thresh'] = 1
        
    return df

def testMultivariate(df, show = False, outcome = 'RS', NCDB = False, prognostic_plots = 'none'):
    """Tests a multivariate model in the University of Chicago Cohort

    Parameters
    ----------
    df - dataframe containing predictors and outcome of interest
    show - if True will show plots of predictive accuracy
    outcome - outcome to predict
    NCDB - if True, will generate clinical model from NCDB; else will use Tennessee Score
    prognostic_plots - 'roc' - plot training / validation ROC curves. 'all' - also plot survival curves using threshold as cutpoint. 'none' - do not save any plots
    """

    pred_cols = ['percent_tiles_positive0', 'ten_score']
    outcome_cat = 'odx85_cat'
    plt.figure(1)
    if prognostic_plots == 'all':
        fig, axs = plt.subplots(1,2, figsize=(10,5) , dpi=300, sharey = 'row', num = "roc")
    else:
        fig, axs = plt.subplots(1,2, figsize=(10,5), dpi=300, sharey='row', num = "roc")
    if outcome == 'mpscore':
        outcome_cat = 'mp_cat'
    if NCDB:
        pred_cols = ['percent_tiles_positive0', 'pred_score']
        if prognostic_plots == 'all':
            clf, clf_ncdb, thresh, r0 = fitMultivariate(ax = axs[0], outcome = outcome_cat, NCDB = NCDB, prognostic_plots = prognostic_plots)
        else:
            clf, clf_ncdb, thresh, r0 = fitMultivariate(ax = axs[0], outcome=outcome_cat, NCDB=NCDB, prognostic_plots = prognostic_plots)
    else:
        if prognostic_plots == 'all':
            clf, thresh, r0 = fitMultivariate(ax = axs[0], outcome=outcome_cat, NCDB=NCDB, prognostic_plots = prognostic_plots)
        else:
            clf, thresh, r0 = fitMultivariate(ax = axs[0], outcome=outcome_cat, NCDB=NCDB, prognostic_plots = prognostic_plots)
    if SUFFIX != "_REV":
        calcClinModelUCMC(df)
    else:
        calcClinModelTCGA(df)
    df = df.dropna(subset = ['percent_tiles_positive0']).copy()
    if NCDB:
        if outcome == 'RS':
            df['pred_score'] = clf_ncdb.predict_proba(df[ODX_vars])[:, 1]
        elif outcome == 'mpscore':
            df['pred_score'] = clf_ncdb.predict_proba(df[MP_vars])[:, 1]
    df['comb'] = predictLogisticRegression(clf, df[pred_cols])
    df = applyThresh(df, pred_cols[1], thresh)
    if prognostic_plots == 'save':
        return df
    else:
        r1, thresh2 = eval_model(df, axs, pred_cols[1], outcome, prognostic_plots)
        for ci in ['percent_tiles_positive0', pred_cols[1], 'comb']:
            r0[ci] = r0[ci] + "," + r1[ci]
        if prognostic_plots == 'all' and SUFFIX != '_REV':
            global CUR_OUT_FILE
            f = open(CUR_OUT_FILE, 'a')
            writer = csv.writer(f)
            writer.writerow(["**Performance characteristics at 95% sensitivity threshold**"])
            writer.writerow("Sen (TCGA), Spec (TCGA), PPV (TCGA), NPV (TCGA), Sen (UCMC), Spec (UCMC), PPV (UCMC), NPV (UCMC), HR, HRendo, HR_RFS, HR_RFSendo".split(","))
            for c in ['percent_tiles_positive0', pred_cols[1],  'comb']:
                writer.writerow((str(round(thresh[c]['Sen']*100,1)) + "," + str(round(thresh[c]['Spec']*100,1))+ "," + str(round(thresh[c]['PPV']*100,1)) + "," + str(round(thresh[c]['NPV']*100,1))+ "," + str(round(thresh2[c]['Sen']*100,1)) + "," + str(round(thresh2[c]['Spec']*100,1))+ "," + str(round(thresh2[c]['PPV']*100,1)) + "," + str(round(thresh2[c]['NPV']*100,1))+ "," +thresh2[c]['HR'] + "," +thresh2[c]['HRendo'] + "," +thresh2[c]['HR_RFS'] + "," +thresh2[c]['HRendo_RFS']).split(","))
            writer.writerow([""])
            f.close()
        return r0

def testODX(ten_score = True, NCDB = False, race_subset = None, prognostic_plots = 'none'):
    """Tests prediction of OncotypeDx in the University of Chicago Cohort

    Parameters
    ----------
    ten_score - if True, show predictions for combined model with Tennesse Model
    NCDB - if True, show predictions for combined model with NCDB clinical model
    race_subset - if None, will display results for whole dataset. If 'White' or 'Black', will show results for that subset of patients
    prognostic_plots - 'roc' - plot training / validation ROC curves. 'all' - also plot survival curves using threshold as cutpoint. 'none' - do not save any plots
    """
    global odx_train
    df1 = pd.read_csv(join(PROJECT_ROOT, "UCH_RS", "uch_brca_complete.csv"))
    df1.dropna(subset=['percent_tiles_positive0_odx'])
    if SUFFIX == "_REV":
        df1 = pd.read_csv(join(PROJECT_ROOT, "UCH_RS", "tcga_brca_complete.csv"))
        df1 = df1[(df1.ER_Status_By_IHC == 'positive') | (df1.PR == 1)] 
        df1 = df1[df1.HER2Calc == 'negative']
    if not RUN_FROM_OLD_STATS:
        SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS"))
        if SUFFIX == "_REV":
            df2 = pd.read_csv(join(find_eval(SFP, "ODX_" + exp_suffix + SUFFIX, "RS", 1), "patient_predictions_odx85_eval.csv"))
        else:
            df2 = pd.read_csv(join(find_eval(SFP, "ODX_" + exp_suffix + SUFFIX, odx_train, 1), "patient_predictions_RSHigh_eval.csv"))
        df2 = df2[['patient', 'percent_tiles_positive0']]
        df1['patient'] = df1['patient'].astype(str)
        df2['patient'] = df2['patient'].astype(str)
        df1 = df1.merge(df2, on='patient', how='left')
    else:
        df1['percent_tiles_positive0'] = df1['percent_tiles_positive0_odx']
    if race_subset == 'White':
        df1 = df1[df1.race_1 == '01 White']
        prognostic_plots = 'none'
    if race_subset == 'Black':
        df1 = df1[df1.race_1 == '02 Black']
        prognostic_plots = 'none'
        
    if SUFFIX != "_REV":
        df1 = df1[df1.hist_type != 'DCIS'].copy()
        df1 = df1[df1.RSHigh.isin(['H','L'])]
    
    df1 = df1.drop_duplicates(subset='patient')

    if ten_score:
        r0 = testMultivariate(df1, show = True, outcome = 'RS', NCDB = False, prognostic_plots = prognostic_plots)
    if NCDB:
        r1 = testMultivariate(df1, show=True, outcome='RS', NCDB=True, prognostic_plots=prognostic_plots)

    global CUR_OUT_FILE
    f = open(CUR_OUT_FILE, 'a')
    writer = csv.writer(f)
    writer.writerow(["**Performance for Prediction in TCGA / Validation**"])
    writer.writerow("Model,Pearson r (TCGA),Spearman ρ (TCGA),AUROC (TCGA), AUPRC (TCGA), AUPRC Null (TCGA), Pearson r (UCMC),Spearman ρ (UCMC), AUROC (UCMC), AUPRC (UCMC), AUPRC Null (UCMC), z,p".split(","))
    if ten_score:
        for k, v in r0.items():
            writer.writerow((k + "," + v).split(","))
    if NCDB:
        for k,v in r1.items():
            writer.writerow((k + "," + v).split(","))
    writer.writerow([""])
    f.close()

def predictExternal(outcome = 'RS', dataset = "uch_brca_complete.csv", exp_label = None):
    """Tests prediction of OncotypeDx in the University of Chicago Cohort

    Parameters
    ----------
    ten_score - if True, show predictions for combined model with Tennesse Model
    NCDB - if True, show predictions for combined model with NCDB clinical model
    race_subset - if None, will display results for whole dataset. If 'White' or 'Black', will show results for that subset of patients
    prognostic_plots - 'roc' - plot training / validation ROC curves. 'all' - also plot survival curves using threshold as cutpoint. 'none' - do not save any plots
    """
    global odx_train, mp_train, exp_suffix
    df1 = pd.read_csv(join(PROJECT_ROOT, "UCH_RS", dataset))
    SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS"))
    if outcome == 'RS':
        df2 = pd.read_csv(join(find_eval(SFP, "ODX_" + exp_suffix, odx_train, 1), "patient_predictions_RSHigh_eval.csv"))
    else:
        df2 = pd.read_csv(join(find_eval(SFP, "MP_" + exp_suffix, mp_train, 1), "patient_predictions_MPHigh_eval.csv"))
    df2 = df2[['patient', 'percent_tiles_positive0']]
    df1['patient'] = df1['patient'].astype(str)
    df2['patient'] = df2['patient'].astype(str)
    df1 = df1.merge(df2, on='patient', how='left')
    df1 = df1.drop_duplicates(subset='patient')
    if outcome == 'MP':
        df1['percent_tiles_positive0'] = -1 * df1['percent_tiles_positive0'] 
    df = None
    if outcome == 'MP':
        df = testMultivariate(df1, show=True, outcome='mpscore', NCDB=True, prognostic_plots='save')
        df = df[['patient', 'percent_tiles_positive0', 'pred_score', 'comb', 'percent_tiles_positive0_thresh', 'pred_score_thresh', 'comb_thresh']]
    else:
        df = testMultivariate(df1, show = True, outcome = 'RS', NCDB = False, prognostic_plots = 'save')
        df = df[['patient', 'percent_tiles_positive0', 'ten_score', 'comb', 'percent_tiles_positive0_thresh', 'ten_score_thresh', 'comb_thresh']]
    df.to_csv(dataset.replace(".csv",exp_label + "_predictions.csv"), index = False)

def testMP(ten_score = False, NCDB = True, prognostic_plots = 'none'):
    """Tests prediction of MammaPrint in the University of Chicago Cohort

    Parameters
    ----------
    ten_score - if True, show predictions for combined model with Tennesse Model
    NCDB - if True, show predictions for combined model with NCDB clinical model
    prognostic_plots - 'roc' - plot training / validation ROC curves. 'all' - also plot survival curves using threshold as cutpoint. 'none' - do not save any plots
    """
    global mp_train, exp_suffix
    df1 = pd.read_csv(join(PROJECT_ROOT, "UCH_RS", "uch_brca_complete.csv"))
    df1.dropna(subset=['percent_tiles_positive0_mp'])
    if not RUN_FROM_OLD_STATS:
        SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS"))
        df2 = pd.read_csv(join(find_eval(SFP, "MP_" + exp_suffix + SUFFIX, mp_train, 1), "patient_predictions_MPHigh_eval.csv"))
        df2 = df2[['patient', 'percent_tiles_positive0']]
        df1['patient'] = df1['patient'].astype(str)
        df2['patient'] = df2['patient'].astype(str)
        df1 = df1.merge(df2, on='patient', how='left')
    else:
        df1['percent_tiles_positive0'] = df1['percent_tiles_positive0_mp']
    df1['percent_tiles_positive0'] = -1 * df1['percent_tiles_positive0_mp']
    df1 = df1.drop_duplicates(subset='patient')
    if ten_score:
        r0 = testMultivariate(df1, show = True, outcome = 'mpscore', NCDB = False, prognostic_plots = prognostic_plots)
    if NCDB:
        r1 = testMultivariate(df1, show=True, outcome='mpscore', NCDB=True, prognostic_plots=prognostic_plots)
    global CUR_OUT_FILE
    f = open(CUR_OUT_FILE, 'a')
    writer = csv.writer(f)
    writer.writerow(["**Performance for Prediction in TCGA / Validation**"])
    writer.writerow("Model,Pearson r (TCGA),Spearman ρ (TCGA),AUROC (TCGA), AUPRC (TCGA), AUPRC Null (TCGA), Pearson r (UCMC),Spearman ρ (UCMC), AUROC (UCMC), AUPRC (UCMC), AUPRC Null (UCMC), z,p".split(","))
    if ten_score:
        for k, v in r0.items():
            writer.writerow((k + "," + v).split(","))
    if NCDB:
        for k,v in r1.items():
            writer.writerow((k + "," + v).split(","))
    writer.writerow([""])
    f.close()

def organizeDataUCMC(df):
    """Categorizes the University of Chicago Dataset

    Parameters
    ----------
    df - dataframe containing the dataset

    Returns
    ----------
    The organized dataframe

    """
    df['sex'] = df['sex'].map({'1 Male': 'Male', '2 Female': 'Female'})
    df['race_1'] = df['race_1'].map({'01 White': 'White', '02 Black': 'Black',
                                     '04 Chinese': 'Asian', '05 Japanese': 'Asian', '06 Filipino': 'Asian',
                                     '08 Korean': 'Asian', '10 Vietnamese': 'Asian',
                                     '15 Asian Indian or Pakistani, NOS': 'Asian',
                                     '16 Asian Indian': 'Asian',
                                     '96 Other Asian, including Asian/Oriental, NOS': 'Asian',
                                     '03 American Indian, Aleutian, or Eskimo': 'Other', '98 Other': 'Other',
                                     '99 Unknown': np.nan})
    df['Spanish_Hispanic_Origin'] = df['Spanish_Hispanic_Origin'].map({'0 Non-Spanish; non-Hispanic': 'Non-Hispanic',
                                                                       '1 Mexican (includes Chicano)': 'Hispanic',
                                                                       '2 Puerto Rican': 'Hispanic',
                                                                       '4 South or Central American (except Brazil)': 'Hispanic',
                                                                       '6 Spanish, NOS; Hispanic, NOS; Latino, NOS': 'Hispanic',
                                                                       '9 Unknown whether Spanish or not; not stated in pt record': np.nan})
    df['hist_type'] = df['hist_type'].map(
        {'Ductal': 'Ductal', 'Lobular': 'Lobular', 'D&L': 'Ductal and Lobular', 'Metaplasia': 'Other',
         'Mucinous': 'Other', 'Others': 'Other', 'Paget disease': 'Other', 'Papillary': 'Other', 'Tabular': 'Other'})
    df['nodes'] = 'Negative'
    df.loc[df['regionalnodespositive'] > 0, 'nodes'] = 'Positive'
    df['ER'] = df['ER'].map({'Pos': 'Positive', 'Neg': 'Negative'})
    df['PR'] = df['PR'].map({'Pos': 'Positive', 'Neg': 'Negative', '.5':'Positive'})
    df['HER2'] = df['HER2'].map({'Pos': 'Positive', 'Neg': 'Negative'})
    df['dup_chemo'] = df['dup_chemo'].map({1.0: 'Yes', 2.0: 'Yes', np.nan: 'No'})
    df['recur'] = df['recur'].map({1.0: 'Recurred', 0.0: 'Disease Free'})
    df['dead'] = df['dead'].map({1.0: 'Dead', 0.0: 'Alive'})
    return df

def prognosticComparison():
    """Compares prognosis between the TCGA and UCMC datasets.
    Prognostic plot is saved under prognostic_comparison.png in the root directory

    """
    df = pd.read_csv(join(PROJECT_ROOT, "UCH_RS", "uch_brca_complete.csv"))
    df = df.dropna(subset=['RS', 'recur', 'year_recur'])
    df = df.drop_duplicates(subset ='patient')
    df['DS'] = 0
    df2 = pd.read_csv(join(PROJECT_ROOT, "UCH_RS", "tcga_brca_complete.csv"))
    df2 = df2[(df2.ER_Status_By_IHC == 'positive') | (df2.PR == 1)]
    df2 = df2[df2.HER2Calc == 'negative']
    df2 = df2.drop_duplicates(subset ='patient')
    df2['recur'] = df2['Disease_Free_Status'].map({'censored':0,'progressed':1})
    df2['year_recur'] = df2['Disease_Free_Months'] / 12    
    df2['DS'] = 1
    cph = CoxPHFitter(penalizer = 0.1)
    dfcph = pd.concat([df[['recur', 'year_recur', 'DS']], df2[['recur', 'year_recur', 'DS']]]).dropna()
    cph.fit(dfcph, duration_col='year_recur', event_col='recur')
    fig, ax = plt.subplots(dpi = 300)
    if ax:
        kmf1 = KaplanMeierFitter()
        kmf2 = KaplanMeierFitter()
        kmf1.fit(df['year_recur'], df['recur'], label='UCMC').plot_survival_function(ax=ax, ci_show=False)
        kmf2.fit(df2['year_recur'], df2['recur'], label='TCGA').plot_survival_function(ax=ax, ci_show=False)
        add_at_risk_counts(kmf1, kmf2, ax=ax)
        ax.set_xlabel("Years")
        ax.legend(loc = 'lower left')
        ax.annotate("HR " + str(round(cph.hazard_ratios_[0], 2)) + " (" + str(round(exp(cph.confidence_intervals_["95% lower-bound"][0]), 2)) + " - " + str(round(exp(cph.confidence_intervals_["95% upper-bound"][0]), 2)) + "), p < 0.001", (9, 0.82), annotation_clip=False, fontsize=12)
    ax.set_ylabel("Recurrence-Free Interval")
    plt.tight_layout()
    plt.savefig("Prognostic Comparison.png")
    plt.clf()
    
    
def describeCohortODX():
    """Generates a baseline demographics table for University of Chicago patients with OncotypeDx testing
    """

    df = pd.read_csv(join(PROJECT_ROOT, "UCH_RS", "uch_brca_complete.csv"))
    df = df.dropna(subset=['RS'])
    df = df.drop_duplicates(subset ='patient')

    columns = ['Age', 'sex', 'race_1', 'Spanish_Hispanic_Origin', 'hist_type', 'grade', 'tumor_size', 'nodes', 'ER', 'PR', 'HER2', 'RS', 'dup_chemo', 'year_FU', 'recur', 'dead']
    categorical = ['sex', 'race_1', 'Spanish_Hispanic_Origin', 'hist_type', 'grade', 'nodes', 'ER', 'PR', 'HER2', 'dup_chemo', 'recur', 'dead']
    labels = ['Age', 'Sex', 'Race', 'Ethnicity', 'Histologic Subtype', 'Grade', 'Tumor Size', 'Nodal Status', 'ER Status', 'PR Status', 'HER2 Status', 'OncotypeDx Score', 'Chemotherapy', 'Follow-up', 'Recurrence', 'Vital Status']
    df = organizeDataUCMC(df)
    l = dict(zip(columns, labels))
    mytable = TableOne(df, columns=columns, categorical=categorical, labels = l)
    #print(mytable.tabulate(tablefmt='github'))
    mytable.to_excel('UCMC Oncotype Cohort.xlsx')

def describeCohortMP():
    """Generates a baseline demographics table for University of Chicago patients with MammaPrint testing
    """

    df = pd.read_csv(join(PROJECT_ROOT, "UCH_RS", "uch_brca_complete.csv"))
    df = df.dropna(subset=['mpscore'])
    df = df.drop_duplicates(subset ='patient')
    columns = ['Age', 'sex', 'race_1', 'Spanish_Hispanic_Origin', 'hist_type', 'grade', 'tumor_size', 'nodes', 'ER', 'PR', 'HER2', 'mpscore', 'dup_chemo', 'year_FU', 'recur', 'dead']
    categorical = ['sex', 'race_1', 'Spanish_Hispanic_Origin', 'hist_type', 'grade', 'nodes', 'ER', 'PR', 'HER2', 'dup_chemo', 'recur', 'dead']
    labels = ['Age', 'Sex', 'Race', 'Ethnicity', 'Histologic Subtype', 'Grade', 'Tumor Size', 'Nodal Status', 'ER Status', 'PR Status', 'HER2 Status', 'MammaPrint Score', 'Chemotherapy', 'Follow-up', 'Recurrence', 'Vital Status']
    df = organizeDataUCMC(df)
    l = dict(zip(columns, labels))
    mytable = TableOne(df, columns=columns, categorical=categorical, labels = l)
    #print(mytable.tabulate(tablefmt='github'))
    mytable.to_excel('UCMC MammaPrint Cohort.xlsx')

def describeCohortTCGA():
    """Generates a baseline demographics table for TCGA patients
    """
    global odx_train, mp_train
    columns = ['age_at_diagnosis', 'gender', 'RACE_repo', 'ethnicity', 'hist_type', 'Grade', 'tumor_size', 'nodes', 'ER_Status_By_IHC', 'PR', 'HER2Calc', odx_train, mp_train, 'dup_chemo', 'Overall_Survival_Months', 'Disease_Free_Status', 'Overall_Survival_Status']
    categorical = ['gender', 'RACE_repo', 'ethnicity', 'hist_type', 'Grade', 'nodes', 'ER_Status_By_IHC', 'PR', 'HER2Calc', 'dup_chemo', 'Disease_Free_Status', 'Overall_Survival_Status']
    labels = ['Age', 'Sex', 'Race', 'Ethnicity', 'Histologic Subtype', 'Grade', 'Tumor Size', 'Nodal Status', 'ER Status', 'PR Status', 'HER2 Status', 'Research-Only OncotypeDx Score', 'Research-Only MammaPrint Score', 'Chemotherapy', 'Follow-up', 'Recurrence', 'Vital Status']
    df = pd.read_csv(join(PROJECT_ROOT, "UCH_RS", "tcga_brca_complete.csv"))
    df['hist_type'] = 'Other'
    df = df.drop_duplicates(subset='patient')
    
    df.loc[df['2016 Histology Annotations'] == 'Invasive ductal carcinoma', 'hist_type'] = 'Ductal'
    df.loc[df['2016 Histology Annotations'] == 'Invasive lobular carcinoma', 'hist_type'] = 'Lobular'
    df.loc[df['2016 Histology Annotations'] == 'Mixed', 'hist_type'] = 'Ductal and Lobular'

    df['gender'] = df['gender'].map({'MALE': 'Male', 'FEMALE': 'Female'})

    df['RACE_repo'] = df['RACE_repo'].map({'White': 'White', 'Black': 'Black',
                                     'Asian': 'Asian', 'AI/AN': 'Other'})
    df['ethnicity'] = df['ethnicity'].map({'NOT HISPANIC OR LATINO': 'Non-Hispanic',
                                                                       'HISPANIC OR LATINO': 'Hispanic'})
    df['nodes'] = 'Positive'
    df.loc[df['AJCC_N'].isin(['n0','n0 (i-)','n0 (i+)','n0 (mol+)']), 'nodes'] = 'Negative'
    df.loc[df['AJCC_N'].isin(['nx']), 'nodes'] = np.nan
    df['ER_Status_By_IHC'] = df['ER_Status_By_IHC'].map({'positive': 'Positive', 'negative': 'Negative'})
    df['PR'] = df['PR'].map({1: 'Positive', 0: 'Negative'})
    df['HER2Calc'] = df['HER2Calc'].map({'positive': 'Positive', 'negative': 'Negative'})
    df['dup_chemo'] = np.nan
    df.loc[(df['Neoadjuvant_Therapy'] =='no') | (df['Adjuvant_Therapy_Administered'] =='no'), 'dup_chemo'] = 'No'
    df.loc[(df['Neoadjuvant_Therapy'] =='yes') | (df['Adjuvant_Therapy_Administered'] =='yes'), 'dup_chemo'] = 'Yes'
    df['Overall_Survival_Months'] = pd.to_numeric(df['Overall_Survival_Months'], errors='coerce')/12
    df['age_at_diagnosis'] = pd.to_numeric(df['age_at_diagnosis'], errors='coerce')
    df['tumor_size'] = pd.to_numeric(df['tumor_size'], errors='coerce')
    df[odx_train] = pd.to_numeric(df[odx_train], errors='coerce')
    df[mp_train] = pd.to_numeric(df[mp_train], errors='coerce')

    df['Disease_Free_Status'] = df['Disease_Free_Status'].map({'progressed': 'Recurred', 'censored': 'Disease Free'})
    df['Overall_Survival_Status'] = df['Overall_Survival_Status'].map({'dead': 'Dead', 'alive':'Alive'})
    l = dict(zip(columns, labels))
    mytable = TableOne(df, columns=columns, categorical=categorical, labels = l)
    #print(mytable.tabulate(tablefmt='github'))
    mytable.to_excel('TCGA Cohort.xlsx')
    
    df = df[(df.ER_Status_By_IHC == 'Positive') | (df.PR == 'Positive')] 
    df = df[df.HER2Calc == 'Negative']
    mytable = TableOne(df, columns=columns, categorical=categorical, labels = l)
    #print(mytable.tabulate(tablefmt='github'))
    mytable.to_excel('TCGA HRHER2 Cohort.xlsx')
    
def correlateNecrosisLVI():
    """Prints correlation between various pathologic parameters and model predictions in the TCGA dataset
    """

    pred_cols = ['percent_tiles_positive0']
    df1 = pd.read_csv(join(PROJECT_ROOT, "UCH_RS", "tcga_brca_complete.csv"))
    dfs = []
    for i in [1,2,3]:
        dfa = pd.read_csv(join(PROJECT_ROOT, "saved_results", "odx85_cat" + str(i)+ ".csv"))
        dfa['cv_linear'] = i
        dfs += [dfa]
    df1 = df1.merge(pd.concat(dfs), left_on="patient", right_on="patient", how="left")
    
    global CUR_OUT_FILE
    f = open(CUR_OUT_FILE, 'a')
    writer = csv.writer(f)
    
    writer.writerow(["OVERALL COHORT"])
    writer.writerow("Characteristic, n, T-statistic, p-value".split(","))
    def printTstat(name, n, t):
        writer.writerow((name + "," + str(n) +"," + str(round(t[0], 3)) + "," + str(t[1])).split(","))
    printTstat("Necrosis", len(df1[df1['Necrosis'].isin(['Present','Absent'])].index), ttest_ind(df1.loc[df1.Necrosis == 'Present', 'percent_tiles_positive0'], df1.loc[df1.Necrosis == 'Absent', 'percent_tiles_positive0']))
    printTstat("LVI", len(df1[df1['Lymphovascular Invasion (LVI)'].isin(['Present','Absent'])].index), ttest_ind(df1.loc[df1['Lymphovascular Invasion (LVI)'] == 'Present', 'percent_tiles_positive0'], df1.loc[df1['Lymphovascular Invasion (LVI)'] == 'Absent', 'percent_tiles_positive0']))
    printTstat("Epithelial Grade", len(df1[df1['Epithelial'].isin([3,2,1])].index), ttest_ind(df1.loc[df1['Epithelial'] < 3, 'percent_tiles_positive0'], df1.loc[df1['Epithelial'] == 3, 'percent_tiles_positive0']))
    printTstat("Pleomorph Grade", len(df1[df1['Pleomorph'].isin([3,2,1])].index), ttest_ind(df1.loc[df1['Pleomorph'] < 3, 'percent_tiles_positive0'], df1.loc[df1['Pleomorph'] == 3, 'percent_tiles_positive0']))
    printTstat("Mitosis Grade", len(df1[df1['Mitosis'].isin(['(score = 3) >10 per 10 HPF', '(score = 1) 0 to 5 per 10 HPF', '(score = 2) 6 to 10 per 10 HPF'])].index), ttest_ind(df1.loc[df1['Mitosis'].isin(['(score = 1) 0 to 5 per 10 HPF', '(score = 2) 6 to 10 per 10 HPF']), 'percent_tiles_positive0'], df1.loc[df1['Mitosis'] == '(score = 3) >10 per 10 HPF', 'percent_tiles_positive0']))
    printTstat("Overall Grade", len(df1[df1['Grade'].isin([3,2,1])].index), ttest_ind(df1.loc[df1['Grade'] < 3, 'percent_tiles_positive0'], df1.loc[df1['Grade'] == 3, 'percent_tiles_positive0']))
    writer.writerow([""])
    df1 = df1[(df1.ER_Status_By_IHC == 'positive') | (df1.PR == 1)] 
    df1 = df1[df1.HER2Calc == 'negative']
    writer.writerow(["HR+/HER2-"])
    writer.writerow("Characteristic, n, T-statistic, p-value".split(","))

    printTstat("Necrosis", len(df1[df1['Necrosis'].isin(['Present','Absent'])].index), ttest_ind(df1.loc[df1.Necrosis == 'Present', 'percent_tiles_positive0'], df1.loc[df1.Necrosis == 'Absent', 'percent_tiles_positive0']))
    printTstat("LVI", len(df1[df1['Lymphovascular Invasion (LVI)'].isin(['Present','Absent'])].index), ttest_ind(df1.loc[df1['Lymphovascular Invasion (LVI)'] == 'Present', 'percent_tiles_positive0'], df1.loc[df1['Lymphovascular Invasion (LVI)'] == 'Absent', 'percent_tiles_positive0']))
    printTstat("Epithelial Grade", len(df1[df1['Epithelial'].isin([3,2,1])].index), ttest_ind(df1.loc[df1['Epithelial'] < 3, 'percent_tiles_positive0'], df1.loc[df1['Epithelial'] == 3, 'percent_tiles_positive0']))
    printTstat("Pleomorph Grade", len(df1[df1['Pleomorph'].isin([3,2,1])].index), ttest_ind(df1.loc[df1['Pleomorph'] < 3, 'percent_tiles_positive0'], df1.loc[df1['Pleomorph'] == 3, 'percent_tiles_positive0']))
    printTstat("Mitosis Grade", len(df1[df1['Mitosis'].isin(['(score = 3) >10 per 10 HPF', '(score = 1) 0 to 5 per 10 HPF', '(score = 2) 6 to 10 per 10 HPF'])].index), ttest_ind(df1.loc[df1['Mitosis'].isin(['(score = 1) 0 to 5 per 10 HPF', '(score = 2) 6 to 10 per 10 HPF']), 'percent_tiles_positive0'], df1.loc[df1['Mitosis'] == '(score = 3) >10 per 10 HPF', 'percent_tiles_positive0']))
    printTstat("Overall Grade", len(df1[df1['Grade'].isin([3,2,1])].index), ttest_ind(df1.loc[df1['Grade'] < 3, 'percent_tiles_positive0'], df1.loc[df1['Grade'] == 3, 'percent_tiles_positive0']))
    f.close()
    
def main():

    parser = argparse.ArgumentParser(description = "Helper to guide through model training.")
    parser.add_argument('-s', '--saved', action="store_true", help='If provided, will use saved stats for parameter calculation.')
    parser.add_argument('-p', '--project_root', required=False, type=str, help='Path to project directory (if not provided, assumes subdirectory of this script).')
    parser.add_argument('-uf', '--use_filtered', action="store_true", help='If provided, will load predictions generated from filtered TCGA-BRCA slides (instead of annotated slides)')
    parser.add_argument('-tr', '--train_receptors', action="store_true", help='If provided, will load predictions generated when training on only HR+/HER2- TCGA-BRCA slides (instead of entire dataset)')
    parser.add_argument('-rev', '--train_reverse', action="store_true", help='If provided, will run predictions trained on UCMC and validated on TCGA')
    parser.add_argument('-pred', '--predict', action="store_true", help='If provided, will just generate predictions on the specified external dataset')
    parser.add_argument('-o', '--outcome', required=False, type=str, help='whether to predict for Oncotype or MammaPrint. If set to MP will predict MammaPrint, otherwise Oncotype')
    parser.add_argument('-d', '--dataset', required=False, type=str, help='name of csv file in the UCH_RS folder for annotations of external dataset for predictions. Defaults to uch_brca_complete.csv')
    parser.add_argument('-exp', '--experiment_label', required=False, type=str, help='name of experiment label used when making pathologic model predictions. Defaults to Final_BRCAROI.')
    
    
    args = parser.parse_args()
    global RUN_FROM_OLD_STATS
    global PROJECT_ROOT
    global SUFFIX
    global exp_suffix
    if args.use_filtered:
        SUFFIX = "_UF"
    if args.train_receptors:
        SUFFIX = "_TR"
    if args.train_reverse:
        SUFFIX = "_REV"
    if args.project_root:
        PROJECT_ROOT = args.project_root
    RUN_FROM_OLD_STATS = args.saved
    if args.experiment_label:
        exp_suffix = args.experiment_label
    if args.predict:
        exp_label = None
        dataset = "uch_brca_complete.csv"
        if args.dataset:
            dataset = args.dataset
        outcome = "RS"
        if args.outcome == "MP":
            outcome = "MP"
        predictExternal(outcome = "RS", dataset = dataset, exp_label = exp_label)
    else:
        print("SAVING COHORT DESCRIPTIONS")
        describeCohortODX()
        describeCohortMP()
        describeCohortTCGA() 
        plot_hp_search('hpopt.csv')
        prognosticComparison()
        global CUR_OUT_FILE
        CUR_OUT_FILE = "predictions_odx.csv"
        f = open(CUR_OUT_FILE, 'w')
        writer = csv.writer(f)
        writer.writerow(["---------PREDICTIONS FOR ONCOTYPE MODEL---------"])
        f.close()
        print("GENERATING PREDICTIONS FOR ONCOTYPE MODEL")
        testODX(ten_score = True, NCDB = False, prognostic_plots = 'all')
        
        CUR_OUT_FILE = "predictions_mp.csv"
        f = open(CUR_OUT_FILE, 'w')
        writer = csv.writer(f)
        print("GENERATING PREDICTIONS FOR MAMMAPRINT MODEL")
        writer.writerow(["---------PREDICTIONS FOR MAMMAPRINT MODEL---------"])
        f.close()        
        testMP(ten_score = False, NCDB = True, prognostic_plots = 'roc')

        CUR_OUT_FILE = "predictions_odx.csv"
        f = open(CUR_OUT_FILE, 'a')
        writer = csv.writer(f)        
        print("GENERATING SECONDARY ANALYSES")
        writer.writerow(["---------PREDICTIONS FOR ONCOTYPE MODEL (WHITE SUBSET)---------"])
        f.close()        
        testODX(ten_score = True, NCDB = False, race_subset = 'White')

        f = open(CUR_OUT_FILE, 'a')
        writer = csv.writer(f)        
        writer.writerow(["---------PREDICTIONS FOR ONCOTYPE MODEL (BLACK SUBSET)---------"])
        f.close()
        testODX(ten_score = True, NCDB = False, race_subset = 'Black')

        CUR_OUT_FILE = "correlate_pathparams.csv"
        f = open(CUR_OUT_FILE, 'a')
        writer = csv.writer(f)        
        writer.writerow(["---------CORRELATION BETWEEN PREDICTIONS AND PATH FEATURES---------"])
        f.close()
        correlateNecrosisLVI()

if __name__ == '__main__':
    main()