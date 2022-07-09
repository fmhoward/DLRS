import os
#os.environ['SF_BACKEND'] = 'tensorflow'
from slideflow.model import ModelParams
from slideflow.project_utils import get_validation_settings
import matplotlib.pyplot as plt
import ConfigSpace as CS
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario
import pandas as pd
from os.path import join
from slideflow.util import log
import slideflow as sf
import json
import pandas as pd
import shutil
from os.path import basename, dirname, join
import argparse

PROJECT_ROOT = os.getcwd() + "/PROJECTS/"

hp_roi = ModelParams(tile_px= 299,
                    tile_um = 302,
                    normalizer = 'reinhard_fast',
                    normalizer_source = None,
                    l2 = 1e-05,
                    augment=True,
                    training_balance = "category",
                    validation_balance = "none",
                    batch_size = 128,
                    drop_images = False,
                    dropout = 0.5,
                    early_stop = True,
                    early_stop_method = "loss",
                    early_stop_patience = 15,
                    epochs = 1,
                    hidden_layer_width = 256,
                    hidden_layers = 3,
                    learning_rate = 0.0001,
                    learning_rate_decay = 0.97,
                    learning_rate_decay_steps = 100000,
                    loss = "sparse_categorical_crossentropy",
                    model = "xception",
                    optimizer = "Adam",
                    pooling = "avg",
                    toplayer_epochs = 0,
                    trainable_layers = 0)
                    
hp_opt = ModelParams(tile_px= 299,
                    tile_um = 302,
                    normalizer = 'reinhard_fast',
                    normalizer_source = None,
                    augment="xyr",
                    training_balance = "patient",
                    validation_balance = "none",
                    batch_size = 55,
                    drop_images = False,
                    dropout = 0.31155932,
                    early_stop = True,
                    early_stop_method = "loss",
                    early_stop_patience = 0,
                    epochs = 1,
                    hidden_layer_width = 267,
                    hidden_layers = 1,
                    learning_rate = 0.000999282,
                    l1 = 0,
                    l1_dense = 0.008306567,
                    l2 = 0.034253246,
                    l2_dense = 0.039803477,
                    learning_rate_decay = 0.210307662,
                    learning_rate_decay_steps = 1023,
                    loss = "mean_squared_error",
                    model = "xception",
                    optimizer = "Adam",
                    pooling = "avg",
                    toplayer_epochs = 0,
                    trainable_layers = 0,
                    onehot_pool = 'false')
                    
                                        


def get_model_results(path, metric, variable):
    """Reads results/metrics from a trained model.

    Parameters
    ----------
    path - path to the trained model
    metric - metric to return (i.e. 'tile_auc' or similar)
    variable - variable for which outcome of interest should be returned
    """    
    try:
        csv = pd.read_csv(join(path, 'results_log.csv'))
        model_res = next(csv.iterrows())[1]
        return eval(model_res[metric])[variable][0]
    except Exception as e:
        return -1

def find_model(project, label, outcome, epoch=None, kfold=None):
    """Searches for a model in a project model directory.

    Parameters:
    ----------
    project (slideflow.Project): Project.
    label (str): Experimental label.
    outcome (str): Outcome name. 
    epoch (int, optional): Epoch to search for. If not None, returns
        path to the saved model. If None, returns path to parent model
        folder. Defaults to None.
    kfold (int, optional): K-fold iteration. Defaults to None.

    Returns:
    ----------
    str: Path to matching model.
    """
    tail = '' if kfold is None else f'-kfold{kfold}'
    model_name = f'{outcome}-{label}-HP0{tail}'
    matching = [
        o for o in os.listdir(project.models_dir)
        if o[6:] == model_name
    ]
    if len(matching) > 1:
        msg = f"Multiple matching models found matching {model_name}"
        raise Exception(msg)
    elif not len(matching):
        msg = f"No matching model found matching {model_name}."
        raise Exception(msg)
    elif epoch is not None:
        return join(
            project.models_dir,
            matching[0],
            f'{outcome}-{label}-HP0{tail}_epoch{epoch}'
        )
    else:
        return join(project.models_dir, matching[0])

def find_cv(project, label, outcome, epoch=None, k=3):
    """Finds paths to cross-validation models.

    Parameters:
    ----------
    project (slideflow.Project): Project.
    label (str): Experimental label.
    outcome (str): Outcome name
    epoch (int, optional): Epoch number of saved model. Defaults to None.
    kfold (int, optional): K-fold iteration. Defaults to None.


    Returns:
    ----------
    list(str): Paths to cross-validation models.
    """
    return [
        find_model(project, label, outcome=outcome, epoch=epoch, kfold=_k)
        for _k in range(1, k+1)
    ]

def get_model_performance(SFP, prefix, count, outcomes = ['odx85', 'GHI_RS_Model_NJEM.2004_PMID.15591335']):
    """Get the performance of a specified model when many iterations are being tested during Bayesian optimization

    Parameters
    ----------
    SFP - slideflow project in which model is housed
    prefix - experiment prefix 
    count - iteration of model to return
    outcomes - list of outcomes used in optimization
    
    Returns
    ----------
    Average Tile AUROC over 3 cross folds for speicifed model
    """    
    res = []
    cv_models = None
    outcome = "odx85"
    for outcome in outcomes:
        try:
            cv_models = find_cv(
                project=SFP,
                label=prefix + str(count),
                outcome=outcome
            )
            break
        except Exception as e:
            continue
    if cv_models:
        for m in cv_models:
            if get_model_results(m, metric="tile_auc", variable=outcome) == -1:
                continue
            res += [get_model_results(m, metric="tile_auc", variable=outcome)]
        return sum(res)/len(res)
    return 0

def get_model_hyperparameters(SFP, prefix, count, outcomes = ['odx85', 'GHI_RS_Model_NJEM.2004_PMID.15591335']):
    """Get the hyperparameters of specified model with variable possible outcomes

    Parameters
    ----------
    SFP - slideflow project in which model is housed
    prefix - experiment prefix 
    count - iteration of model to return
    outcomes - list of outcomes used in optimization
    
    Returns
    ----------
    Dict of hyperparameters for specified model (or None if unsusccessful)
    """    
    model = None
    for outcome in outcomes:
        try:
            model = find_model(project = SFP, label = prefix + str(count), outcome = outcome, kfold = 1, epoch = 1)
            break
        except Exception as e:
            continue
    if model:
        par = join(model, "params.json")
        with open(par) as json_data:
            return json.load(json_data)['hp']
    return None

def assign_tumor_roi_model(SFP_TUMOR_ROI, tile_px, normalizer):
    roi_prefix = "BRCA"
    if tile_px == 512:
        if normalizer == "reinhard_fast":
            return find_model(SFP_TUMOR_ROI, roi_prefix + "_RHNORM_512", outcome = "roi", epoch = 1)
        else:
            return find_model(SFP_TUMOR_ROI, roi_prefix + "_NONORM_512", outcome = "roi", epoch = 1)
    if tile_px == 299:
        if normalizer == "reinhard_fast":
            return find_model(SFP_TUMOR_ROI, roi_prefix + "_RHNORM_299", outcome = "roi", epoch = 1)
        else:
            return find_model(SFP_TUMOR_ROI, roi_prefix + "_NONORM_299", outcome = "roi", epoch = 1)   

def get_best_model_params(SFP, prefix, start, max_count):
    """Get the hyperparameters of the best model after optimization

    Parameters
    ----------
    SFP - slideflow project in which model is housed
    prefix - experiment prefix 
    start - first model index for hyperparameter optimization
    max_count - maximum number of models tested during hyperparameter optimization
    
    Returns
    ----------
    Dict of hyperparameters for specified model (or None if unsusccessful)
    """    
    count = start
    cur_max = 0
    cur_index = 0
    while count < max_count:
        this_max = 0
        this_max += get_model_performance(SFP, prefix, count) / 2
        this_max += get_model_performance(SFP, prefix, count + 1) / 2
        if this_max > cur_max:
            cur_max = this_max
            cur_index = count
        count = count + 2
    json_dict = get_model_hyperparameters(SFP, prefix, cur_index)
    export_hpopt_to_csv(SFP, prefix, start, max_count)
    return sf.model.ModelParams.from_dict(json_dict)



def get_runner(SFP, prefix):
    """Get runner to train models over two sets of three crossfolds for prediction of recurrence score and average tile level AUROCs

    Parameters
    ----------
    SFP - slideflow project for model training
    prefix - model prefix
    
    Returns
    ----------
    train_model function for running optimiation
    """    
    def train_model(config):
        model_name = f'odx85-{prefix}-HP0-kfold1'
        matching = [
            o for o in os.listdir(SFP.models_dir)
            if re.search("\d{5}-(odx85|GHI_RS_Model_NJEM\.2004_PMID\.15591335)-"+prefix+"\d{1,3}-kfold1", o)
        ]
        model_count = len(matching) + 1
        hp = ModelParams(
            model='xception', 
            tile_px=299,
            tile_um=302,
            batch_size=config["batch_size"],
            epochs=1,
            early_stop=False,
            early_stop_method='loss',
            dropout=config["dropout"],
            uq=False,
            hidden_layer_width=config["hidden_layer_width"],
            optimizer='Adam',
            learning_rate=config["learning_rate"],
            learning_rate_decay=0,
            loss=config["loss"],
            normalizer=config["normalizer"],
            normalizer_source=None,
            include_top=False,
            hidden_layers=config["hidden_layers"],
            pooling='avg',
            weight_model=None,
            onehot_pool = 'false',
            augment=config["augment"])
        if config["l1_weight"]:
            hp.l1 = config["l1"]
        if config["l2_weight"]:
            hp.l2 = config["l2"]
        if config["l1_dense_weight"]:
            hp.l1_dense = config["l1_dense"]
        if config["l2_dense_weight"]:
            hp.l2_dense = config["l2_dense"]

        if config["learning_rate_decay_true"]:
            hp.learning_rate_decay = config["learning_rate_decay"]
            hp.learning_rate_decay_steps = config["learning_rate_decay_steps"]
        if config["normalizer"] == "none":
            hp.normalizer = None
        if config["loss"] == "sparse_categorical_crossentropy":
            hp.onehot_pool = 'true'
        outcome = "odx85"
        res_avg = 0
        for cv_head in ["CV3_odx85_mip","CV3_mp85_mip"]:
            if config["loss"] == "sparse_categorical_crossentropy":
                outcome = "odx85"
                SFP.train(exp_label = prefix + str(model_count), outcome_label_headers="odx85", params = hp, val_strategy = 'k-fold-manual', val_k_fold_header=cv_head, multi_gpu = True)
            else:
                outcome = "GHI_RS_Model_NJEM.2004_PMID.15591335"
                SFP.train(exp_label = prefix + str(model_count), outcome_label_headers="GHI_RS_Model_NJEM.2004_PMID.15591335", val_outcome_label_headers="odx85", params = hp, val_strategy = 'k-fold-manual', val_k_fold_header=cv_head, multi_gpu = True)
            
            cv_models = find_cv(
                project=SFP,
                label=prefix + str(model_count),
                outcome=outcome
            )
            model_count = model_count + 1
            res = []
            for m in cv_models:
                if get_model_results(m, metric="tile_auc", variable=outcome) == -1:
                    continue
                res += [get_model_results(m, metric="tile_auc", variable=outcome)]             
            res_avg += sum(res)/len(res)
        return 1 - res_avg
    return train_model
        
def hyperparameter_optimization(SFP, prefix, runcount = 50):
    """Run Bayesian Optimization of model hyperparameters

    Parameters
    ----------
    SFP - slideflow project for model optimization
    prefix - model prefix to be used for hyperparameter optimization
    runcount - number of sets of hyperparameters to evaluate
    """    
    configspace = ConfigurationSpace()
    configspace.add_hyperparameter(UniformFloatHyperparameter("dropout", 0, 0.5))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("hidden_layer_width", 128, 1024))
    configspace.add_hyperparameter(UniformFloatHyperparameter("learning_rate", 0.00001, 0.001))
    lr_decay_steps = UniformIntegerHyperparameter("learning_rate_decay_steps", 128, 1024)
    lr_decay_rate = UniformFloatHyperparameter("learning_rate_decay", 0, 1)
    lr_decay = CategoricalHyperparameter("learning_rate_decay_true", [True, False], default_value = False)
    configspace.add_hyperparameters([lr_decay_steps, lr_decay_rate, lr_decay])
    configspace.add_condition(CS.EqualsCondition(lr_decay_steps, lr_decay, True))
    configspace.add_condition(CS.EqualsCondition(lr_decay_rate, lr_decay, True))
    configspace.add_hyperparameter(CategoricalHyperparameter("loss", ["mean_squared_error", "mean_absolute_error", "sparse_categorical_crossentropy"], default_value = "mean_absolute_error"))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("batch_size", 8,128))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("hidden_layers", 1,5))
    configspace.add_hyperparameter(CategoricalHyperparameter("augment", ["xyr", "xyrj", "xyrjb"], default_value = "xyrjb"))
    configspace.add_hyperparameter(CategoricalHyperparameter("normalizer", ["reinhard_fast", "none"], default_value = "none"))
    l1 = UniformFloatHyperparameter("l1", 0, 0.1)
    l1_weight = CategoricalHyperparameter("l1_weight", [True, False], default_value = False)
    l2 = UniformFloatHyperparameter("l2", 0, 0.1)
    l2_weight = CategoricalHyperparameter("l2_weight", [True, False], default_value = False)
    l1_dense = UniformFloatHyperparameter("l1_dense", 0, 0.1)
    l1_dense_weight = CategoricalHyperparameter("l1_dense_weight", [True, False], default_value = False)
    l2_dense = UniformFloatHyperparameter("l2_dense", 0, 0.1)
    l2_dense_weight = CategoricalHyperparameter("l2_dense_weight", [True, False], default_value = False)
    configspace.add_hyperparameters([l1, l1_weight, l2, l2_weight, l1_dense, l1_dense_weight, l2_dense, l2_dense_weight])
    configspace.add_condition(CS.EqualsCondition(l1, l1_weight, True))
    configspace.add_condition(CS.EqualsCondition(l2, l2_weight, True))
    configspace.add_condition(CS.EqualsCondition(l1_dense, l1_dense_weight, True))
    configspace.add_condition(CS.EqualsCondition(l2_dense, l2_dense_weight, True))
    configspace.add_condition(CS.EqualsCondition(l2_dense, l2_dense_weight, True))

    # Provide meta data for the optimization
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality (alternatively runtime)
        "runcount-limit": runcount,  # Max number of function evaluations (the more the better)
        "cs": configspace,
    })

    smac = SMAC4BB(scenario=scenario, tae_runner=get_runner(SFP, prefix))
    smac.optimize()

    

def brca_cancer_detection_module(tile_px = 299, tile_um = 302, normalizer = 'reinhard_fast'):
    """Trains and validates a breast cancer detection module from TCGA-BRCA

    Parameters
    ----------
    tile_px - tile pixel size
    tile_um - tile size in micrometers
    normalizer - normalizer method ('reinhard_fast' or None)
    """    
    SFP = sf.Project(join(PROJECT_ROOT, "TCGA_BRCA_ROI"))
    SFP.annotations = join(PROJECT_ROOT, "TCGA_BRCA_ROI", "tumor_roi.csv")
    SFP.sources = ["TCGA_BRCA_FULL_ROI", "TCGA_BRCA_NORMAL"]

    hp = hp_roi
    hp.tile_px = tile_px
    hp.tile_um = tile_um
    hp.normalizer = normalizer
    exp_label = "BRCA_NONORM_"
    if normalizer == 'reinhard_fast':
        exp_label = "BRCA_RHNORM_"
    exp_label += str(tile_px)
    SFP.train(exp_label = exp_label, outcome_label_headers="roi", params = hp, val_strategy = 'none', multi_gpu = True)
    exp_label = "brca_roi_evaluate"
    if normalizer == 'reinhard_fast':
        exp_label += "_rhn"
    exp_label += "_" + str(tile_px)
    SFP.train(exp_label=exp_label, outcome_label_headers = 'roi', val_strategy = 'k-fold-manual', val_k_fold=3, val_k_fold_header='CV3_odx85_mip', params = hp, multi_gpu = True, save_predictions=True)
    cv_models = find_cv(
        project=SFP,
        label=exp_label,
        outcome="roi"
    )
    res = []
    for m in cv_models:
        if get_model_results(m, metric="tile_auc", variable="roi") == -1:
            continue
        res += [get_model_results(m, metric="tile_auc", variable="roi")]
    print("AVERAGE TILE-LEVEL AUROC FOR CANCER PREDICTION: " + str(sum(res)/len(res)))


def export_hpopt_to_csv(SFP, prefix, start, max_count):
    """Exports runs of hyperparameter optimization to csv

    Parameters
    ----------
    SFP - slideflow project used for hyperparameter optimization
    prefix - prefix used for hyperparameter optimization
    start - start count of models to include in export_hpopt_to_csv
    max_count - total number of models to include in export
    """    
    count = start
    all_points = None
    while count < max_count:
        this_max = 0
        this_max += get_model_performance(SFP, prefix, count) / 2
        this_max += get_model_performance(SFP, prefix, count + 1) / 2
        if this_max == 0:
            break
        hp = get_model_hyperparameters(SFP, prefix, count)
        if all_points:
            for k, v in hp.items():
                all_points[k] += [v]
            all_points["avg_tile_auc"] += [this_max]
        else:
            all_points = {}
            for k,v in hp.items():
                all_points[k] = [v]
            all_points["avg_tile_auc"] = [this_max]
        count = count + 2
    pd.DataFrame.from_dict(all_points).to_csv(join(PROJECT_ROOT, "hpopt.csv"))
    
def setup_projects(tile_px = 299, tile_um = 302, overwrite = False):
    """Extracts tiles to setup all projects

    Parameters
    ----------
    tile_px - tile pixel size
    tile_um - tile size in micrometers

    """    
    SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS"))
    SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "uch_brca_complete.csv")
    SFP.sources = ["UCH_BRCA_RS"]
    SFP.extract_tiles(tile_px=tile_px, tile_um=tile_um, skip_missing_roi=False, save_tiles=False, skip_extracted=overwrite, roi_method = 'ignore', source="UCH_BRCA_RS", buffer=join(PROJECT_ROOT, "buffer"))
    SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "tcga_brca_complete.csv")
    SFP.sources = ["TCGA_BRCA_FULL_ROI"]
    SFP.extract_tiles(tile_px=tile_px, tile_um=tile_um, skip_missing_roi=False, save_tiles=False, skip_extracted=overwrite, roi_method = 'inside', source="TCGA_BRCA_FULL_ROI", buffer=join(PROJECT_ROOT, "buffer"))
    SFP.sources = ["TCGA_BRCA_NO_ROI"]
    SFP.extract_tiles(tile_px=tile_px, tile_um=tile_um, skip_missing_roi=False, save_tiles=False, skip_extracted=overwrite, roi_method = 'ignore', source="TCGA_BRCA_NO_ROI", buffer=join(PROJECT_ROOT, "buffer"))
    SFP = sf.Project(join(PROJECT_ROOT, "TCGA_BRCA_ROI"))
    SFP.annotations = join(PROJECT_ROOT, "TCGA_BRCA_ROI", "tumor_roi.csv")
    dataset = SFP.dataset(299, 302, verification=None, sources = ["TCGA_BRCA_NORMAL"])
    normal_slide = dataset.slide_paths()
    for slide in normal_slide:
        dir_name = os.path.dirname(slide)
        base_name = os.path.basename(slide)
        shutil.move(join(dir_name, base_name), join(dir_name, "norm_" + base_name))
    SFP.extract_tiles(tile_px=299, tile_um=302, skip_missing_roi=True, save_tiles=False, skip_extracted=overwrite, roi_method = 'outside', source="TCGA_BRCA_NORMAL",  buffer=join(PROJECT_ROOT, "buffer"))
    dataset = SFP.dataset(299, 302, verification=None, sources = ["TCGA_BRCA_NORMAL"])
    normal_slide = dataset.slide_paths()
    for slide in normal_slide:
        dir_name = os.path.dirname(slide)
        base_name = os.path.basename(slide)
        shutil.move(join(dir_name, base_name), join(dir_name, base_name[5:]))

def train_models(hpsearch = 'old', prefix_hpopt = 'hp_new2', start = 0, max_count = 50):
    SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS"))
    SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "tcga_brca_complete.csv")
    SFP.sources = ["TCGA_BRCA_FULL_ROI"]
    SFP_TUMOR_ROI = sf.Project(join(PROJECT_ROOT, "TCGA_BRCA_ROI"))
    hp = None

    #Get optimal hyperparameters for recurrence score prediction
    if hpsearch == 'old':
        hp = hp_opt
    elif hpsearch == 'read':
        hp = get_best_model_params(SFP, prefix = prefix_hpopt, start = start, max_count = max_count *2)

    #Train Breast Cancer Detection Module
    brca_cancer_detection_module(hp.tile_px, hp.tile_um, hp.normalizer)

    hp.weight_model = assign_tumor_roi_model(SFP_TUMOR_ROI, hp.tile_px, hp.normalizer)


    exp_label = "ODX_Final_BRCAROI"
    exp_label_mp = "MP_Final_BRCAROI"
    #Train external model on all data
    SFP.train(exp_label=exp_label, outcome_label_headers="GHI_RS_Model_NJEM.2004_PMID.15591335",  params = hp, val_strategy = 'none', multi_gpu=True, save_predictions=True)
    SFP.train(exp_label=exp_label_mp, outcome_label_headers="Pcorr_NKI70_Good_Correlation_Nature.2002_PMID.11823860",  params = hp, val_strategy = 'none', multi_gpu=True, save_predictions=True)

    #Train internal CV model for logistic regression fitting
    SFP.train(exp_label=exp_label, outcome_label_headers="GHI_RS_Model_NJEM.2004_PMID.15591335",  val_outcome_label_headers="odx85", params = hp, val_strategy = 'k-fold-manual', val_k_fold=3, val_k_fold_header = "CV3_odx85_mip", multi_gpu=True, save_predictions=True)
    SFP.train(exp_label=exp_label_mp, outcome_label_headers="Pcorr_NKI70_Good_Correlation_Nature.2002_PMID.11823860",  val_outcome_label_headers="mphr", params = hp, val_strategy = 'k-fold-manual', val_k_fold=3, val_k_fold_header = "CV3_mp85_mip", multi_gpu=True, save_predictions=True)

    exp_label_roi = "brca_roi_evaluate"
    if hp.normalizer == 'reinhard_fast':
        exp_label_roi += "_rhn"
    exp_label_roi += "_" + str(hp.tile_px)

    SFP.sources = ["TCGA_BRCA_NO_ROI"]
    for i in [1,2,3]:
        m = find_model(SFP, exp_label, outcome='GHI_RS_Model_NJEM.2004_PMID.15591335', epoch=1, kfold=i)
        params = sf.util.get_model_config(m)
        params["hp"]["loss"] = "sparse_categorical_crossentropy"
        params["model_type"] = "categorical"
        params["outcome_labels"] = {"0":"H","1":"L"}
        params["onehot_pool"] = 'false'
        params["weight_model"] = find_model(SFP_TUMOR_ROI, exp_label_roi, outcome='roi', epoch=1, kfold=i)
        sf.util.write_json(params, join(m, "params_eval.json"))
        SFP.evaluate(m, outcome_label_headers = 'odx85', filters={'CV3_odx85_mip':str(i)}, save_predictions=True, model_config=join(m, "params_eval.json"))

        m = find_model(SFP, exp_label_mp, outcome='Pcorr_NKI70_Good_Correlation_Nature.2002_PMID.11823860', epoch=1, kfold=i)
        params = sf.util.get_model_config(m)
        params["hp"]["loss"] = "sparse_categorical_crossentropy"
        params["model_type"] = "categorical"
        params["outcome_labels"] = {"0":"H","1":"L"}
        params["onehot_pool"] = 'false'
        params["weight_model"] = find_model(SFP_TUMOR_ROI, exp_label_roi, outcome='roi', epoch=1, kfold=i)
        sf.util.write_json(params, join(m, "params_eval.json"))
        SFP.evaluate(m, outcome_label_headers = 'mphr', filters={'CV3_mp85_mip':str(i)}, save_predictions=True, model_config=join(m, "params_eval.json"))

def test_models():
    #Finally, validate on external dataset
    SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS"))
    SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "uch_brca_complete.csv")
    SFP.sources = ["UCH_BRCA_RS"]
    SFP_TUMOR_ROI = sf.Project(join(PROJECT_ROOT, "TCGA_BRCA_ROI"))
    SFP_TUMOR_ROI.annotations = join(PROJECT_ROOT, "TCGA_BRCA_ROI", "tumor_roi.csv")

    SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "uch_brca_complete.csv")
    m = find_model(SFP, exp_label, outcome = "GHI_RS_Model_NJEM.2004_PMID.15591335", epoch=hp.epochs[0])
    params = sf.util.get_model_config(m)
    params["hp"]["loss"] = "sparse_categorical_crossentropy"
    params["model_type"] = "categorical"
    params["outcome_labels"] = {"0":"H","1":"L"}
    params["onehot_pool"] = 'false'
    sf.util.write_json(params, join(m, "params_eval.json"))
    SFP.evaluate(model=m, outcome_label_headers="RSHigh", save_predictions=True, model_config=join(m, "params_eval.json"))

    m = find_model(SFP, exp_label_mp, outcome = "Pcorr_NKI70_Good_Correlation_Nature.2002_PMID.11823860", epoch=hp.epochs[0])
    params = sf.util.get_model_config(m)
    params["hp"]["loss"] = "sparse_categorical_crossentropy"
    params["model_type"] = "categorical"
    params["outcome_labels"] = {"0":"H","1":"L"}
    params["onehot_pool"] = 'false'
    sf.util.write_json(params, join(m, "params_eval.json"))
    SFP.evaluate(model=m, outcome_label_headers="MPHigh", save_predictions=True, model_config=join(m, "params_eval.json"))

    
def main(): 
    parser = argparse.ArgumentParser(description = "Helper to guide through model training.")
    parser.add_argument('-p', '--project_root', required=False, type=str, help='Path to project directory (if not provided, assumes subdirectory of this script).')
    parser.add_argument('--hpsearch', required=False, type=str, help='Set to \'old\' to use saved hyperparameters, \'run\' to perform hyperparameter search, \'read\' to find best hyperparameters from prior run.')
    parser.add_argument('-e', '--extract', required=False, action="store_true", help='If provided, will extract tiles from slide directory.')
    parser.add_argument('-t', '--train', required=False, action="store_true", help='If provided, will train models in TCGA for tumor ROI / recurrnece score prediction.')
    parser.add_argument('-v', '--validate', required=False, action="store_true", help='If provided, will validate models in the University of Chicago dataset.')

    parser.add_argument('--hpprefix', required=False, type=str, help='Provide prefix for models trained during hyperparameter search (must be specified if desired to read hyperparameters from old HP search).')
    parser.add_argument('--hpstart', required=False, type=int, help='Provide the starting index for models to check among prior hyperparameter search.')
    parser.add_argument('--hpcount', required=False, type=int, help='Provide the number of models to check among prior hyperparameter search.')
    parser.add_argument('--heatmaps_tumor_roi', required=False, type=str, help='Will save heatmaps for tumor region of interest model. Set to TCGA to provide heatmaps from TCGA and UCH to provide heatmaps from validation dataset.')
    parser.add_argument('--heatmaps_odx_roi', required=False, type=str, help='Will save heatmaps for OncotypeDx model. Set to TCGA to provide heatmaps from TCGA and UCH to provide heatmaps from validation dataset.')

    args = parser.parse_args()  
    global PROJECT_ROOT
    if args.project_root:
        PROJECT_ROOT = args.project_root
    if args.extract:
        setup_projects()
    if not args.hpsearch:
        args.hpsearch = 'old'
    if not args.hpprefix:
        args.hpprefix = 'hp_new2'
    if not args.hpstart:
        args.hpstart = 0
    if not args.hpcount:
        args.hpcount = 50
    if args.hpsearch == 'run':
        SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS"))
        SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "tcga_brca_complete.csv")
        SFP.sources = ["TCGA_BRCA_FULL_ROI"]
        hyperparameter_optimization(SFP, args.hpprefix, args.hpcount)
        args.hpsearch = 'read'
    if args.train:
        train_models(args.hpsearch, args.hpprefix, args.hpstart, args.hpcount)
    if args.validate:
        test_models()
    if args.heatmaps_odx_roi:
        if args.heatmaps_odx_roi == 'TCGA':
            SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS"))
            exp_label = "ODX_Final_BRCAROI"
            SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "tcga_brca_complete.csv")            
            SFP.sources = ["TCGA_BRCA_FULL_ROI"]
            for i in [1,2,3]:
                m = find_model(SFP, exp_label, outcome='GHI_RS_Model_NJEM.2004_PMID.15591335', epoch=1, kfold=i)
                SFP.generate_heatmaps(model=m, filters={'CV3_odx85_mip':str(i)}, outdir = join(PROJECT_ROOT, 'UCH_RS/heatmaps_tcga'), resolution='low', batch_size=32, roi_method ='none', show_roi = True, buffer=join(PROJECT_ROOT, 'buffer'))
        if args.heatmaps_odx_roi == 'UCH':
            SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS"))
            SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "uch_brca_complete.csv")
            SFP.sources = ["UCH_BRCA_RS"]
            exp_label = "ODX_Final_BRCAROI"
            m = find_model(SFP, exp_label, outcome='GHI_RS_Model_NJEM.2004_PMID.15591335', epoch=1)
            SFP.generate_heatmaps(model=m, outdir = join(PROJECT_ROOT, 'UCH_RS/heatmaps_uch'), resolution='low', batch_size=32, roi_method ='none', show_roi = True, buffer=join(PROJECT_ROOT, 'buffer'))
    if args.heatmaps_tumor_roi:
        hp = None
        if hpsearch == 'old':
            hp = hp_opt
        elif hpsearch == 'read':
            hp = get_best_model_params(SFP, args.hpprefix, args.hpstart, 2*args.hpcount)
        normalizer = hp.normalizer
        tile_px = hp.tile_px
        if args.heatmaps_tumor_roi == 'TCGA':
            SFP = sf.Project(join(PROJECT_ROOT, "TCGA_BRCA_ROI")) 
            SFP.sources = ["TCGA_BRCA_FULL_ROI"]
            SFP.annotations = join(PROJECT_ROOT, "TCGA_BRCA_ROI", "tumor_roi.csv")
            exp_label = "brca_roi_evaluate"
            if normalizer == 'reinhard_fast':
                exp_label += "_rhn"
            exp_label += "_" + str(tile_px)
            for i in [1,2,3]:
                m = find_model(SFP, exp_label, epoch=1, kfold=i, outcome='roi')
                SFP.generate_heatmaps(model=m, filters={'CV3_odx85_mip':str(i)}, outdir = join(PROJECT_ROOT, 'TCGA_BRCA_ROI/heatmaps_tcga'), resolution='low', batch_size=32, roi_method ='none', show_roi = True, buffer=join(PROJECT_ROOT, 'buffer'))

        if args.heatmaps_tumor_roi == 'UCH':
            SFP = sf.Project(join(PROJECT_ROOT, "TCGA_BRCA_ROI")) 
            SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "uch_brca_complete.csv")
            SFP.sources = ["UCH_BRCA_RS"]
            exp_label = "BRCA_NONORM_"
            if normalizer == 'reinhard_fast':
                exp_label = "BRCA_RHNORM_"
            exp_label += str(tile_px)
            m = find_model(SFP, exp_label, outcome='GHI_RS_Model_NJEM.2004_PMID.15591335', epoch=1)
            SFP.generate_heatmaps(model=m, outdir = join(PROJECT_ROOT, 'TCGA_BRCA_ROI/heatmaps_uch'), resolution='low', batch_size=32, roi_method ='none', show_roi = True, buffer=join(PROJECT_ROOT, 'buffer'))



if __name__ == '__main__':
    main()
