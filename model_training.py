import os
from slideflow.model import ModelParams
import matplotlib.pyplot as plt
import ConfigSpace as CS
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario
import pandas as pd
from os.path import join, isfile, exists
from slideflow.util import log
import slideflow as sf
import json
import pandas as pd
import shutil
from os.path import basename, dirname, join
import argparse
import tensorflow as tf
from tqdm import tqdm
from slideflow.io import tfrecords
import re

PROJECT_ROOT = os.getcwd() + "/PROJECTS/"

odx_train = "odx_train"
mp_train = "mp_train"

exp_suffix = "Final_BRCAROI"

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

def get_model_performance(SFP, prefix, count, outcomes = None):
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
    global odx_train
    if outcomes is None:
        outcomes = ['odx85', odx_train]
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
            if get_model_results(m, metric="tile_auc_avgpool", variable=outcome) == -1:
                continue
            res += [get_model_results(m, metric="tile_auc_avgpool", variable=outcome)]
        try:
            return sum(res)/len(res)
        except:
            return 0
    return 0

def get_model_hyperparameters(SFP, prefix, count, outcomes = None):
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
    if outcomes is None:
        outcomes = ['odx85', odx_train]
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
    """After training, finds the appropriate tumor ROI model based on tile size and normalizer use
    Parameters
    ----------
    SFP_TUMOR_ROI - the slideflow project in which the ROI model was trained (default /PROJECTS/TCGA_BRCA_ROI)
    tile_px - tile pixel size
    normalizer - normalizer method ('reinhard_fast' or None)
    
    Returns
    ----------
    File path to tumor detection model
    """    
    roi_prefix = "BRCA"
    if normalizer == "reinhard_fast":
        return find_model(SFP_TUMOR_ROI, roi_prefix + "_RHNORM_" + str(tile_px), outcome = "roi", epoch = 1)
    else:
        return find_model(SFP_TUMOR_ROI, roi_prefix + "_NONORM_"  + str(tile_px), outcome = "roi", epoch = 1)


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
        global odx_train
        model_name = f'odx85-{prefix}-HP0-kfold1'
        matching = [
            o for o in os.listdir(SFP.models_dir)
            if re.search("\d{5}-(odx85|" + odx_train + ")-"+prefix+"\d{1,3}-HP0-kfold1", o)
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
                outcome = odx_train
                SFP.train(exp_label = prefix + str(model_count), outcome_label_headers=odx_train, val_outcome_label_headers="odx85", params = hp, val_strategy = 'k-fold-manual', val_k_fold_header=cv_head, multi_gpu = True)
            
            cv_models = find_cv(
                project=SFP,
                label=prefix + str(model_count),
                outcome=outcome
            )
            model_count = model_count + 1
            res = []
            for m in cv_models:
                if get_model_results(m, metric="tile_auc_avgpool", variable=outcome) == -1:
                    continue
                res += [get_model_results(m, metric="tile_auc_avgpool", variable=outcome)]    
            try:
                res_avg += sum(res)/len(res)
            except:
                print("NO VIABLE MODELS TO TEST")
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

def test_brca_cancer_detection(tile_px = 299, tile_um = 302, normalizer = 'reinhard_fast'):
    """Tests a trained cancer detection model with 3 fold cross validation. Since slides are extracted
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
    exp_label = "brca_roi_evaluate"
    if normalizer == 'reinhard_fast':
        exp_label += "_rhn"
    exp_label += "_" + str(tile_px)
    SFP.train(exp_label=exp_label, outcome_label_headers = 'roi', val_strategy = 'k-fold-manual', val_k_fold=3, val_k_fold_header='CV3_odx85_mip', params = hp, multi_gpu = True, save_predictions=True)

def generateFilteredDataset(outdir, source, model):
    """Generates a 'filtered' version of an existing dataset, by removing tiles that are not predicted to be tumor (with at least 50% likelihood) by a specified tumor detection model
    
    Parameters
    ----------
    outdir - directory to save tfrecords
    source - directory of source tfrecords
    model - tumor detection model to use
    
    """    
    k_model = tf.keras.models.load_model(model)
    wsi_tfrecords = [tfr for tfr in os.listdir(source) if isfile(join(source, tfr)) and sf.util.path_to_ext(tfr) == 'tfrecords']

    if not exists(outdir):
        os.makedirs(outdir)

    pb = tqdm(wsi_tfrecords, ncols=80)
    for tfr in pb:
        pb.set_description("Working...")
        num_wrote = 0
        tfr_path = join(source, tfr)
        parser = tfrecords.get_tfrecord_parser(tfr_path, ('image_raw',), decode_images=True, standardize=True, img_size=299)
        pred_dataset = tf.data.TFRecordDataset(tfr_path)
        pred_dataset = pred_dataset.map(parser, num_parallel_calls=8)
        pred_dataset = pred_dataset.batch(128, drop_remainder=False)
        roi_pred = k_model.predict(pred_dataset)
            
        writer = tf.io.TFRecordWriter(join(outdir, tfr))
        dataset = tf.data.TFRecordDataset(tfr_path)
        feature_description, _ = tfrecords.detect_tfrecord_format(tfr_path)
        for i, record in enumerate(dataset):
            if roi_pred[i][1] > 0.5:
                writer.write(tfrecords._read_and_return_record(record, feature_description))
                num_wrote += 1
        tqdm.write(f'Finished {tfr} : wrote {num_wrote}')
        writer.close()

def generateFilteredBRCA(tile_px = 299, tile_um = 302, normalizer = 'reinhard_fast'):
    """Generates a 'filtered' version of the TCGA_BRCA_NO_ROI dataset, by removing tiles that are not predicted to be tumor (with at least 50% likelihood) by a specified tumor detection model 
    Parameters
    ----------
    tile_px - tile pixel size (to specify tumor detection model)
    tile_um - tile size in micrometers (to specify tumor detection model)
    normalizer - normalizer method ('reinhard_fast' or None, used to specify tumor detection model)
    
    """    
    SFP = sf.Project(join(PROJECT_ROOT, "TCGA_BRCA_ROI"))
    loaded_config = sf.util.load_json(SFP.dataset_config)
    loaded_sources = {k:v for (k,v) in loaded_config.items()}
    label = f"{tile_px}px_{tile_um}um"
    generateFilteredDataset(outdir = os.path.join(loaded_sources["TCGA_BRCA_FILTERED"]["tfrecords"], label), source = os.path.join(loaded_sources["TCGA_BRCA_NO_ROI"]["tfrecords"], label), model = assign_tumor_roi_model(SFP, tile_px, normalizer)) 
    
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
    
def setup_projects(tile_px = 299, tile_um = 302, overwrite = False, annotation = None, source = None):
    """Extracts tiles to setup all projects

    Parameters
    ----------
    tile_px - tile pixel size
    tile_um - tile size in micrometers
    overwrite - whether to overwrite existing data - defaults to False
    annotation - if provided, will extract tiles for a custom validation dataset, where annotations are in the /UCH_RS/<annotation> csv file
    source - if provided, will extract tiles for a custom validation dataset of the name provided
    
    """    
    SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS"))
    SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "uch_brca_complete.csv")
    SFP.sources = ["UCH_BRCA_RS"]
    if annotation:
        SFP.annotations = join(PROJECT_ROOT, "UCH_RS", annotation)
    if source:
        SFP.sources = [source]
    if annotation or source:
        SFP.extract_tiles(tile_px=tile_px, tile_um=tile_um, skip_missing_roi=False, save_tiles=False, skip_extracted=overwrite, roi_method = 'ignore', source=source, buffer=join(PROJECT_ROOT, "buffer"))
    else:
        try:
            SFP.extract_tiles(tile_px=tile_px, tile_um=tile_um, skip_missing_roi=False, save_tiles=False, skip_extracted=overwrite, roi_method = 'ignore', source="UCH_BRCA_RS", buffer=join(PROJECT_ROOT, "buffer"))
        except:
            print("COULD NOT EXTRACT UCH_BRCA_RS DATASET")
        SFP.sources = ["UCH_BRCA_RS_FULL_ROI"]
        try:
            SFP.extract_tiles(tile_px=tile_px, tile_um=tile_um, skip_missing_roi=False, save_tiles=False, skip_extracted=overwrite, roi_method = 'inside', source="UCH_BRCA_RS_FULL_ROI", buffer=join(PROJECT_ROOT, "buffer"))
        except:
            print("COULD NOT EXTRACT UCH_BRCA_RS_FULL_ROI DATASET")
    SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "tcga_brca_complete.csv")
    SFP.sources = ["TCGA_BRCA_FULL_ROI"]
    SFP.extract_tiles(tile_px=tile_px, tile_um=tile_um, skip_missing_roi=False, save_tiles=False, skip_extracted=overwrite, roi_method = 'inside', source="TCGA_BRCA_FULL_ROI", buffer=join(PROJECT_ROOT, "buffer"))
    generateFilteredBRCA()
    SFP.sources = ["TCGA_BRCA_NO_ROI"]
    SFP.extract_tiles(tile_px=tile_px, tile_um=tile_um, skip_missing_roi=False, save_tiles=False, skip_extracted=overwrite, roi_method = 'ignore', source="TCGA_BRCA_NO_ROI", buffer=join(PROJECT_ROOT, "buffer"))
    SFP = sf.Project(join(PROJECT_ROOT, "TCGA_BRCA_ROI"))
    SFP.annotations = join(PROJECT_ROOT, "TCGA_BRCA_ROI", "tumor_roi.csv")
    SFP.extract_tiles(tile_px=299, tile_um=302, skip_missing_roi=True, save_tiles=False, skip_extracted=overwrite, roi_method = 'outside', source="TCGA_BRCA_NORMAL",  buffer=join(PROJECT_ROOT, "buffer"))
    dataset = SFP.dataset(299, 302, verification=None, sources = ["TCGA_BRCA_NORMAL"])
    normal_slide = dataset.slide_paths()
    for slide in normal_slide:
        dir_name = os.path.dirname(slide)
        base_name = os.path.basename(slide)
        shutil.move(join(dir_name, base_name), join(dir_name, base_name[5:]))


def train_models(hpsearch = 'old', prefix_hpopt = 'hp_new2', start = 1, max_count = 50, train_receptors = False, use_filtered = False, train_reverse = False):
    """Trains OncotypeDx and MammaPrint prediction models in TCGA, and generates cross validated predictions

    Parameters
    ----------
    hpsearch - specifies whether to load existing hyperparameters ('old'), or scan for best hyperparameters from new hyperparameter search ('read')
    prefix_hpopt - specifies the prefix for hyperparameter training when reading model results from new hyperparameter search
    start - specifies the starting iteration when reading from a new hyperparameter search
    max_count - specifies the final iteration when reading from a new hyperparameter search
    train_receptors - if True, trains models on HR+/HER2- patients instead of all of TCGA
    use_filtered - if True, trains models on the TCGA_BRCA_FILTERED dataset (filtered to exclude tiles with a < 50% likelihood of being tumor)
    train_reverse - if True, training / cross validation is performed on the UCMC dataset instead of TCGA

    """   
    global odx_train, mp_train
    SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS"))
    SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "tcga_brca_complete.csv")
    SFP.sources = ["TCGA_BRCA_FULL_ROI"]
    SFP_TUMOR_ROI = sf.Project(join(PROJECT_ROOT, "TCGA_BRCA_ROI"))
    hp = None
    filters = None
    #Prepare for additional experiments as per settings
    exp_additions = ""
    
    odx_train_name = odx_train
    odx_val_name = "odx85"
    mp_train_name = mp_train
    mp_val_name = "mphr"
    
    if train_receptors:
        exp_additions += "_TR"
        filters = {"ER_Status_By_IHC":["positive"], "HER2Calc":["negative"]}
            
    if use_filtered:
        exp_additions += "_UF"
        SFP.sources = ["TCGA_BRCA_FILTERED"]

    if train_reverse:
        exp_additions += "_REV"
        odx_train_name = "RS"
        odx_val_name = "RSHigh"
        mp_train_name = "mpscore"
        mp_val_name = "MPHigh"
    #Get optimal hyperparameters for recurrence score prediction
    if hpsearch == 'old':
        hp = hp_opt
    elif hpsearch == 'read':
        hp = get_best_model_params(SFP, prefix = prefix_hpopt, start = start, max_count = max_count *2)

    #Train Breast Cancer Detection Module
    try:
        #Try to load an existing module
        hp.weight_model = assign_tumor_roi_model(SFP_TUMOR_ROI, hp.tile_px, hp.normalizer)
    except:
        #If no module available, train a new one
        brca_cancer_detection_module(hp.tile_px, hp.tile_um, hp.normalizer)
        hp.weight_model = assign_tumor_roi_model(SFP_TUMOR_ROI, hp.tile_px, hp.normalizer)

    global exp_suffix
    exp_label = "ODX_" + exp_suffix + exp_additions 
    exp_label_mp = "MP_" + exp_suffix + exp_additions
    
    if train_reverse:
        SFP.sources = ["UCH_BRCA_RS_FULL_ROI"]
        SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "uch_brca_complete.csv")
    
    #Train external model on all data
    try:
        m = find_model(SFP, exp_label, outcome=odx_train_name, epoch=hp.epochs[0])
        print("Oncotype DX Model Exists with Experiment Label - Skipping Training on Full Dataset")
        print("Change Experiment Name or Delete Model Folder from /PROJECTS/UCH_RS/model/ to Repeat Training")
    except:
        print("Oncotype DX Model Exists with Experiment Label - Training on Full Dataset")
        SFP.train(exp_label=exp_label, outcome_label_headers=odx_train_name,  params = hp, val_strategy = 'none', filters = filters, multi_gpu=True, save_predictions=True)
    try:
        m = find_model(SFP, exp_label_mp, outcome=mp_train_name, epoch=hp.epochs[0])
        print("MammaPrint Model Exist with Experiment Label - Skipping Training on Full Dataset")
        print("Change Experiment Name or Delete Model Folder from /PROJECTS/UCH_RS/model/ to Repeat Training")
    except:
        print("MammaPrint Model Does Not Exist with Experiment Label - Training on Full Dataset")
        SFP.train(exp_label=exp_label_mp, outcome_label_headers=mp_train_name,  params = hp, val_strategy = 'none', filters = filters, multi_gpu=True, save_predictions=True)

    def mergeDict(dict1, dict2):
        if dict1 == None:
            return dict2
        else:
            return dict1.update(dict2)
    #Train internal CV model for logistic regression fitting
    try:
        m = find_model(SFP, exp_label, outcome=odx_train_name, epoch=hp.epochs[0], kfold =1)
        print("Oncotype DX Model Exists with Experiment Label - Skipping Crossfold Training")
        print("Change Experiment Name or Delete Model Folder from /PROJECTS/UCH_RS/model/ to Repeat Training")
    except:
        print("Oncotype DX Model Does Not Exist with Experiment Label - Performing Crossfold Training")
        SFP.train(exp_label=exp_label, outcome_label_headers=odx_train_name,  val_outcome_label_headers=odx_val_name, params = hp, filters = mergeDict(filters, {odx_val_name: ["H","L"]}), val_strategy = 'k-fold-manual', val_k_fold=3, val_k_fold_header = "CV3_odx85_mip", multi_gpu=True, save_predictions=True)
    try:
        m = find_model(SFP, exp_label_mp, outcome=mp_train_name, epoch=hp.epochs[0], kfold =1)
        print("MammaPrint Model Exists with Experiment Label - Skipping Crossfold Training")
        print("Change Experiment Name or Delete Model Folder from /PROJECTS/UCH_RS/model/ to Repeat Training")
    except:
        print("MammaPrint Model Does Not Exist with Experiment Label - Performing Crossfold Training")
        SFP.train(exp_label=exp_label_mp, outcome_label_headers=mp_train_name,  val_outcome_label_headers=mp_val_name, filters = mergeDict(filters, {mp_val_name: ["H","L"]}), params = hp, val_strategy = 'k-fold-manual', val_k_fold=3, val_k_fold_header = "CV3_mp85_mip", multi_gpu=True, save_predictions=True)

    if train_reverse:
        SFP.sources = ["UCH_BRCA_RS"]
    else:
        SFP.sources = ["TCGA_BRCA_NO_ROI"]

    for i in [1,2,3]:
        try:
            find_eval(SFP, exp_label, odx_train, hp.epochs[0], i)
            print("Oncotype DX Predictions Exist for Crossfold " + str(i) + " - Skipping Evaluation")
            print("Change Experiment Name or Delete Results Folder from /PROJECTS/UCH_RS/eval/ to Repeat Evaluation")
        except:
            print("Oncotype DX Predictions Do Not Exist for Crossfold " + str(i) + " - Evaluating Crossfold")
            m = find_model(SFP, exp_label, outcome=odx_train_name, epoch=hp.epochs[0], kfold=i)
            params = sf.util.get_model_config(m)
            params["hp"]["loss"] = "sparse_categorical_crossentropy"
            params["model_type"] = "categorical"
            params["outcome_labels"] = {"0":"H","1":"L"}
            params["onehot_pool"] = 'false'
            sf.util.write_json(params, join(m, "params_eval.json"))
            SFP.evaluate(m, outcome_label_headers = odx_val_name, filters={'CV3_odx85_mip':str(i)}, save_predictions=True, model_config=join(m, "params_eval.json"))

        try:
            find_eval(SFP, exp_label_mp, odx_train, hp.epochs[0], i)
            print("MammaPrint Predictions Exist for Crossfold " + str(i) + " - Skipping Evaluation")
            print("Change Experiment Name or Delete Results Folder from /PROJECTS/UCH_RS/eval/ to Repeat Evaluation")
        except:
            print("MammaPrint Predictions Do Not Exist for Crossfold " + str(i) + " - Evaluating Crossfold")
            m = find_model(SFP, exp_label_mp, outcome=mp_train_name, epoch=hp.epochs[0], kfold=i)
            params = sf.util.get_model_config(m)
            params["hp"]["loss"] = "sparse_categorical_crossentropy"
            params["model_type"] = "categorical"
            params["outcome_labels"] = {"0":"H","1":"L"}
            params["onehot_pool"] = 'false'
            sf.util.write_json(params, join(m, "params_eval.json"))
            SFP.evaluate(m, outcome_label_headers = mp_val_name, filters={'CV3_mp85_mip':str(i)}, save_predictions=True, model_config=join(m, "params_eval.json"))

def test_models(hpsearch = 'old', prefix_hpopt = 'hp_new2', start = 1, max_count = 50, train_receptors = False, use_filtered = False, train_reverse = False, annotation = None, source = None):
    """Validates OncotypeDx and MammaPrint prediction models in UCMC

    Parameters
    ----------
    hpsearch - specifies whether to load existing hyperparameters ('old'), or scan for best hyperparameters from new hyperparameter search ('read')
    prefix_hpopt - specifies the prefix for hyperparameter training when reading model results from new hyperparameter search
    start - specifies the starting iteration when reading from a new hyperparameter search
    max_count - specifies the final iteration when reading from a new hyperparameter search
    train_receptors - if True, validates models trained on HR+/HER2- patients instead of all of TCGA
    use_filtered - if True, validates models trained on the TCGA_BRCA_FILTERED dataset (filtered to exclude tiles with a < 50% likelihood of being tumor)
    train_reverse - if True, validation is performed on TCGA instead of UCMC
    annotation - if provided, will extract tiles for a custom validation dataset, where annotations are in the /UCH_RS/<annotation> csv file
    source - if provided, will extract tiles for a custom validation dataset of the name provided
    
    """   
    global odx_train, mp_train

    #Finally, validate on external dataset
    SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS"))
    if train_reverse:
        SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "tcga_brca_complete.csv")
        SFP.sources = ["TCGA_BRCA_NO_ROI"]
    else:
        SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "uch_brca_complete.csv")
        SFP.sources = ["UCH_BRCA_RS"]
    if annotation:
        SFP.annotations = join(PROJECT_ROOT, "UCH_RS", annotation)
    if source:
        SFP.sources = [source]

    SFP_TUMOR_ROI = sf.Project(join(PROJECT_ROOT, "TCGA_BRCA_ROI"))
    SFP_TUMOR_ROI.annotations = join(PROJECT_ROOT, "TCGA_BRCA_ROI", "tumor_roi.csv")

    odx_train_name = odx_train
    odx_val_name = "RSHigh"
    mp_train_name = mp_train
    mp_val_name = "MPHigh"
    

    if hpsearch == 'old':
        hp = hp_opt
    elif hpsearch == 'read':
        hp = get_best_model_params(SFP, prefix = prefix_hpopt, start = start, max_count = max_count *2)

    hp.weight_model = assign_tumor_roi_model(SFP_TUMOR_ROI, hp.tile_px, hp.normalizer)
    exp_additions = ""
    if train_receptors:
        exp_additions += "_TR"
    if use_filtered:
        exp_additions += "_UF"
    if train_reverse:
        exp_additions += "_REV"
        odx_train_name = "RS"
        odx_val_name = "odx85"
        mp_train_name = "mpscore"
        mp_val_name = "mphr"
    global exp_suffix
    exp_label = "ODX_" + exp_suffix + exp_additions
    exp_label_mp = "MP_" + exp_suffix + exp_additions

    try:
        find_eval(SFP, exp_label, odx_train_name, hp.epochs[0])
        print("Oncotype DX Predictions Exist for External Dataset - Skipping Evaluation")
        print("Change Experiment Name or Delete Results Folder from /PROJECTS/UCH_RS/eval/ to Repeat Evaluation")
    except:
        print("Oncotype DX Predictions Do Not Exist for External Dataset - Evaluating Dataset")
        m = find_model(SFP, exp_label, outcome = odx_train_name, epoch=hp.epochs[0])
        params = sf.util.get_model_config(m)
        params["hp"]["loss"] = "sparse_categorical_crossentropy"
        params["model_type"] = "categorical"
        params["outcome_labels"] = {"0":"H","1":"L"}
        params["onehot_pool"] = 'false'
        sf.util.write_json(params, join(m, "params_eval.json"))
        SFP.evaluate(model=m, outcome_label_headers=odx_val_name, save_predictions=True, model_config=join(m, "params_eval.json"))
    try:
        find_eval(SFP, exp_labelmp, mp_train_name, hp.epochs[0])
        print("MammaPrint Predictions Exist for External Dataset - Skipping Evaluation")
        print("Change Experiment Name or Delete Results Folder from /PROJECTS/UCH_RS/eval/ to Repeat Evaluation")
    except:
        print("MammaPrint Predictions Do Not Exist for External Dataset - Evaluating Dataset")
        m = find_model(SFP, exp_label_mp, outcome = mp_train_name, epoch=hp.epochs[0])
        params = sf.util.get_model_config(m)
        params["hp"]["loss"] = "sparse_categorical_crossentropy"
        params["model_type"] = "categorical"
        params["outcome_labels"] = {"0":"H","1":"L"}
        params["onehot_pool"] = 'false'
        sf.util.write_json(params, join(m, "params_eval.json"))
        SFP.evaluate(model=m, outcome_label_headers=mp_val_name, save_predictions=True, model_config=join(m, "params_eval.json"))

    
def main(): 
    global exp_suffix
    parser = argparse.ArgumentParser(description = "Helper to guide through model training.")

    parser.add_argument('-p', '--project_root', required=False, type=str, help='Path to project directory (if not provided, assumes subdirectory of this script).')
    parser.add_argument('--hpsearch', required=False, type=str, help='Set to \'old\' to use saved hyperparameters, \'run\' to perform hyperparameter search, \'read\' to find best hyperparameters from prior run.')
    parser.add_argument('-e', '--extract', required=False, action="store_true", help='If provided, will extract tiles from slide directory.')
    parser.add_argument('-t', '--train', required=False, action="store_true", help='If provided, will train models in TCGA for tumor ROI / recurrnece score prediction.')
    parser.add_argument('-v', '--validate', required=False, action="store_true", help='If provided, will validate models in the University of Chicago dataset.')
    parser.add_argument('-a', '--annotation', required=False, type=str, help='Can be used to select an alternate annotation file for extraction / model testing')
    parser.add_argument('-s', '--source', required=False, type=str, help='Can be used to select an alternate dataset source for extraction / model testing')
    parser.add_argument('-exp', '--experiment_label', required=False, type=str, help='Can be used to set a unique experiment name for training / validation. Defaults to Final_BRCAROI')

    parser.add_argument('-uf', '--use_filtered', action="store_true", help='If provided, will load predictions generated from filtered TCGA-BRCA slides (instead of annotated slides)')
    parser.add_argument('-tr', '--train_receptors', action="store_true", help='If provided, will load predictions generated when training on only HR+/HER2- TCGA-BRCA slides (instead of entire dataset)')
    parser.add_argument('-rev', '--train_reverse', action="store_true", help='If provided, will run predictions trained on UCMC and validated on TCGA')

    parser.add_argument('--hpprefix', required=False, type=str, help='Provide prefix for models trained during hyperparameter search (must be specified if desired to read hyperparameters from old HP search).')
    parser.add_argument('--hpstart', required=False, type=int, help='Provide the starting index for models to check among prior hyperparameter search.')
    parser.add_argument('--hpcount', required=False, type=int, help='Provide the number of models to check among prior hyperparameter search.')
    parser.add_argument('--heatmaps_tumor_roi', required=False, type=str, help='Will save heatmaps for tumor region of interest model. Set to TCGA to provide heatmaps from TCGA and UCH to provide heatmaps from validation dataset.')
    parser.add_argument('--heatmaps_odx', required=False, type=str, help='Will save heatmaps for OncotypeDx model. Set to TCGA to provide heatmaps from TCGA and UCH to provide heatmaps from validation dataset.')

    args = parser.parse_args()  
    global PROJECT_ROOT, odx_train, mp_train, exp_suffix
    if args.project_root:
        PROJECT_ROOT = args.project_root
    if not args.hpsearch:
        args.hpsearch = 'old'
    if not args.hpprefix:
        args.hpprefix = 'hp_optimize'
    if not args.hpstart:
        args.hpstart = 1
    if not args.hpcount:
        args.hpcount = 50
    if args.experiment_label:
        exp_suffix = args.experiment_label
    annotation = None
    if args.annotation:
        annotation = args.annotation
    source = None
    if args.source:
        source = args.source
    if args.extract:
        setup_projects(annotation = annotation, source = source)
    if args.hpsearch == 'run':
        SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS"))
        SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "tcga_brca_complete.csv")
        SFP.sources = ["TCGA_BRCA_FULL_ROI"]
        hyperparameter_optimization(SFP, args.hpprefix, args.hpcount)
        args.hpsearch = 'read'
    if args.train:
        train_models(args.hpsearch, args.hpprefix, args.hpstart, args.hpcount, train_receptors = args.train_receptors, use_filtered = args.use_filtered, train_reverse = args.train_reverse)
    if args.validate:
        test_models(hpsearch = args.hpsearch, prefix_hpopt =  args.hpprefix, start = args.hpstart, max_count = args.hpcount, train_receptors = args.train_receptors, use_filtered = args.use_filtered, train_reverse = args.train_reverse, annotation = annotation, source = source)


    if args.heatmaps_odx:
        if args.heatmaps_odx == 'TCGA':
            SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS"))
            exp_label = "ODX_" + exp_suffix 
            SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "tcga_brca_complete.csv")            
            SFP.sources = ["TCGA_BRCA_NO_ROI"]
            for i in [1,2,3]:
                m = find_model(SFP, exp_label, outcome=odx_train, epoch=1, kfold=i)
                SFP.generate_heatmaps(model=m, filters={'CV3_odx85_mip':str(i)}, outdir = join(PROJECT_ROOT, 'UCH_RS/heatmaps_tcga_roi'), resolution='low', batch_size=32, roi_method ='none', show_roi = True, buffer=join(PROJECT_ROOT, 'buffer'))
        if args.heatmaps_odx == 'UCH':
            print("HEATMAPS ODX UCH")
            SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS"))
            SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "uch_brca_complete.csv")
            SFP.sources = ["UCH_BRCA_RS"]
            exp_label = "ODX_" + exp_suffix
            m = find_model(SFP, exp_label, outcome=odx_train, epoch=1)
            SFP.generate_heatmaps(model=m, outdir = join(PROJECT_ROOT, 'UCH_RS/heatmaps_uch_roi'), resolution='low', batch_size=32, roi_method ='none', show_roi = True)
    if args.heatmaps_tumor_roi:
        hp = None
        if args.hpsearch == 'old':
            hp = hp_opt
        elif args.hpsearch == 'read':
            SFP = sf.Project(join(PROJECT_ROOT, "UCH_RS"))
            SFP.annotations = join(PROJECT_ROOT, "UCH_RS", "tcga_brca_complete.csv")
            SFP.sources = ["TCGA_BRCA_FULL_ROI"]
            hp = get_best_model_params(SFP, args.hpprefix, args.hpstart, 2*args.hpcount)
        normalizer = hp.normalizer
        tile_px = hp.tile_px
        if args.heatmaps_tumor_roi == 'TCGA':
            SFP = sf.Project(join(PROJECT_ROOT, "TCGA_BRCA_ROI")) 
            SFP.sources = ["TCGA_BRCA_NO_ROI"]
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
            m = find_model(SFP, exp_label, outcome='roi', epoch=1)
            SFP.generate_heatmaps(model=m, outdir = join(PROJECT_ROOT, 'TCGA_BRCA_ROI/heatmaps_uch'), resolution='low', batch_size=32, roi_method ='none', show_roi = True, buffer=join(PROJECT_ROOT, 'buffer'))

if __name__ == '__main__':
    main()