import os
import sys
import csv
import types
import time
import pickle
import seaborn as sns
import numpy as np
import pandas as pd
import multiprocessing as mp
import slideflow as sf

from tqdm import tqdm
from functools import partial
from os.path import join
from slideflow.util import log, ProgressBar, to_onehot
from slideflow import errors
from scipy import stats
from scipy.special import softmax
from random import sample
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib.widgets import LassoSelector
from lifelines.utils import concordance_index as c_index

# TODO: remove 'hidden_0' reference as this may not be present if the model does not have hidden layers
# TODO: convert all this x /y /meta /values stuff to just a pandas dataframe?

class SlideMap:
    """Two-dimensional slide map used for visualization, as well as subsequent construction of mosaic maps.

    Slides are mapped in 2D either explicitly with pre-specified coordinates, or with dimensionality reduction
    from post-convolutional layer weights, provided from :class:`slideflow.model.DatasetFeatures`.

    """

    def __init__(self, slides, cache=None):
        """Backend for mapping slides into two dimensional space. Can use an DatasetFeatures object
        to map slides according to UMAP of features, or map according to pre-specified coordinates.

        Args:
            slides (list(str)): List of slide names
            cache (str, optional): Path to PKL file to cache activations. Defaults to None (caching disabled).
        """

        self.slides = slides
        self.cache = cache
        self.x = []
        self.y = []
        self.point_meta = []
        self.labels = []
        self.map_meta = {}
        if self.cache: self.load_cache() # Try to load from cache

    @classmethod
    def from_precalculated(cls, slides, x, y, meta, labels=None, cache=None):
        """Initializes map from precalculated coordinates.

        Args:
            slides (list(str)): List of slide names.
            x (list(int)): List of X coordinates for tfrecords.
            y (list(int)): List of Y coordinates for tfrecords.
            meta (list(dict)): List of dicts containing metadata for each point on the map (representing a single tfrecord).
            labels (list(str)): Labels assigned to each tfrecord, used for coloring TFRecords according to labels.
            cache (str, optional): Path to PKL file to cache coordinates. Defaults to None (caching disabled).
        """

        obj = cls(slides)
        obj.x = np.array(x) if type(x) == list else x
        obj.y = np.array(y) if type(y) == list else y
        obj.point_meta = np.array(meta) if type(meta) == list else meta
        obj.cache = cache
        obj.labels = np.array(labels) if type(labels) == list else labels
        if obj.labels == []:
            obj.labels = np.array(['None' for i in range(len(obj.point_meta))])
        obj.save_cache()
        return obj

    @classmethod
    def from_features(cls, df, exclude_slides=None, prediction_filter=None, recalculate=False,
                         map_slide=None, cache=None, low_memory=False, umap_dim=2):
        """Initializes map from dataset features.

        Args:
            df (:class:`slideflow.model.DatasetFeatures`): DatasetFeatures object.
            exclude_slides (list, optional): List of slides to exclude from map.
            prediction_filter (list, optional) Restrict outcome predictions to only these provided categories.
            recalculate (bool, optional):  Force recalculation of umap despite presence of cache.
            use_centroid (bool, optional): Calculate and map centroid activations.
            map_slide (str, optional): Either None (default), 'centroid', or 'average'.
                If None, will map all tiles from each slide.
            cache (str, optional): Path to PKL file to cache coordinates. Defaults to None (caching disabled).
        """

        if map_slide is not None and map_slide not in ('centroid', 'average'):
            raise errors.SlideMapError(f"map_slide must be None (default), 'centroid', or 'average', not '{map_slide}'")

        if not exclude_slides:
            slides = df.slides
        else:
            slides = [slide for slide in df.slides if slide not in exclude_slides]

        obj = cls(slides, cache=cache)
        obj.df = df
        if map_slide:
            obj._calculate_from_slides(method=map_slide,
                                       prediction_filter=prediction_filter,
                                       recalculate=recalculate,
                                       low_memory=low_memory)
        else:
            obj._calculate_from_tiles(prediction_filter=prediction_filter,
                                      recalculate=recalculate,
                                      low_memory=low_memory,
                                      dim=umap_dim)
        return obj

    def _calculate_from_tiles(self, prediction_filter=None, recalculate=False, **umap_kwargs):

        """Internal function to guide calculation of UMAP from final layer features / activations,
        as provided by DatasetFeatures.

        Args:
            prediction_filter (list, optional): Restrict predictions to this list of logits. Default is None.
            recalculate (bool, optional): Recalculate of UMAP despite loading from cache. Defaults to False.

        Keyword Args:
            dim (int): Number of dimensions for UMAP. Defaults to 2.
            n_neighbors (int): Number of neighbors for UMAP. Defaults to 50.
            min_dist (float): Minimum distance for UMAP. Defaults to 0.1.
            metric (str): UMAP metric. Defaults to 'cosine'.
            low_memory (bool). Operate UMAP in low memory mode. Defaults to False.
        """

        if prediction_filter:
            log.info("UMAP logit predictions are masked through a provided prediction filter.")

        if len(self.x) and len(self.y) and not recalculate:
            log.info("UMAP loaded from cache, will not recalculate")

            # First, filter out slides not included in provided activations
            if not isinstance(self.point_meta, np.ndarray):
                self.point_meta = np.array(self.point_meta)
            filtered_idx = np.array([i for i, x in enumerate(self.point_meta) if x['slide'] in self.df.slides])
            self.x = self.x[filtered_idx]
            self.y = self.y[filtered_idx]
            self.point_meta = self.point_meta[filtered_idx]

            # If UMAP already calculated, update predictions if prediction filter is provided
            for i in range(len(self.point_meta)):
                slide = self.point_meta[i]['slide']
                tile_index = self.point_meta[i]['index']
                logits = self.df.logits[slide][tile_index]
                prediction = filtered_prediction(logits, prediction_filter)
                self.point_meta[i]['logits'] = logits
                self.point_meta[i]['prediction'] = prediction
            return

        # Calculate UMAP
        node_activations = np.concatenate([self.df.activations[slide] for slide in self.slides])
        self.map_meta['num_features'] = self.df.num_features
        log.info("Calculating UMAP...")
        for slide in self.slides:
            for i in range(self.df.activations[slide].shape[0]):
                location = self.df.locations[slide][i]
                logits = self.df.logits[slide][i]
                pred = filtered_prediction(logits, prediction_filter) if self.df.logits[slide] != [] else None
                pm = {
                    'slide': slide,
                    'index': i,
                    'prediction': pred,
                    'logits': logits,
                    'loc': location
                }
                if self.df.hp.uq:
                    pm.update({'uncertainty': self.df.uncertainty[slide][i]})
                self.point_meta += [pm]

        coordinates = gen_umap(node_activations, **umap_kwargs)

        self.x = np.array([c[0] for c in coordinates])
        if umap_kwargs['dim'] > 1:
            self.y = np.array([c[1] for c in coordinates])
        else:
            self.y = np.array([0 for i in range(len(self.x))])

        assert self.x.shape[0] == self.y.shape[0]
        assert self.x.shape[0] == len(self.point_meta)

        self.save_cache()

    def _calculate_from_slides(self, method='centroid', prediction_filter=None, recalculate=False, **umap_kwargs):

        """ Internal function to guide calculation of UMAP from final layer activations for each tile,
            as provided via DatasetFeatures, and then map only the centroid tile for each slide.

        Args:
            method (str, optional): Either 'centroid' or 'average'. If centroid, will calculate UMAP only
                from centroid tiles for each slide. If average, will calculate UMAP based on average node
                activations across all tiles within the slide, then display the centroid tile for each slide.
            prediction_filter (list, optional): List of int. If provided, will restrict predictions to these categories.
            recalculate (bool, optional): Recalculate of UMAP despite loading from cache. Defaults to False.
            low_memory (bool, optional): Calculate UMAP in low-memory mode. Defaults to False.

        Keyword Args:
            dim (int): Number of dimensions for UMAP. Defaults to 2.
            n_neighbors (int): Number of neighbors for UMAP. Defaults to 50.
            min_dist (float): Minimum distance for UMAP. Defaults to 0.1.
            metric (str): UMAP metric. Defaults to 'cosine'.
            low_memory (bool). Operate UMAP in low memory mode. Defaults to False.
        """

        if method not in ('centroid', 'average'):
            raise errors.SlideMapError(f'Method must be either "centroid" or "average", not {method}')

        log.info("Calculating centroid indices...")
        optimal_slide_indices, centroid_activations = calculate_centroid(self.df.activations)

        # Restrict mosaic to only slides that had enough tiles to calculate an optimal index from centroid
        successful_slides = list(optimal_slide_indices.keys())
        num_warned = 0
        for slide in self.df.slides:
            if slide not in successful_slides:
                log.debug(f"Unable to calculate centroid for {sf.util.green(slide)}; will not include")
        if num_warned:
            log.warning(f"Unable to calculate centroid for {num_warned} slides.")

        if len(self.x) and len(self.y) and not recalculate:
            log.info("UMAP loaded from cache, will filter to include only provided tiles")
            new_x, new_y, new_meta = [], [], []
            for i in range(len(self.point_meta)):
                slide = self.point_meta[i]['slide']
                if slide in optimal_slide_indices and self.point_meta[i]['index'] == optimal_slide_indices[slide]:
                    new_x += [self.x[i]]
                    new_y += [self.y[i]]
                    if prediction_filter:
                        tile_index = self.point_meta[i]['index']
                        logits = self.df.logits[slide][tile_index]
                        prediction = filtered_prediction(logits, prediction_filter)
                        meta = {
                            'slide': slide,
                            'index': tile_index,
                            'logits': logits,
                            'prediction': prediction,
                        }
                    else:
                        meta = self.point_meta[i]
                    new_meta += [meta]
            self.x = np.array(new_x)
            self.y = np.array(new_y)
            self.point_meta = np.array(new_meta)
        else:
            log.info(f"Calculating UMAP from slide-level {method}...")
            umap_input = []
            for slide in self.slides:
                if method == 'centroid':
                    umap_input += [centroid_activations[slide]]
                elif method == 'average':
                    activation_averages = np.mean(self.df.activations[slide])
                    umap_input += [activation_averages]
                self.point_meta += [{
                    'slide': slide,
                    'index': optimal_slide_indices[slide],
                    'logits': [],
                    'prediction': 0
                }]

            coordinates = gen_umap(np.array(umap_input), **umap_kwargs)
            self.x = np.array([c[0] for c in coordinates])
            self.y = np.array([c[1] for c in coordinates])
            self.save_cache()

    def cluster(self, n_clusters):
        """Performs clustering on data and adds to metadata labels. Requires a DatasetFeatures backend.

        Clusters are saved to self.point_meta[i]['cluster'].

        Args:
            n_clusters (int): Number of clusters for K means clustering.

        Returns:
            ndarray: Array with cluster labels corresponding to tiles in self.point_meta.
        """

        activations = [self.df.activations[pm['slide']][pm['index']] for pm in self.point_meta]
        log.info(f"Calculating K-means clustering (n={n_clusters})")
        kmeans = KMeans(n_clusters=n_clusters).fit(activations)
        labels = kmeans.labels_
        for i, label in enumerate(labels):
            self.point_meta[i]['cluster'] = label
        return np.array([p['cluster'] for p in self.point_meta])

    def export_to_csv(self, filename):
        """Exports calculated UMAP coordinates in csv format.

        Args:
            filename (str): Path to CSV file in which to save coordinates.

        """

        with open(filename, 'w') as outfile:
            csvwriter = csv.writer(outfile)
            header = ['slide', 'index', 'x', 'y']
            csvwriter.writerow(header)
            for index in range(len(self.point_meta)):
                x = self.x[index]
                y = self.y[index]
                meta = self.point_meta[index]
                slide = meta['slide']
                index = meta['index']
                row = [slide, index, x, y]
                csvwriter.writerow(row)

    def neighbors(self, slide_categories=None, algorithm='kd_tree'):
        """Calculates neighbors among tiles in this map, assigning neighboring statistics
            to tile metadata 'num_unique_neighbors' and 'percent_matching_categories'.

        Args:
            slide_categories (dict, optional): Dict mapping slides to categories. Defaults to None.
                If provided, will be used to calculate 'percent_matching_categories' statistic.
            algorithm (str, optional): NearestNeighbor algorithm, either 'kd_tree', 'ball_tree', or 'brute'.
                Defaults to 'kd_tree'.
        """

        from sklearn.neighbors import NearestNeighbors
        log.info("Initializing neighbor search...")
        X = np.array([self.df.activations[pm['slide']][pm['index']] for pm in self.point_meta])
        nbrs = NearestNeighbors(n_neighbors=100, algorithm=algorithm, n_jobs=-1).fit(X)
        log.info("Calculating nearest neighbors...")
        _, indices = nbrs.kneighbors(X)
        for i, ind in enumerate(indices):
            num_unique_slides = len(list(set([self.point_meta[_i]['slide'] for _i in ind])))
            self.point_meta[i]['num_unique_neighbors'] = num_unique_slides
            if slide_categories:
                matching_categories = [_i for _i in ind if slide_categories[self.point_meta[_i]['slide']] == \
                                                           slide_categories[self.point_meta[i]['slide']]]
                percent_matching_categories = len(matching_categories)/len(ind)
                self.point_meta[i]['percent_matching_categories'] = percent_matching_categories

    def filter(self, slides):
        """Filters map to only show tiles from the given slides.

        Args:
            slides (list(str)): List of slide names.
        """

        if not hasattr(self, 'full_x'):
            # Backup full coordinates
            self.full_x, self.full_y, self.full_meta = self.x, self.y, self.point_meta
        else:
            # Restore backed up full coordinates
            self.x, self.y, self.point_meta = self.full_x, self.full_y, self.full_meta

        self.x = np.array([self.x[xi] for xi in range(len(self.x)) if self.point_meta[xi]['slide'] in slides])
        self.y = np.array([self.y[yi] for yi in range(len(self.y)) if self.point_meta[yi]['slide'] in slides])
        self.point_meta = np.array([pm for pm in self.point_meta if pm['slide'] in slides])

    def show_neighbors(self, neighbor_df, slide):
        """Filters map to only show neighbors with a corresponding neighbor DatasetFeatures and neighbor slide.

        Args:
            neighbor_df (:class:`slideflow.model.DatasetFeatures`): DatasetFeatures object
                containing activations for neighboring slide.
            slide (str): Name of neighboring slide.
        """

        if slide not in neighbor_df.activations:
            raise errors.SlideMapError(f"Slide {slide} not found in DatasetFeatures, unable to find neighbors")
        if not hasattr(self, 'df'):
            raise errors.SlideMapError(f"SlideMap does not have an DatasetFeatures, unable to calculate neighbors")

        tile_neighbors = self.df.neighbors(neighbor_df, slide, n_neighbors=5)

        if not hasattr(self, 'full_x'):
            # Backup full coordinates
            self.full_x, self.full_y, self.full_meta = self.x, self.y, self.point_meta
        else:
            # Restore backed up full coordinates
            self.x, self.y, self.point_meta = self.full_x, self.full_y, self.full_meta

        def filter_by_neighbors(arr):
            tn = tile_neighbors
            f_arr = [v for i,v in enumerate(arr) if (self.point_meta[i]['slide'] in tn and
                                                     self.point_meta[i]['index'] in tn[self.point_meta[i]['slide]']])]
            return np.array(f_arr)

        self.x = filter_by_neighbors(self.x)
        self.y = filter_by_neighbors(self.y)
        self.meta = filter_by_neighbors(self.meta)

    def label_by_uncertainty(self, index=0):
        """Labels each point with the tile-level uncertainty, if available.

        Args:
            index (int, optional): Uncertainty index. Defaults to 0.
        """

        if not self.df.hp.uq:
            raise errors.DatasetError('Unable to label by uncertainty; UQ estimations not available.')
        else:
            self.labels = np.array([m['uncertainty'][index] for m in self.point_meta])

    def label_by_logits(self, index):
        """Displays each point with label equal to the logits (linear from 0-1)

        Args:
            index (int): Logit index.
        """

        self.labels = np.array([m['logits'][index] for m in self.point_meta])

    def label_by_slide(self, slide_labels=None):
        """Displays each point as the name of the corresponding slide.
            If slide_labels is provided, will use this dictionary to label slides.

        Args:
            slide_labels (dict, optional): Dict mapping slide names to labels.
        """

        if slide_labels:
            self.labels = np.array([slide_labels[m['slide']] for m in self.point_meta])
        else:
            self.labels = np.array([m['slide'] for m in self.point_meta])

    def label_by_meta(self, tile_meta, translation_dict=None):
        """Displays each point with label equal a value in tile metadata (e.g. 'prediction')

        Args:
            tile_meta (str): Key to metadata from which to read
            translation_dict (dict, optional): If provided, will translate the read metadata through this dictionary.
        """

        if translation_dict:
            try:
                self.labels = np.array([translation_dict[m[tile_meta]] for m in self.point_meta])
            except KeyError:
                # Try by converting metadata to string
                self.labels = np.array([translation_dict[str(m[tile_meta])] for m in self.point_meta])
        else:
            self.labels = np.array([m[tile_meta] for m in self.point_meta])

    def save_2d_plot(self, *args, **kwargs):
        """Deprecated function; please use `save`."""

        log.warning("save_2d_plot() is deprecated, please use save()")
        self.save(*args, **kwargs)

    def save(self, filename, subsample=None, title=None, cmap=None, xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), xlabel=None,
             ylabel=None, legend=None, dpi=300, **scatter_kwargs):

        """Saves plot of data to a provided filename.

        Args:
            filename (str): File path to save the image.
            subsample (int, optional): Subsample to only include this many tiles on plot. Defaults to None.
            title (str, optional): Title for plot.
            cmap (dict, optional): Dict mapping labels to colors.
            xlim (list, optional): List of float indicating limit for x-axis. Defaults to (-0.05, 1.05).
            ylim (list, optional): List of float indicating limit for y-axis. Defaults to (-0.05, 1.05).
            xlabel (str, optional): Label for x axis. Defaults to None.
            ylabel (str, optional): Label for y axis. Defaults to None.
            legend (str, optional): Title for legend. Defaults to None.
            dpi (int, optional): DPI for final image. Defaults to 300.
        """

        # Subsampling
        if subsample:
            ri = sample(range(len(self.x)), min(len(self.x), subsample))
        else:
            ri = list(range(len(self.x)))

        df = pd.DataFrame()
        x = self.x[ri]
        y = self.y[ri]
        df['umap_x'] = x
        df['umap_y'] = y

        if len(self.labels):
            labels = self.labels[ri]

            # Check for categorical labels
            if not np.issubdtype(self.labels.dtype, np.floating):
                log.debug("Interpreting labels as categorical")
                df['category'] = pd.Series(labels, dtype='category')
                unique_categories = list(set(labels))
                unique_categories.sort()
                if len(unique_categories) >= 12:
                    seaborn_palette = sns.color_palette("Paired", len(unique_categories))
                else:
                    seaborn_palette = sns.color_palette('hls', len(unique_categories))
                if cmap is None:
                    cmap = {unique_categories[i]:seaborn_palette[i] for i in range(len(unique_categories))}
            else:
                log.debug("Interpreting labels as continuous")
                df['category'] = labels
        else:
            labels = ['NA']
            df['category'] = 'NA'

        # Make plot
        plt.clf()
        umap_2d = sns.scatterplot(x=x, y=y, data=df, hue='category', palette=cmap, **scatter_kwargs)
        plt.gca().set_ylim(*((None, None) if not ylim else ylim))
        plt.gca().set_xlim(*((None, None) if not xlim else xlim))
        umap_2d.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1, title=legend)
        umap_2d.set(xlabel=xlabel, ylabel=ylabel)
        umap_figure = umap_2d.get_figure()
        umap_figure.set_size_inches(6, 4.5)
        if title: umap_figure.axes[0].set_title(title)
        umap_figure.savefig(filename, bbox_inches='tight', dpi=dpi)
        log.info(f"Saved 2D UMAP to {sf.util.green(filename)}")

    def save_3d_plot(self, filename, z=None, feature=None, subsample=None):
        """Saves a plot of a 3D umap, with the 3rd dimension representing values provided by argument "z".

        Args:
            filename (str): Filename to save image of plot.
            z (list, optional): Values for z axis. Must supply z or feature. Defaults to None.
            feature (int, optional): Int, feature to plot on 3rd axis. Must supply z or feature. Defaults to None.
            subsample (int, optional): Subsample to only include this many tiles on plot. Defaults to None.
        """

        title = f"UMAP with feature {feature} focus"

        if not filename:
            filename = "3d_plot.png"

        if (z is None) and (feature is None):
            raise errors.SlideMapError("Must supply either 'z' or 'feature'.")

        # Get feature activations for 3rd dimension
        if z is None:
            z = np.array([self.df.activations[m['slide']][m['index']][feature] for m in self.point_meta])

        # Subsampling
        if subsample:
            ri = sample(range(len(self.x)), min(len(self.x), subsample))
        else:
            ri = list(range(len(self.x)))

        x = self.x[ri]
        y = self.y[ri]
        z = z[ri]

        # Plot tiles on a 3D coordinate space with 2 coordinates from UMAP & 3rd from the value of the excluded feature
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x, y, z, c=z,
                            cmap='viridis',
                            linewidth=0.5,
                            edgecolor="black")
        ax.set_title(title)
        log.info(f"Saving 3D UMAP to {sf.util.green(filename)}...")
        plt.savefig(filename, bbox_inches='tight')

    def get_tiles_in_area(self, x_lower=-999, x_upper=999, y_lower=-999, y_upper=999):
        """Returns dictionary of slide names mapping to tile indices,
            or tiles that fall within the specified location on the umap.

        Args:
            x_lower (int, optional): X-axis lower limit. Defaults to -999.
            x_upper (int, optional): X-axis upper limit. Defaults to 999.
            y_lower (int, optional): Y-axis lower limit. Defaults to -999.
            y_upper (int, optional): Y-axis upper limit. Defaults to 999.

        Returns:
            dict: Dict mapping slide names to tile indices, for tiles included in the given area.
        """

        # Find tiles that meet UMAP location criteria
        filtered_tiles = {}
        num_selected = 0
        for i in range(len(self.point_meta)):
            if (x_lower < self.x[i] < x_upper) and (y_lower < self.y[i] < y_upper):
                slide = self.point_meta[i]['slide']
                tile_index = self.point_meta[i]['index']
                if slide not in filtered_tiles:
                    filtered_tiles.update({slide: [tile_index]})
                else:
                    filtered_tiles[slide] += [tile_index]
                num_selected += 1
        log.info(f"Selected {num_selected} tiles by filter criteria.")
        return filtered_tiles

    def save_cache(self):
        """Save cache of coordinates to PKL file."""
        if self.cache:
            try:
                with open(self.cache, 'wb') as cache_file:
                    pickle.dump([self.x, self.y, self.point_meta, self.map_meta], cache_file)
                    log.info(f"Wrote UMAP cache to {sf.util.green(self.cache)}")
            except:
                log.info(f"Error attempting to write UMAP cache to {sf.util.green(self.cache)}")

    def load_cache(self):
        """Load coordinates from PKL cache."""
        try:
            with open(self.cache, 'rb') as cache_file:
                self.x, self.y, self.point_meta, self.map_meta = pickle.load(cache_file)
                log.info(f"Loaded UMAP cache from {sf.util.green(self.cache)}")
                return True
        except FileNotFoundError:
            log.info(f"No UMAP cache found at {sf.util.green(self.cache)}")
        return False

def _generate_tile_roc(i, y_true, y_pred, data_dir, label_start, histogram=False, neptune_run=None):
    """Generates tile-level ROC. Defined as a separate function for use with multiprocessing."""
    try:
        auc, ap, thresh = generate_roc(y_true[:, i], y_pred[:, i], data_dir, f'{label_start}tile_ROC{i}', neptune_run)
        if histogram:
            save_histogram(y_true[:, i], y_pred[:, i], data_dir, f'{label_start}tile_histogram{i}', neptune_run)
    except IndexError:
        log.warning(f"Unable to generate tile-level stats for outcome {i}")
        return None, None, None
    return auc, ap, thresh # ROC AUC, Average Precision, Optimal Threshold

def _get_average_by_group(prediction_array, prediction_label, unique_groups, tile_to_group, y_true_group,
                            num_cat, label_end, uncertainty=None, save_predictions=False, data_dir=None, label='group', weights=None, filters=None, filter_thresh=None):

    """Internal function to generate group-level averages (e.g. slide-level or patient-level).

    For a given tile-level prediction array, calculate spercent predictions
    in each outcome by group (e.g. patient, slide), and saves to CSV if specified.
    """

    groups = {g:[] for g in unique_groups}
    group_weights = {g:[] for g in unique_groups}
    group_uncertainty = {g:[] for g in unique_groups}
    group_filters = {g:[] for g in unique_groups}
    group_percents = {g:[] for g in unique_groups}
    def update_group(ttg):
        i, g = ttg
        groups[g] += [prediction_array[i]]
        if uncertainty is not None:
            group_uncertainty[g] += [uncertainty[i]]
        if weights is not None:
            group_weights[g] += [[weights[i][1], weights[i][1]]]
        if filters is not None:
            group_filters[g] += [filters[i]]

    with mp.dummy.Pool(processes=16) as p:
        p.map(update_group, enumerate(tile_to_group))

    group_uncertainty = {g:np.array(u) for g,u in group_uncertainty.items()}

    if uncertainty is not None:
        #log.info(f'Using uncertainty weighting for {label}-level predictions')
        #try:
        #    group_percents = {g:np.average(np.array(groups[g]), weights=(((group_uncertainty[g] - group_uncertainty[g].max()) * -1) + group_uncertainty[g].min() + 0.1)**2, axis=0) for g in unique_groups}
        #except:
        #    log.error("Error with weighted predictions; calculation without weighting")
        #    group_percents = {g:np.average(np.array(groups[g]), axis=0) for g in unique_groups}
        group_percents = {g:np.average(np.array(groups[g]), axis=0) for g in unique_groups}
        uncertainty_by_group = [np.array(group_uncertainty[g]).mean(axis=0) for g in unique_groups]
    elif weights is not None:
        group_percents = {g:np.divide(np.sum(np.multiply(np.array(groups[g]), np.array(group_weights[g])), axis = 0), np.sum(np.array(group_weights[g]), axis = 0)) for g in unique_groups}
    elif filters is not None:
        for g in unique_groups:
            count = len(np.array(group_filters[g]))
            if count <= 20:
                group_percents[g] = np.array(groups[g]).mean(axis=0)
            else:
                filter_cutoff = np.sort(group_filters[g])[20]
                if filter_cutoff < filter_thresh:
                    filter_cutoff = filter_thresh
                group_percents[g] = np.average(np.array(groups[g]), axis = 0, weights = np.ndarray.flatten((np.array(group_filters[g]) < filter_cutoff)))
    else:
        group_percents = {g:np.array(groups[g]).mean(axis=0) for g in unique_groups}
    avg_by_group = np.array([group_percents[g] for g in unique_groups])

    # --- Save predictions to CSV ---------------------------------------------
    if save_predictions:
        save_path = join(data_dir, f"{label}_predictions{label_end}.csv")
        with open(save_path, 'w') as outfile:
            writer = csv.writer(outfile)
            header = [label] + [f"y_true{i}" for i in range(num_cat)] + [f"{prediction_label}{j}" for j in range(num_cat)]
            if uncertainty is not None:
                header += [f'uncertainty{i}' for i in range(num_cat)]
            writer.writerow(header)
            for i, group in enumerate(unique_groups):
                if uncertainty is not None:
                    row = np.concatenate([ [group], y_true_group[group], avg_by_group[i], np.array(uncertainty_by_group[i]) ])
                else:
                    row = np.concatenate([ [group], y_true_group[group], avg_by_group[i] ])
                writer.writerow(row)
    # -------------------------------------------------------------------------

    return avg_by_group

def _cph_metrics(args):
    """Internal function to calculate tile, slide, and patient level metrics for a CPH outcome."""
    # Detect number of outcome categories
    num_cat = args.y_pred.shape[1]

    # Generate c_index
    args.c_index['tile'] = concordance_index(args.y_true, args.y_pred)

    # Generate and save slide-level averages of each outcome
    averages_by_slide = _get_average_by_group(args.y_pred,
                                            prediction_label="average",
                                            unique_groups=args.unique_slides,
                                            tile_to_group=args.tile_to_slides,
                                            y_true_group=args.y_true_slide,
                                            num_cat=num_cat,
                                            label_end=args.label_end,
                                            save_predictions=args.save_slide_predictions,
                                            data_dir=args.data_dir,
                                            label="slide")
    y_true_by_slide = np.array([args.y_true_slide[slide] for slide in args.unique_slides])
    args.c_index['slide'] = concordance_index(y_true_by_slide, averages_by_slide)
    if not args.patient_error:
        # Generate and save patient-level averages of each outcome
        averages_by_patient = _get_average_by_group(args.y_pred,
                                                    prediction_label="average",
                                                    unique_groups=args.patients,
                                                    tile_to_group=args.tile_to_patients,
                                                    y_true_group=args.y_true_patient,
                                                    num_cat=num_cat,
                                                    label_end=args.label_end,
                                                    save_predictions=args.save_patient_predictions,
                                                    data_dir=args.data_dir,
                                                    label="patient")
        y_true_by_patient = np.array([args.y_true_patient[patient] for patient in args.patients])
        args.c_index['patient'] = concordance_index(y_true_by_patient, averages_by_patient)

def _linear_metrics(args):
    """Internal function to calculate tile, slide, and patient level metrics for a linear outcome."""
    # Detect number of outcome categories
    num_cat = args.y_pred.shape[1]

    # Main loop
    # Generate R-squared
    args.r_squared['tile'] = generate_scatter(args.y_true,
                                              args.y_pred,
                                              args.data_dir,
                                              args.label_end,
                                              plot=args.plot,
                                              neptune_run=args.neptune_run)

    # Generate and save slide-level averages of each outcome
    averages_by_slide = _get_average_by_group(args.y_pred,
                                            prediction_label="average",
                                            unique_groups=args.unique_slides,
                                            tile_to_group=args.tile_to_slides,
                                            y_true_group=args.y_true_slide,
                                            num_cat=num_cat,
                                            label_end=args.label_end,
                                            save_predictions=args.save_slide_predictions,
                                            data_dir=args.data_dir,
                                            label="slide")
    y_true_by_slide = np.array([args.y_true_slide[slide] for slide in args.unique_slides])
    args.r_squared['slide'] = generate_scatter(y_true_by_slide,
                                               averages_by_slide,
                                               args.data_dir,
                                               args.label_end+"_by_slide",
                                               neptune_run=args.neptune_run)
    if not args.patient_error:
        # Generate and save patient-level averages of each outcome
        averages_by_patient = _get_average_by_group(args.y_pred,
                                                    prediction_label="average",
                                                    unique_groups=args.patients,
                                                    tile_to_group=args.tile_to_patients,
                                                    y_true_group=args.y_true_patient,
                                                    num_cat=num_cat,
                                                    label_end=args.label_end,
                                                    save_predictions=args.save_patient_predictions,
                                                    data_dir=args.data_dir,
                                                    label="patient",
                                                    uncertainty=args.y_std)

        y_true_by_patient = np.array([args.y_true_patient[patient] for patient in args.patients])
        args.r_squared['patient'] = generate_scatter(y_true_by_patient,
                                                     averages_by_patient,
                                                     args.data_dir,
                                                     args.label_end+"_by_patient",
                                                     neptune_run=args.neptune_run)

def zero_args(args, level, outcome_name, weighted, onehot_pool, filtered_uq):
    if onehot_pool and not weighted:
        args.auc[level][outcome_name] = []
        args.ap[level][outcome_name] = []
    if not onehot_pool and not weighted:
        args.auc_avgpool[level][outcome_name] = []
        args.ap_avgpool[level][outcome_name] = []
    if onehot_pool and weighted:
        args.auc_weight[level][outcome_name] = []
        args.ap_weight[level][outcome_name] = []
    if not onehot_pool and weighted:
        args.auc_avgpool_weight[level][outcome_name] = []
        args.ap_avgpool_weight[level][outcome_name] = []
    if onehot_pool and filtered_uq:
        args.auc_filter[level][outcome_name] = []
        args.ap_filter[level][outcome_name] = []
    if not onehot_pool and filtered_uq:
        args.auc_avgpool_filter[level][outcome_name] = []
        args.ap_avgpool_filter[level][outcome_name] = []
    return args
    
def set_args(args, level, outcome_name, weighted, onehot_pool, filtered_uq, auc, ap):
    if onehot_pool and not weighted:
        args.auc[level][outcome_name] += [auc]
        args.ap[level][outcome_name] += [ap]
    if not onehot_pool and not weighted:
        args.auc_avgpool[level][outcome_name] += [auc]
        args.ap_avgpool[level][outcome_name] += [ap]
    if onehot_pool and weighted:
        args.auc_weight[level][outcome_name] += [auc]
        args.ap_weight[level][outcome_name] += [ap]
    if not onehot_pool and weighted:
        args.auc_avgpool_weight[level][outcome_name] += [auc]
        args.ap_avgpool_weight[level][outcome_name] += [ap]
    if onehot_pool and filtered_uq:
        args.auc_filter[level][outcome_name] += [auc]
        args.ap_filter[level][outcome_name] += [ap]
    if not onehot_pool and filtered_uq:
        args.auc_avgpool_filter[level][outcome_name] += [auc]
        args.ap_avgpool_filter[level][outcome_name] += [ap]
    return args 
    
def _categorical_metrics(args, outcome_name, starttime=None, onehot_pool=True):

    """Internal function to calculate tile, slide, and patient level metrics for a categorical outcome."""
    start = starttime
    start = starttime
    num_cat = max(np.max(args.y_true)+1, args.y_pred.shape[1])
    # For categorical models, convert to one-hot encoding
    weighted = args.weights is not None
    filtered_uq = args.filters is not None
    printstr = ""
    if not onehot_pool:
        printstr += " Avg Pool"
    if weighted:
        printstr += " Weighted"
    if filtered_uq:
        printstr += " UQ Filtered"
    printstr += " "
    for level in ('tile', 'slide', 'patient'):
        zero_args(args, level, outcome_name, weighted, onehot_pool, filtered_uq)

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=8) as p:
    # TODO: this is memory inefficient as it copies y_true / y_pred to each subprocess
    # Furthermore, it copies all categories when only one category is needed for each process
    # Consider implementing shared memory (although this would eliminate compatibility with python 3.7)
        try:
            for i, (auc, ap, thresh) in enumerate(p.imap(partial(_generate_tile_roc,
                                                                y_true=args.y_true,
                                                                y_pred=args.y_pred,
                                                                data_dir=args.data_dir,
                                                                label_start=args.label_start + outcome_name + "_",
                                                                histogram=args.histogram), range(num_cat))):
                set_args(args, 'tile', outcome_name, weighted, onehot_pool, filtered_uq, auc, ap)                                            
                log.info("Tile-level" + printstr + f"AUC (cat #{i:>2}): {auc:.3f}, AP: {ap:.3f} (opt. threshold: {thresh:.3f})")
        except ValueError as e:
            # Occurs when predictions contain NaN
            log.error(f'Error encountered when generating AUC: {e}')
            for level in ('tile', 'slide', 'patient'):
                set_args(args, level, outcome_name, weighted, onehot_pool, filtered_uq, -1, -1)  
            return

    # Convert predictions to one-hot encoding
    if onehot_pool:
        onehot_predictions = np.array([to_onehot(x, num_cat) for x in np.argmax(args.y_pred, axis=1)])
    else:
        onehot_predictions = args.y_pred
    # Compare one-hot predictions to one-hot y_true for category-level accuracy
    split_predictions = np.split(onehot_predictions, num_cat, 1)
    for ci, cat_pred_array in enumerate(split_predictions):
        try:
            y_true_in_category = args.y_true[:, ci]
            num_tiles_in_category = np.sum(y_true_in_category)
            correct_pred = np.sum(cat_pred_array[np.argwhere(y_true_in_category>0)])
            category_accuracy = correct_pred / num_tiles_in_category
            cat_percent_acc = category_accuracy * 100
            log.info(f"Category {ci} accuracy: {cat_percent_acc:.1f}% ({correct_pred}/{num_tiles_in_category})")
        except IndexError:
            log.warning(f"Unable to generate category-level accuracy stats for category index {ci}")

    # Generate slide-level percent calls
    percent_calls_by_slide = _get_average_by_group(onehot_predictions,
                                                   prediction_label="percent_tiles_positive",
                                                   unique_groups=args.unique_slides,
                                                   tile_to_group=args.tile_to_slides,
                                                   y_true_group=args.y_true_slide,
                                                   num_cat=num_cat,
                                                   label_end="_" + outcome_name + args.label_end,
                                                   save_predictions=args.save_slide_predictions,
                                                   data_dir=args.data_dir,
                                                   label="slide",
                                                   weights = args.weights,
                                                   filters = args.filters,
                                                   filter_thresh = args.filter_thresh)

    # Generate slide-level ROC
    for i in range(num_cat):
        try:
            slide_y_pred = percent_calls_by_slide[:, i]
            slide_y_true = np.array([args.y_true_slide[slide][i] for slide in args.unique_slides])
            roc_res = generate_roc(slide_y_true,
                                   slide_y_pred,
                                   args.data_dir, f'{args.label_start}{outcome_name}_slide_ROC{i}',
                                   neptune_run=args.neptune_run)
            roc_auc, ap, thresh = roc_res
            set_args(args, 'slide', outcome_name, weighted, onehot_pool, filtered_uq, roc_auc, ap)
            log.info("Slide-level" + printstr + f"AUC (cat #{i:>2}): {roc_auc:.3f}, AP: {ap:.3f} (opt. threshold: {thresh:.3f})")
        except IndexError:
            log.warning(f"Unable to generate slide-level stats for outcome {i}")

    if not args.patient_error:
        # Generate patient-level percent calls
        percent_calls_by_patient = _get_average_by_group(onehot_predictions,
                                                         prediction_label="percent_tiles_positive",
                                                         unique_groups=args.patients,
                                                         tile_to_group=args.tile_to_patients,
                                                         y_true_group=args.y_true_patient,
                                                         num_cat=num_cat,
                                                         label_end="_" + outcome_name + args.label_end,
                                                         save_predictions=args.save_patient_predictions,
                                                         data_dir=args.data_dir,
                                                         uncertainty=args.y_std,
                                                         label="patient",
                                                         weights = args.weights,
                                                         filters = args.filters,
                                                         filter_thresh = args.filter_thresh)

        # Generate patient-level ROC
        for i in range(num_cat):
            try:
                patient_y_pred = percent_calls_by_patient[:, i]
                patient_y_true = np.array([args.y_true_patient[patient][i] for patient in args.patients])
                roc_res = generate_roc(patient_y_true,
                                       patient_y_pred,
                                       args.data_dir,
                                       f'{args.label_start}{outcome_name}_patient_ROC{i}',
                                       neptune_run=args.neptune_run)
                roc_auc, ap, thresh = roc_res
                set_args(args, 'patient', outcome_name, weighted, onehot_pool, filtered_uq, roc_auc, ap)
                log.info(f"Patient-level" + printstr + f"AUC (cat #{i:>2}): {roc_auc:.3f}, AP: {ap:.3f} (opt. threshold: {thresh:.3f})")
            except IndexError:
                log.warning(f"Unable to generate patient-level stats for outcome {i}")

def filtered_prediction(logits, filter=None):
    """Generates a prediction from a logits vector masked by a given filter.

    Args:
        filter (list(int)): List of logit indices to include when generating a prediction. All other logits will be masked.

    Returns:
        int: index of prediction.
    """
    if filter is None:
        prediction_mask = np.zeros(logits.shape, dtype=np.int)
    else:
        prediction_mask = np.ones(logits.shape, dtype=np.int)
        prediction_mask[filter] = 0
    masked_logits = np.ma.masked_array(logits, mask=prediction_mask)
    return np.argmax(masked_logits)

def get_centroid_index(input_array):
    """Calculate index nearest to centroid from a given two-dimensional input array."""
    km = KMeans(n_clusters=1).fit(input_array)
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, input_array)
    return closest[0]

def calculate_centroid(activations):
    """Calcultes slide-level centroid indices for a provided activations dict.

    Args:
        activations (dict): Dict mapping slide names to ndarray of activations across tiles,
            of shape (n_tiles, n_features)

    Returns:
        dict: Dict mapping slides to index of tile nearest to centroid
        dict: Dict mapping slides to activations of tile nearest to centroid
    """

    optimal_indices = {}
    centroid_activations = {}
    for slide in activations:
        if not len(activations[slide]): continue
        km = KMeans(n_clusters=1).fit(activations[slide])
        closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, activations[slide])
        closest_index = closest[0]
        closest_activations = activations[slide][closest_index]
        optimal_indices.update({slide: closest_index})
        centroid_activations.update({slide: closest_activations})
    return optimal_indices, centroid_activations

def normalize_layout(layout, min_percentile=1, max_percentile=99, relative_margin=0.1):
    """Removes outliers and scales layout to between [0,1]."""

    # compute percentiles
    mins = np.percentile(layout, min_percentile, axis=(0))
    maxs = np.percentile(layout, max_percentile, axis=(0))

    # add margins
    mins -= relative_margin * (maxs - mins)
    maxs += relative_margin * (maxs - mins)

    # `clip` broadcasts, `[None]`s added only for readability
    clipped = np.clip(layout, mins, maxs)

    # embed within [0,1] along both axes
    clipped -= clipped.min(axis=0)
    clipped /= clipped.max(axis=0)

    return clipped

def gen_umap(array, dim=2, n_neighbors=50, min_dist=0.1, metric='cosine', low_memory=False):
    """Generates and returns a umap from a given array, using umap.UMAP"""
    import umap # Imported in this function due to long import times
    try:
        layout = umap.UMAP(n_components=dim,
                           verbose=(log.getEffectiveLevel() <= 20),
                           n_neighbors=n_neighbors,
                           min_dist=min_dist,
                           metric=metric,
                           low_memory=low_memory).fit_transform(array)
    except ValueError:
        raise errors.StatsError("Error performing UMAP. Please make sure you are supplying a non-empty TFRecord " + \
                                "array and that the TFRecords are not empty.")

    return normalize_layout(layout)

def save_histogram(y_true, y_pred, outdir, name='histogram', neptune_run=None, subsample=500):
    """Generates histogram of y_pred, labeled by y_true, saving to outdir."""
    # Subsample
    if subsample and y_pred.shape[0] > subsample:
        idx = np.arange(y_pred.shape[0])
        idx = np.random.choice(idx, subsample)
        y_pred = y_pred[idx]
        y_true = y_true[idx]

    cat_false = [yp for i, yp in enumerate(y_pred) if y_true[i] == 0]
    cat_true = [yp for i, yp in enumerate(y_pred) if y_true[i] == 1]

    plt.clf()
    plt.title('Tile-level Predictions')
    try:
        sns.histplot(cat_false, bins=30, kde=True, stat="density", linewidth=0, color="skyblue", label="Negative")
        sns.histplot(cat_true, bins=30, kde=True, stat="density", linewidth=0, color="red", label="Positive")
    except np.linalg.LinAlgError:
        log.warning("Unable to generate histogram, insufficient data")
    plt.legend()
    plt.savefig(os.path.join(outdir, f'{name}.png'))
    if neptune_run:
        neptune_run[f'results/graphs/{name}'].upload(os.path.join(outdir, f'{name}.png'))

def generate_roc(y_true, y_pred, save_dir=None, name='ROC', neptune_run=None):
    """Generates and saves an ROC with a given set of y_true, y_pred values."""
    # ROC
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    # Precision recall
    precision, recall, pr_thresholds = metrics.precision_recall_curve(y_true, y_pred)
    average_precision = metrics.average_precision_score(y_true, y_pred)

    # --- Binomal Z score -----------------------------------------------------
    #def binomial_p(n1, p1, n2, p2):
    #    n=n1+n2
    #    p0=(n1*p1+n2*p2)/(n1+n2)
    #    z = ((p1-p2)-0)/(np.sqrt(2*(p0*(1-p0)/n)))
    #    return z, 1-stats.norm.cdf(abs(z))

    #def str_num(n):
    #    if n < 0.001: return f"{n:.2e}"
    #    else: return f"{n:.4f}"

    # True category = 1
    #n1 = np.sum(y_true)
    #p1 = np.sum(y_pred[np.argwhere(y_true == 1)]) / n1

    # True category = 0
    #n2 = y_true.shape[0] - n1
    #p2 = np.sum(y_pred[np.argwhere(y_true == 0)]) / n2

    #binom_z, binom_p = binomial_p(n1, p1, n2, p2)
    #log.debug(sf.util.blue("Binomial Z: ") + f"{binom_z:.3f} {sf.util.blue('P')}: {str_num(binom_p)}")
    # -------------------------------------------------------------------------

    # Calculate optimal cutoff via maximizing Youden's J statistic (sens+spec-1, or TPR - FPR)
    try:
        optimal_threshold = threshold[list(zip(tpr,fpr)).index(max(zip(tpr,fpr), key=lambda x: x[0]-x[1]))]
    except:
        optimal_threshold = -1

    # Plot
    if save_dir:
        # ROC
        plt.clf()
        plt.title('ROC Curve')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.savefig(os.path.join(save_dir, f'{name}.png'))

        # Precision recall
        plt.clf()
        plt.title('Precision-Recall Curve')
        plt.plot(precision, recall, 'b', label = 'AP = %0.2f' % average_precision)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Recall')
        plt.xlabel('Precision')
        plt.savefig(os.path.join(save_dir, f'{name}-PRC.png'))

        if neptune_run:
            neptune_run[f'results/graphs/{name}'].upload(os.path.join(save_dir, f'{name}.png'))
            neptune_run[f'results/graphs/{name}-PRC'].upload(os.path.join(save_dir, f'{name}-PRC.png'))

    return roc_auc, average_precision, optimal_threshold

def generate_combined_roc(y_true, y_pred, save_dir, labels, name='ROC', neptune_run=None):
    """Generates and saves overlapping ROCs with a given combination of y_true and y_pred."""
    # Plot
    plt.clf()
    plt.title(name)
    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

    rocs = []
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        fpr, tpr, threshold = metrics.roc_curve(yt, yp)
        roc_auc = metrics.auc(fpr, tpr)
        rocs += [roc_auc]
        plt.plot(fpr, tpr, colors[i % len(colors)], label = labels[i] + f' (AUC: {roc_auc:.2f})')

    # Finish plot
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('TPR')
    plt.xlabel('FPR')

    plt.savefig(os.path.join(save_dir, f'{name}.png'))

    if neptune_run:
        neptune_run[f'results/graphs/{name}'].upload(os.path.join(save_dir, f'{name}.png'))

    return rocs

def read_predictions(predictions_file, level):
    """Reads predictions from a previously saved CSV file."""
    predictions = {}
    y_pred_label = "percent_tiles_positive" if level in ("patient", "slide") else "y_pred"
    with open(predictions_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        prediction_labels = [h.split('y_true')[-1] for h in header if "y_true" in h]
        for label in prediction_labels:
            predictions.update({label: {
                'y_true': [],
                'y_pred': []
            }})
        for row in reader:
            for label in prediction_labels:
                yti = header.index(f'y_true{label}')
                ypi = header.index(f'{y_pred_label}{label}')
                predictions[label]['y_true'] += [int(row[yti])]
                predictions[label]['y_pred'] += [float(row[ypi])]
    return predictions

def generate_scatter(y_true, y_pred, data_dir, name='_plot', plot=True, neptune_run=None):
    '''Generate and save scatter plots and calculate R2 statistic for each outcome variable.
        y_true and y_pred are both 2D arrays; the first dimension is each observation,
        the second dimension is each outcome variable.'''

    # Error checking
    if y_true.shape != y_pred.shape:
        log.error(f"Y_true (shape: {y_true.shape}) and y_pred (shape: {y_pred.shape}) must \
                    have the same shape to generate a scatter plot")
        return
    if y_true.shape[0] < 2:
        log.error(f"Must have more than one observation to generate a scatter plot with R2 statistics.")
        return

    # Perform scatter for each outcome variable
    r_squared = []
    for i in range(y_true.shape[1]):

        # Statistics
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_true[:,i], y_pred[:,i])
        r_squared += [r_value ** 2]

        if plot:
            # Plot
            p = sns.jointplot(x=y_true[:,i], y=y_pred[:,i], kind="reg")
            p.set_axis_labels('y_true', 'y_pred')
            plt.savefig(os.path.join(data_dir, f'Scatter{name}-{i}.png'))

            if neptune_run:
                neptune_run[f'results/graphs/Scatter{name}-{i}'].upload(os.path.join(data_dir, f'Scatter{name}-{i}.png'))

    return r_squared

def basic_metrics(y_true, y_pred):
    '''Generates basic performance metrics, including sensitivity, specificity, and accuracy.'''
    assert(len(y_true) == len(y_pred))
    assert([y in [0,1] for y in y_true])
    assert([y in [0,1] for y in y_pred])

    TP = 0 # True positive
    TN = 0 # True negative
    FP = 0 # False positive
    FN = 0 # False negative

    for i, yt in enumerate(y_true):
        yp = y_pred[i]
        if yt == 1 and yp == 1:
            TP += 1
        elif yt == 1 and yp == 0:
            FN += 1
        elif yt == 0 and yp == 1:
            FP += 1
        elif yt == 0 and yp == 0:
            TN += 1

    metrics = {}
    metrics['accuracy'] = (TP + TN) / (TP + TN + FP + FN)
    metrics['sensitivity'] = TP / (TP + FN)
    metrics['specificity'] = TN / (TN + FP)
    metrics['precision'] = metrics.precision_score(y_true, y_pred)
    metrics['recall'] = metrics.recall_score(y_true, y_pred)
    metrics['f1_score'] = metrics.f1_score(y_true, y_pred)
    metrics['kappa'] = metrics.cohen_kappa_score(y_true, y_pred)
    return metrics

def concordance_index(y_true, y_pred):
    '''Calculates concordance index from a given y_true and y_pred.'''
    E = y_pred[:, -1]  # HERE
    y_pred = y_pred[:, :-1]  # HERE
    y_pred = y_pred.flatten()
    E = E.flatten()
    y_true = y_true.flatten()
    y_pred = - y_pred # Need to take negative to get concordance index since these are log hazard ratios
    return c_index(y_true, y_pred, E)

def predictions_to_dataframe(y_true, y_pred, tile_to_slides, outcome_names, uncertainty=None):
    # Save tile-level predictions
    # Assumes structure of y_true, y_pred, uncertainty is:
    # -- List of length num_outcomes, containing numpy arrays
    # -- Each np array is either shape (num_tiles) [single linear outcome] or (num_tiles, num_categories) [categorical]

    if isinstance(y_true, list):
        assert len(y_true) == len(y_pred), "Number of outcomes in y_true and y_pred must match"
        assert len(y_true) == len(outcome_names), "Number of provided outcome names must equal number of y_true outcomes"
    else:
        # Check for multiple linear outcomes in ndarray format
        if len(outcome_names) > 1 and len(outcome_names) == y_pred.shape[1]:
            if y_true is not None:
                assert y_true.shape == y_pred.shape, f"Shape mismatch: y_true ({y_true.shape}) != y_pred ({y_pred.shape})"
                y_true = [y_true[:, i] for i in range(y_true.shape[1])]
            y_pred = [y_pred[:, i] for i in range(y_pred.shape[1])]
        elif len(outcome_names) > 1:
            raise errors.StatsError("If providing only one y_pred, length of outcome_names must be one")
        else:
            y_true = [y_true]

    if not isinstance(y_pred, list):
        y_pred = [y_pred]

    if uncertainty is not None and not isinstance(uncertainty, list):
        uncertainty = [uncertainty]

    pd_dict = {
        'slide': pd.Series(tile_to_slides, dtype=str)
    }

    for oi, outcome in enumerate(outcome_names):
        if y_true is not None and y_true[oi] is not None:
            # Single label for each image (e.g. for single linear outcomes)
            if len(y_true[oi].shape) == 1:
                pd_dict[f'{outcome}_y_true0'] = pd.Series(y_true[oi])
            # Multiple labels for each image (e.g. onehot-encoded categorical outcomes)
            else:
                for j in range(y_true[oi].shape[1]):
                    pd_dict[f'{outcome}_y_true{j}'] = pd.Series(y_true[oi][:, j])

        # Single output for each image (e.g. for single linear outcomes)
        if len(y_pred[oi].shape) == 1:
            pd_dict[f'{outcome}_y_pred0'] = pd.Series(y_pred[oi])
        # Multiple outputs for each image (e.g. softmax logits for categorical outcomes)
        else:
            for j in range(y_pred[oi].shape[1]):
                pd_dict[f'{outcome}_y_pred{j}'] = pd.Series(y_pred[oi][:, j])

        if uncertainty is not None:
            # Single uncertainty for each image (e.g. for single linear outcomes)
            if len(uncertainty[oi].shape) == 1:
                pd_dict[f'{outcome}_uncertainty0'] = pd.Series(uncertainty[oi])
            # Multiple uncertainty values for each image (e.g. variance of each softmax logit, for categorical outcomes)
            else:
                for j in range(uncertainty[oi].shape[1]):
                    pd_dict[f'{outcome}_uncertainty{j}'] = pd.Series(uncertainty[oi][:, j])

    return pd.DataFrame(pd_dict)

def metrics_from_predictions(y_true, y_pred, tile_to_slides, labels, patients, model_type, y_std=None, outcome_names=None,
                             label=None, data_dir=None, save_predictions=True, histogram=False, plot=True,
                             neptune_run=None, y_roi = None, y_unc = None, filter_thresh = None, onehot_pool = 'true'):

    """Generates metrics from a set of predictions.

    For multiple outcomes, y_true and y_pred are expected to be a list of numpy arrays
    (each numpy array coroneresponding to whole-dataset predictions for a single outcome)

    Args:
        y_true (ndarray): True labels for the dataset.
        y_pred (ndarray): Predicted labels for the dataset.
        tile_to_slides (list(str)): List of length y_true of slide names.
        labels (dict): Dictionary mapping slidenames to outcomes.
        patients (dict): Dictionary mapping slidenames to patients.
        model_type (str): Either 'linear', 'categorical', or 'cph'.
        outcome_names (list, optional): List of str, names for outcomes. Defaults to None.
        label (str, optional): Label prefix/suffix for saving. Defaults to None.
        min_tiles (int, optional): Minimum tiles per slide to include in metrics. Defaults to 0.
        data_dir (str, optional): Path to data directory for saving. Defaults to None.
        save_predictions (bool, optional): Save tile, slide, and patient-level predictions to CSV. Defaults to True.
            May take a substantial amount of time for very large datasets.
        histogram (bool, optional): Write histograms to data_dir. Defaults to False.
            Takes a substantial amount of time for large datasets, potentially hours.
        plot (bool, optional): Save scatterplot for linear outcomes. Defaults to True.
        neptune_run (:class:`neptune.Run`, optional): Neptune run in which to log results. Defaults to None.
    """

    start = time.time()
    label_end = "" if not label else f"_{label}"
    label_start = "" if not label else f"{label}_"

    tile_to_patients = np.array([patients[slide] for slide in tile_to_slides])
    unique_patients = np.unique(tile_to_patients)
    unique_slides = np.unique(tile_to_slides)

    # Set up annotations
    y_true_slide = labels
    y_true_patient = {patients[s]: labels[s] for s in labels}
    # Verify patient outcomes are consistent if multiples slides are present for each patient
    patient_error = False
    for slide in labels:
        patient = patients[slide]
        if  y_true_slide[slide] != y_true_patient[patient]:
            log.error("Data integrity failure; patient assigned to multiple slides w/ different outcomes")
            patient_error = True

    # Function to determine which predictions, if any, should be exported to CSV
    def should_save_predictions(group):
        return (save_predictions == True or
               (type(save_predictions) == str and save_predictions == group) or
               (type(save_predictions) == list and group in save_predictions))

    metric_args = types.SimpleNamespace(
        y_true = y_true,
        y_pred = y_pred,
        y_std = y_std,
        unique_slides = unique_slides,
        tile_to_slides = tile_to_slides,
        tile_to_patients = tile_to_patients,
        label_start = label_start,
        label_end = label_end,
        save_slide_predictions = should_save_predictions('slide'),
        save_patient_predictions = should_save_predictions('patient'),
        save_tile_predictions = should_save_predictions('tile'),
        data_dir = data_dir,
        patient_error = patient_error,
        patients = unique_patients,
        r_squared = {'tile': None, 'slide': None, 'patient': None},
        c_index = {'tile': None, 'slide': None, 'patient': None},
        auc = {'tile': {}, 'slide': {}, 'patient': {}},
        ap = {'tile': {}, 'slide': {}, 'patient': {}},
        auc_avgpool = {'tile': {}, 'slide': {}, 'patient': {}},
        ap_avgpool = {'tile': {}, 'slide': {}, 'patient': {}},
        auc_weight = {'tile': {}, 'slide': {}, 'patient': {}},
        ap_weight = {'tile': {}, 'slide': {}, 'patient': {}},
        auc_avgpool_weight = {'tile': {}, 'slide': {}, 'patient': {}},
        ap_avgpool_weight = {'tile': {}, 'slide': {}, 'patient': {}},
        auc_filter = {'tile': {}, 'slide': {}, 'patient': {}},
        ap_filter = {'tile': {}, 'slide': {}, 'patient': {}},
        auc_avgpool_filter = {'tile': {}, 'slide': {}, 'patient': {}},
        ap_avgpool_filter = {'tile': {}, 'slide': {}, 'patient': {}},
        plot = plot,
        histogram = histogram,
        neptune_run = neptune_run
    )
    print("Generating predictions for model type: " + model_type + "\n")
    if model_type == 'categorical':

        # Detect the number of outcomes by y_true
        if type(y_true) == list:
            num_outcomes_by_y_true = len(y_true)
        elif len(y_true.shape) == 1:
            num_outcomes_by_y_true = 1
        else:
            raise errors.StatsError(f"y_true expected to be formated as list of numpy arrays for each outcome category.")

        # Confirm that the number of outcomes provided by y_true match the provided outcome names
        if not outcome_names:
            outcome_names = {f"Outcome {i}" for i in range(num_outcomes_by_y_true)}
        elif len(outcome_names) != num_outcomes_by_y_true:
            raise errors.StatsError(f"Number of outcome names {len(outcome_names)} does not " + \
                                    f"match y_true {num_outcomes_by_y_true}")

        for oi, outcome in enumerate(outcome_names):
            if len(outcome_names) > 1:
                metric_args.y_true_slide = {s:v[oi] for s,v in y_true_slide.items()}
                metric_args.y_true_patient = {s:v[oi] for s,v in y_true_patient.items()}
                metric_args.y_pred = y_pred[oi]
                metric_args.y_true = y_true[oi]
            else:
                metric_args.y_true_slide = y_true_slide
                metric_args.y_true_patient = y_true_patient
                metric_args.y_pred = y_pred
                metric_args.y_true = y_true
            metric_args.weights = None
            metric_args.filters = None
            metric_args.filter_thresh = filter_thresh
            num_observed_outcome_categories = np.max(metric_args.y_true)+1
            if num_observed_outcome_categories != metric_args.y_pred.shape[1]:
                log.warning(f"Model predictions have different number of outcome categories ({metric_args.y_pred.shape[1]}) " + \
                            f"than provided annotations ({num_observed_outcome_categories})!")
                if num_observed_outcome_categories == 2 and metric_args.y_pred.shape[1] == 1:
                    log.warning(f"Assuming linear model being evaluated as categorical.")
                    print(metric_args.y_pred[0])
                    print(metric_args.y_pred[1])
                    print(metric_args.y_pred[2])
                    metric_args.y_pred = np.array([[x[0], x[0]] for x in metric_args.y_pred])
                    print(metric_args.y_pred[0])
                    print(metric_args.y_pred[1])
                    print(metric_args.y_pred[2])
            num_cat = max(num_observed_outcome_categories, metric_args.y_pred.shape[1])
            # For categorical models, convert to one-hot encoding 
            metric_args.y_true = np.array([to_onehot(i, num_cat) for i in metric_args.y_true])
            metric_args.y_true_slide = {k:to_onehot(v, num_cat) for k,v in metric_args.y_true_slide.items()}
            metric_args.y_true_patient = {k:to_onehot(v, num_cat) for k,v in metric_args.y_true_patient.items()}

            # If this is from a PyTorch model, predictions may not be in softmax form. We will need to enforce softmax encoding
            # for tile-level statistics.
            if sf.backend() == 'torch':
                metric_args.y_pred = softmax(metric_args.y_pred, axis=1)
            log.info(f"Validation metrics for outcome {sf.util.green(outcome)}:")
            if onehot_pool == 'true' or onehot_pool == 'both':
                _categorical_metrics(metric_args, outcome, starttime=start, onehot_pool=True)
            if onehot_pool == 'false' or onehot_pool == 'both':
                _categorical_metrics(metric_args, outcome, starttime=start, onehot_pool=False)
            if y_roi is not None:
                metric_args.weights = y_roi
                if onehot_pool == 'true' or onehot_pool == 'both':
                    _categorical_metrics(metric_args, outcome, starttime=start, onehot_pool=True)
                if onehot_pool == 'false' or onehot_pool == 'both':
                    _categorical_metrics(metric_args, outcome, starttime=start, onehot_pool=False)
            metric_args.weights = None
            if y_unc is not None:
                metric_args.filters = y_unc
                if onehot_pool == 'true' or onehot_pool == 'both':
                    _categorical_metrics(metric_args, outcome, starttime=start, onehot_pool=True)
                if onehot_pool == 'false' or onehot_pool == 'both':
                    _categorical_metrics(metric_args, outcome, starttime=start, onehot_pool=False)

    elif model_type == 'linear':
        metric_args.y_true_slide = y_true_slide
        metric_args.y_true_patient = y_true_patient
        _linear_metrics(metric_args)

    elif model_type == 'cph':
        metric_args.y_true_slide = y_true_slide
        metric_args.y_true_patient = y_true_patient
        _cph_metrics(metric_args)

    if metric_args.save_tile_predictions:
        df = predictions_to_dataframe(y_true, y_pred, tile_to_slides, outcome_names, uncertainty=metric_args.y_std)
        df.to_csv(os.path.join(data_dir, f"tile_predictions{label_end}.csv"))
        log.debug(f"Predictions saved to {sf.util.green(data_dir)}")

    combined_metrics = {
        'auc': metric_args.auc,
        'ap': metric_args.ap,
        'auc_avgpool': metric_args.auc_avgpool,
        'ap_avgpool': metric_args.ap_avgpool,
        'auc_weight': metric_args.auc_weight,
        'ap_weight': metric_args.ap_weight,
        'auc_avgpool_weight': metric_args.auc_avgpool_weight,
        'ap_avgpool_weight': metric_args.ap_avgpool_weight,
        'auc_filter': metric_args.auc_filter,
        'ap_filter': metric_args.ap_filter,
        'auc_avgpool_filter': metric_args.auc_avgpool_filter,
        'ap_avgpool_filter': metric_args.ap_avgpool_filter,
        'r_squared': metric_args.r_squared,
        'c_index': metric_args.c_index
    }

    return combined_metrics

def predict_from_torch(model, dataset, model_type, pred_args, **kwargs):
    """Generates predictions (y_true, y_pred, tile_to_slide) from a given PyTorch model and dataset.

    Args:
        model (str): Path to PyTorch model.
        dataset (tf.data.Dataset): PyTorch dataloader.
        pred_args (namespace): Namespace containing slide_input, update_corrects, and update_loss functions.
        model_type (str, optional): 'categorical', 'linear', or 'cph'. If multiple linear outcomes are present,
            y_true is stacked into a single vector for each image. Defaults to 'categorical'.

    Returns:
        y_pred, y_std, tile_to_slides
    """

    import torch
    y_true, y_pred, tile_to_slides = [], [], []

    log.debug("Generating predictions from torch model")

    # Get predictions and performance metrics
    model.eval()
    device = torch.device('cuda:0')
    pb = tqdm(desc='Predicting...', total=dataset.num_tiles, ncols=80, unit='img', leave=False)
    for img, yt, slide in dataset: #TODO: support not needing to supply yt

        img = img.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # Slide-level features
                if pred_args.num_slide_features:
                    inp = (img, torch.tensor([pred_args.slide_input[s] for s in slide]).to(device))
                else:
                    inp = (img,)

                res = model(*inp)
                if isinstance(res, list):
                    res = [r.cpu().numpy().copy() for r in res]
                else:
                    res = res.cpu().numpy().copy()
                y_pred += [res]

        tile_to_slides += slide
        pb.update(img.shape[0])

    # Concatenate predictions for each outcome
    if type(y_pred[0]) == list:
        y_pred = [np.concatenate(yp) for yp in zip(*y_pred)]
    else:
        y_pred = np.concatenate(y_pred)
    tile_to_slides = np.array(tile_to_slides)

    if log.getEffectiveLevel() <= 20: sf.util.clear_console()
    log.debug(f"Prediction complete.")

    return y_pred, None, tile_to_slides

def eval_from_torch(model, dataset, model_type, pred_args, **kwargs):
    """Generates predictions (y_true, y_pred, tile_to_slide) from a given PyTorch model and dataset.

    Args:
        model (str): Path to PyTorch model.
        dataset (tf.data.Dataset): PyTorch dataloader.
        pred_args (namespace): Namespace containing slide_input, update_corrects, and update_loss functions.
        model_type (str, optional): 'categorical', 'linear', or 'cph'. If multiple linear outcomes are present,
            y_true is stacked into a single vector for each image. Defaults to 'categorical'.

    Returns:
        y_true, y_pred, y_std, tile_to_slides, accuracy, loss
    """

    import torch
    y_true, y_pred, tile_to_slides = [], [], []
    running_corrects = pred_args.running_corrects
    running_loss = 0
    total = 0

    log.debug("Evaluating torch model")

    model.eval()
    device = torch.device('cuda:0')
    pb = tqdm(desc='Evaluating...', total=dataset.num_tiles, ncols=80, unit='img', leave=False)
    for img, yt, slide in dataset:

        img = img.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # Slide-level features
                if pred_args.num_slide_features:
                    inp = (img, torch.tensor([pred_args.slide_input[s] for s in slide]).to(device))
                else:
                    inp = (img,)

                res = model(*inp)

                running_corrects = pred_args.update_corrects(res, yt, running_corrects)
                running_loss = pred_args.update_loss(res, yt, running_loss, img.size(0))
                if isinstance(res, list):
                    res = [r.cpu().numpy().copy() for r in res]
                else:
                    res = res.cpu().numpy().copy()
                y_pred += [res]

        if type(yt) == dict:
            y_true += [[yt[f'out-{o}'] for o in range(len(yt))]]
        else:
            yt = yt.detach().numpy().copy()
            y_true += [yt]

        tile_to_slides += slide
        total += img.shape[0]
        pb.update(img.shape[0])

    # Concatenate predictions for each outcome
    if type(y_pred[0]) == list:
        y_pred = [np.concatenate(yp) for yp in zip(*y_pred)]
    else:
        y_pred = np.concatenate(y_pred)

    # Concatenate y_true for each outcome
    if type(y_true[0]) == list:
        y_true = [np.concatenate(yt) for yt in zip(*y_true)]
    else:
        y_true = np.concatenate(y_true)

    tile_to_slides = np.array(tile_to_slides)

    # Merge multiple linear outcomes into a single vector
    if model_type == 'linear' and isinstance(y_true, list):
        y_true = np.stack(y_true, axis=1)

    # Calculate final accuracy and loss
    loss = running_loss / total
    if isinstance(running_corrects, dict): acc = {k:v.cpu().numpy()/total for k,v in running_corrects.items()}
    elif isinstance(running_corrects, (int, float)): acc = running_corrects / total
    else: acc = running_corrects.cpu().numpy() / total

    if log.getEffectiveLevel() <= 20: sf.util.clear_console()
    log.debug(f"Evaluation complete.")

    return y_true, y_pred, None, tile_to_slides, acc, loss

def predict_from_tensorflow(model, dataset, model_type, pred_args, num_tiles=0, uq_n=30, weight_model = None):
    """Generates predictions (y_true, y_pred, tile_to_slide) from a given Tensorflow model and dataset.

    Args:
        model (str): Path to Tensorflow model.
        dataset (tf.data.Dataset): Tensorflow dataset.
        model_type (str, optional): 'categorical', 'linear', or 'cph'. Will not attempt to calculate accuracy
            for non-categorical models. Defaults to 'categorical'.
        pred_args (namespace): Namespace containing the property `loss`, loss function used to calculate loss.
        num_tiles (int, optional): Used for progress bar. Defaults to 0.
        uq_n (int, optional): Number of per-tile inferences to perform is calculating uncertainty via dropout.
        evaluate (bool, optional): Calculate and return accuracy and loss. Dataset must also return y_true.

    Returns:
        y_true, y_pred, tile_to_slides, accuracy, loss
    """

    import tensorflow as tf
    from slideflow.model.tensorflow_utils import get_uq_predictions

    if weight_model:
        weight_model = tf.keras.models.load_model(weight_model)
    @tf.function
    def get_predictions(img, training=False):
        if weight_model:
            roi = weight_model(img, training=training)
            return model(img, training=training), roi
        else:
            return model(img, training=training)
            
    y_pred, tile_to_slides, y_roi = [], [], []
    y_std = [] if pred_args.uq else None
    num_vals, num_batches, num_outcomes = 0, 0, 0

    pb = tqdm(total=num_tiles, desc='Predicting...', ncols=80, leave=False)
    for img, yt, slide in dataset: #TODO: support not needing to supply yt
        pb.update(slide.shape[0])
        tile_to_slides += [slide_bytes.decode('utf-8') for slide_bytes in slide.numpy()]
        num_vals += slide.shape[0]
        num_batches += 1
        if pred_args.uq:
            yp_mean, yp_std, num_outcomes = get_uq_predictions(img, get_predictions, num_outcomes, uq_n)
            y_pred += [yp_mean]
            y_std += [yp_std]
        else:
            if weight_model:
                yp, roi = get_predictions(img, training=False)
            else:
                yp, roi = get_predictions(img, training=False), None
            y_pred += [yp]
            y_roi += [roi]
    pb.close()

    tile_to_slides = np.array(tile_to_slides)
    if type(y_pred[0]) == list:
        # Concatenate predictions for each outcome
        y_pred = [np.concatenate(yp) for yp in zip(*y_pred)]
        if pred_args.uq:
            y_std = [np.concatenate(ys) for ys in zip(*y_std)]
    else:
        y_pred = np.concatenate(y_pred)
        if pred_args.uq:
            y_std = np.concatenate(y_std)
    log.debug(f"Prediction complete.")
    if not weight_model:
        y_roi = None
    else:
        if type(y_roi[0]) == list:
            # Concatenate predictions for each outcome
            y_roi = [np.concatenate(yp) for yp in zip(*y_roi)]
        else:
            y_roi = np.concatenate(y_roi)   
    return y_pred, y_std, tile_to_slides, y_roi

def eval_from_tensorflow(model, dataset, model_type, pred_args, num_tiles=0, uq_n=30, weight_model = None, filter_model=None):
    """Generates predictions (y_true, y_pred, tile_to_slide) from a given Tensorflow model and dataset.

    Args:
        model (str): Path to Tensorflow model.
        dataset (tf.data.Dataset): Tensorflow dataset.
        model_type (str, optional): 'categorical', 'linear', or 'cph'. Will not attempt to calculate accuracy
            for non-categorical models. Defaults to 'categorical'.
        pred_args (namespace): Namespace containing the property `loss`, loss function used to calculate loss.
        num_tiles (int, optional): Used for progress bar. Defaults to 0.
        uq_n (int, optional): Number of per-tile inferences to perform is calculating uncertainty via dropout.
        evaluate (bool, optional): Calculate and return accuracy and loss. Dataset must also return y_true.

    Returns:
        y_true, y_pred, tile_to_slides, accuracy, loss
    """

    import tensorflow as tf
    from slideflow.model.tensorflow_utils import get_uq_predictions

    if weight_model:
        weight_model = tf.keras.models.load_model(weight_model)
    if filter_model:
        filter_model = sf.model.tensorflow.UncertaintyInterface(filter_model)
    @tf.function
    def get_predictions_new(img, training=False):
        roi = None
        uncertainty = None
        if weight_model:
            roi = weight_model(img, training=training)
        if filter_model:
            logits, uncertainty = filter_model(img)
        return model(img, training=training), roi, uncertainty

    def get_predictions(img, training=False):
        return model(img, training=training)
    y_true, y_pred, tile_to_slides, y_roi, y_unc = [], [], [], [], []
    y_std = [] if pred_args.uq else None
    num_vals, num_batches, num_outcomes, running_loss = 0, 0, 0, 0
    is_cat = (model_type == 'categorical')
    if not is_cat: acc = None

    pb = tqdm(total=num_tiles, desc='Evaluating...', ncols=80, leave=False)
    for img, yt, slide in dataset:
        pb.update(slide.shape[0])
        tile_to_slides += [slide_bytes.decode('utf-8') for slide_bytes in slide.numpy()]
        num_vals += slide.shape[0]
        num_batches += 1

        if pred_args.uq:
            yp, yp_std, num_outcomes = get_uq_predictions(img, get_predictions, num_outcomes, uq_n)
            y_pred += [yp]
            y_std += [yp_std]
        else:
            yp, roi, unc = get_predictions_new(img, training=False)              
            y_pred += [yp]
            y_roi += [roi]
            y_unc += [unc]
        if type(yt) == dict:
            y_true += [[yt[f'out-{o}'].numpy() for o in range(len(yt))]]
            yt = [yt[f'out-{o}'] for o in range(len(yt))]
        else:
            y_true += [yt.numpy()]
        loss = pred_args.loss(yt, yp)
        running_loss += tf.math.reduce_sum(loss).numpy() * slide.shape[0] 
    pb.close()

    tile_to_slides = np.array(tile_to_slides)
    if type(y_pred[0]) == list:
        # Concatenate predictions for each outcome
        y_pred = [np.concatenate(yp) for yp in zip(*y_pred)]
        if pred_args.uq:
            y_std = [np.concatenate(ys) for ys in zip(*y_std)]
    else:
        y_pred = np.concatenate(y_pred)
        if pred_args.uq:
            y_std = np.concatenate(y_std)

    if type(y_true[0]) == list:
        # Concatenate y_true for each outcome
        y_true = [np.concatenate(yt) for yt in zip(*y_true)]
        if is_cat:
            acc = [np.sum(y_true[i] == np.argmax(y_pred[i], axis=1)) / num_vals for i in range(len(y_true))]
    else:
        y_true = np.concatenate(y_true)
        if is_cat:
            acc = np.sum(y_true == np.argmax(y_pred, axis=1)) / num_vals

    loss = running_loss / num_vals # Note that Keras loss during training includes regularization losses,
                                #  so this loss will not match validation loss calculated during training
    log.debug(f"Evaluation complete.")
    if not weight_model:
        y_roi = None
    else:
        if type(y_roi[0]) == list:
            # Concatenate predictions for each outcome
            y_roi = [np.concatenate(yp) for yp in zip(*y_roi)]
        else:
            y_roi = np.concatenate(y_roi)   
    if not filter_model:
        y_unc = None
    else:
        if type(y_unc[0]) == list:
            # Concatenate predictions for each outcome
            y_unc = [np.concatenate(yp) for yp in zip(*y_unc)]
        else:
            y_unc = np.concatenate(y_unc)   
    return y_true, y_pred, y_std, tile_to_slides, acc, loss, y_roi, y_unc

def predict_from_dataset(model, dataset, model_type, pred_args, **kwargs):
    """Generates predictions (y_pred, tile_to_slide) from a given model and dataset.

    Args:
        model (str): Path to PyTorch model.
        dataset (tf.data.Dataset): PyTorch dataloader.
        pred_args (namespace): Namespace containing slide_input, update_corrects, and update_loss functions.
        model_type (str, optional): 'categorical', 'linear', or 'cph'. If multiple linear outcomes are present,
            y_true is stacked into a single vector for each image. Defaults to 'categorical'.
        num_tiles (int, optional): Used for progress bar with Tensorflow backend. Defaults to 0.

    Returns:
        y_pred, tile_to_slides
    """
    if sf.backend() == 'tensorflow':
        return predict_from_tensorflow(model, dataset, model_type, pred_args, **kwargs)
    else:
        return predict_from_torch(model, dataset, model_type, pred_args, **kwargs)

def eval_from_dataset(*args, **kwargs):
    """Generates predictions (y_true, y_pred, tile_to_slide) and accuracy/loss from a given model and dataset.

    Args:
        model (str): Path to PyTorch model.
        dataset (tf.data.Dataset): PyTorch dataloader.
        pred_args (namespace): Namespace containing slide_input, update_corrects, and update_loss functions.
        model_type (str, optional): 'categorical', 'linear', or 'cph'. If multiple linear outcomes are present,
            y_true is stacked into a single vector for each image. Defaults to 'categorical'.
        num_tiles (int, optional): Used for progress bar with Tensorflow backend. Defaults to 0.

    Returns:
        y_true, y_pred, tile_to_slides, accuracy, loss
    """

    if sf.backend() == 'tensorflow':
        return eval_from_tensorflow(*args, **kwargs)
    else:
        return eval_from_torch(*args, **kwargs)

def predict_from_layer(model, layer_input, input_layer_name='hidden_0', output_layer_index=None):
    """Generate predictions from a model, providing intermediate layer input.

    Args:
        model (str): Path to Tensorflow model
        layer_input (ndarray): Dataset to use as input for the given layer, to generate predictions.
        input_layer_name (str, optional): Name of intermediate layer, to which input is provided. Defaults to 'hidden_0'.
        output_layer_index (int, optional): Excludes layers beyond this index. CPH models include a final
            concatenation layer (softmax + event tensor) that should be excluded. Defaults to None.

    Returns:
        ndarray: Model predictions.
    """
    import tensorflow as tf
    from slideflow.model.tensorflow_utils import get_layer_index_by_name

    first_hidden_layer_index = get_layer_index_by_name(model, input_layer_name)
    input_shape = model.layers[first_hidden_layer_index].get_input_shape_at(0) # get the input shape of desired layer
    x = input_tensor = tf.keras.Input(shape=input_shape) # a new input tensor to be able to feed the desired layer

    # create the new nodes for each layer in the path
    # For CPH models, include hidden layers excluding the final concatenation
    #     (softmax + event tensor) layer
    if output_layer_index is not None:
        for layer in model.layers[first_hidden_layer_index:output_layer_index]:
            x = layer(x)
    else:
        for layer in model.layers[first_hidden_layer_index:]:
            x = layer(x)

    # create the model
    new_model = tf.keras.Model(input_tensor, x)
    y_pred = new_model.predict(layer_input)
    return y_pred

def metrics_from_dataset(model, model_type, labels, patients, dataset, outcome_names=None, label=None, data_dir=None,
                         num_tiles=0, histogram=False, save_predictions=True, neptune_run=None, pred_args=None, weight_model = None, filter_model = None, filter_thresh = None, onehot_pool = 'true'):

    """Evaluate performance of a given model on a given TFRecord dataset,
    generating a variety of statistical outcomes and graphs.

    Args:
        model (tf.keras.Model): Keras model to evaluate.
        model_type (str): 'categorical', 'linear', or 'cph'.
        labels (dict): Dictionary mapping slidenames to outcomes.
        patients (dict): Dictionary mapping slidenames to patients.
        dataset (tf.data.Dataset): Tensorflow dataset.
        outcome_names (list, optional): List of str, names for outcomes. Defaults to None.
        label (str, optional): Label prefix/suffix for saving. Defaults to None.
        data_dir (str, optional): Path to data directory for saving. Defaults to None.
        num_tiles (int, optional): Number of total tiles expected in the dataset. Used for progress bar. Defaults to 0.
        histogram (bool, optional): Write histograms to data_dir. Defaults to False.
            Takes a substantial amount of time for large datasets, potentially hours.
        save_predictions (bool, optional): Save tile, slide, and patient-level predictions to CSV. Defaults to True.
            May take a substantial amount of time for very large datasets.
        neptune_run (:class:`neptune.Run`, optional): Neptune run in which to log results. Defaults to None.
        pred_args (namespace, optional): Additional arguments to tensorflow and torch backends.

    Returns:
        metrics [dict], accuracy [float], loss [float]
    """

    yt, yp, y_std, t_s, acc, loss, yroi, yunc = eval_from_dataset(model, dataset, model_type, pred_args, num_tiles=num_tiles, weight_model = weight_model, filter_model = filter_model)

    before_metrics = time.time()
    metrics = metrics_from_predictions(y_true=yt,
                                       y_pred=yp,
                                       y_std=y_std,
                                       tile_to_slides=t_s,
                                       labels=labels,
                                       patients=patients,
                                       model_type=model_type,
                                       outcome_names=outcome_names,
                                       label=label,
                                       data_dir=data_dir,
                                       save_predictions=save_predictions,
                                       histogram=histogram,
                                       plot=True,
                                       neptune_run=neptune_run,
                                       y_roi = yroi,
                                       onehot_pool = onehot_pool,
                                       y_unc = yunc,
                                       filter_thresh = filter_thresh)
    after_metrics = time.time()
    log.debug(f'Validation metrics generated, time: {after_metrics-before_metrics:.2f} s')
    return metrics, acc, loss

def permutation_feature_importance(model, dataset, labels, patients, model_type, data_dir, outcome_names=None,
                                   label=None, num_tiles=0, feature_names=None, feature_sizes=None,
                                   drop_images=False, neptune_run=None):

    """Calculate metrics (tile, slide, and patient AUC) from a given model that accepts clinical, slide-level feature
        inputs, and permute to find relative feature performance.

    Args:
        model (str): Path to Tensorflow model.
        dataset (tf.data.Dataset): TFRecord dataset which include three items:
            raw image data, labels, and slide names.
        labels (dict): Dictionary mapping slidenames to outcomes.
        patients (dict): Dictionary mapping slidenames to patients.
        model_type (str): 'categorical', 'linear', or 'cph'.
        data_dir (str): Path to output data directory.
        outcome_names (list, optional): List of str, names for outcomes. Defaults to None.
        label (str, optional): Label prefix/suffix for saving. Defaults to None.
        num_tiles (int, optional): Number of total tiles expected in the dataset. Used for progress bar. Defaults to 0.
        feature_names (list, optional): List of str, names for each of the clinical input features.
        feature_sizes (list, optional): List of int, sizes for each of the clinical input features.
        drop_images (bool, optional): Exclude images (predict from clinical features alone). Defaults to False.
        neptune_run (:class:`neptune.Run`, optional): Neptune run in which to log results. Defaults to None.

    Returns:
        Dictiory of AUCs with keys 'tile', 'slide', and 'patient'
    """

    import tensorflow as tf

    y_true = [] # True outcomes for each tile
    tile_to_slides = [] # Associated slide name for each tile
    pre_hl = [] # Activations pre-hidden layers for each tile
    detected_batch_size = 0
    metrics = {}

    # Establish the output layer for the intermediate model.
    #   This layer is just prior to the hidden layers, and includes
    #   input from clinical features (if present) merged with
    #   post-convolution activations from image data (if present)
    hidden_layer_input = "slide_feature_input" if drop_images else "input_merge"
    intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                               outputs=model.get_layer(hidden_layer_input).output)
    # Setup progress bar
    pb = None
    if log.getEffectiveLevel() <= 20:
        msg = f"Generating model activations at layer '{hidden_layer_input}'..."
        sys.stdout.write(f"\r{msg}")
        if num_tiles:
            pb = ProgressBar(num_tiles,
                             counter_text='images',
                             leadtext=msg,
                             show_counter=True,
                             show_eta=True)

    # Create the time-to-event input used for CPH models
    if model_type == 'cph':
        event_input = tf.keras.Model(inputs=model.input, outputs=model.get_layer("event_input").output)
        events = []

    # For all tiles, calculate the intermediate layer (pre-hidden layer) activations,
    #     and if a CPH model is being used, include time-to-event data
    for i, batch in enumerate(dataset):
        if pb: pb.increase_bar_value(detected_batch_size)
        elif log.getEffectiveLevel() <= 20:
            sys.stdout.write(f"\rGenerating predictions (batch {i})...")
            sys.stdout.flush()
        if not detected_batch_size: detected_batch_size = len(batch[1].numpy())

        tile_to_slides += [slide_bytes.decode('utf-8') for slide_bytes in batch[2].numpy()]
        y_true += [batch[1].numpy()]
        pre_hl += [intermediate_layer_model.predict_on_batch(batch[0])]
        if model_type == 'cph':
            events += [event_input.predict_on_batch(batch[0])]

    # Concatenate arrays
    pre_hl = np.concatenate(pre_hl)
    if model_type == 'cph':
        events = np.concatenate(events)
    y_true = np.concatenate(y_true)
    tile_to_slides = np.array(tile_to_slides)

    if log.getEffectiveLevel() <= 20:
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    # Generate baseline model predictions from hidden layers,
    #     Using the pre-hidden layer activations generated just above.
    #    These baseline predictions should be identical to running
    #     the complete model all at once.
    if model_type == 'cph':
        y_pred = predict_from_layer(model, pre_hl, input_layer_name='hidden_0', output_layer_index=-1)
        y_pred = np.concatenate((y_pred, events), axis = 1)
    else:
        y_pred = predict_from_layer(model, pre_hl, input_layer_name='hidden_0')

    # Generate the AUC, R-squared, and C-index metrics
    #     From the generated baseline predictions.
    base_auc, base_r_squared, base_c_index = metrics_from_predictions(y_true=y_true,
                                                                      y_pred=y_pred,
                                                                      tile_to_slides=tile_to_slides,
                                                                      labels=labels,
                                                                      patients=patients,
                                                                      model_type=model_type,
                                                                      outcome_names=outcome_names,
                                                                      label=label,
                                                                      data_dir=data_dir,
                                                                      histogram=False,
                                                                      plot=False,
                                                                      neptune_run=neptune_run)
    base_auc_list = np.array([base_auc['tile'], base_auc['slide'], base_auc['patient']])
    base_r_squared_list = np.array([base_r_squared['tile'], base_r_squared['slide'], base_r_squared['patient']])
    base_c_index_list = np.array([base_c_index['tile'], base_c_index['slide'], base_c_index['patient']])

    total_features = sum(feature_sizes)
    if model_type == 'cph':
        feature_sizes = feature_sizes[1:]
        feature_names = feature_names[1:]
        total_features -= 1

    if not drop_images:
        feature_names += ["Histology"]

    # For each feature, generate permutation metrics
    curCount = 0
    for i, feature in enumerate(feature_names):
        pre_hl_new = np.copy(pre_hl)

        if feature == "Histology":
            pre_hl_new[:,total_features:] = np.random.permutation(pre_hl_new[:,total_features:])
        else:
            if feature_sizes[i] == 1:
                pre_hl_new[:,curCount] = np.random.permutation(pre_hl_new[:,curCount])
            else:
                pre_hl_new[:,curCount:curCount + feature_sizes[i]] = np.random.permutation(pre_hl_new[:,curCount:curCount + feature_sizes[i]])

            curCount = curCount + feature_sizes[i]

        if model_type == 'cph':
            y_pred = predict_from_layer(model, pre_hl_new, input_layer_name='hidden_0', output_layer_index=-1)
            y_pred = np.concatenate((y_pred, events), axis = 1)
        else:
            y_pred = predict_from_layer(model, pre_hl_new, input_layer_name='hidden_0')

        new_auc, new_r, new_c = metrics_from_predictions(y_true=y_true,
                                                y_pred=y_pred,
                                                tile_to_slides=tile_to_slides,
                                                labels=labels,
                                                patients=patients,
                                                model_type=model_type,
                                                outcome_names=outcome_names,
                                                label=None, #label[i] ?
                                                data_dir=data_dir,
                                                histogram=False,
                                                plot=False,
                                                neptune_run=neptune_run)

        if model_type == 'categorical':
            metrics[feature] = base_auc_list - np.array([new_auc['tile'], new_auc['slide'], new_auc['patient']])
        if model_type == 'linear':
            metrics[feature] = base_r_squared_list - np.array([new_r['tile'], new_r['slide'], new_r['patient']])
        if model_type == 'cph':
            metrics[feature] = base_c_index_list - np.array([new_c['tile'], new_c['slide'], new_c['patient']])

    #Probably makes sense to measure only at the tile level - unless we write code to do permutation
    # of patient level data which would be probably more work than its worth
    feature_text = ""
    for feature in feature_names:
        if model_type == 'categorical':
            feature_text += feature + ": " + str(metrics[feature][0][0]) + ", "
        else:
            feature_text += feature + ": " + str(metrics[feature][0]) + ", "
    log.info("Feature importance, tile level: " + feature_text)

    combined_metrics = {
        'auc': base_auc,
        'r_squared': base_auc,
        'c_index': base_c_index
    }

    return combined_metrics