# -*- coding: utf-8 -*-
# +
import cv2
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance
from skimage import exposure

from statistics import mean, stdev


# -

def get_nn_padchest(latent_representations, img_index):
    """
    Get list of euclidean distances and indexes given an index

    :param list latent_representations: List of the image descriptors
    :param index img_index: Index of the query from which to compare
    :return: list of lists with the distance and indexes ordered by distance
             from the query
    """
    
    source_z = latent_representations[img_index]

    indexes = []
    distances = []

    for i, candidate in enumerate(latent_representations):
        if i != img_index:
            distances.append(euclidean(source_z, candidate))
            indexes.append(i)

    neighbors = zip(distances, indexes)
    neighbors = sorted(neighbors, key=lambda x: x[0])
    neighbors = list(neighbors)
    
    return neighbors


def get_recom_results(dataset, model, n_imgs, n_neighbors, vae=False, label=None):
    """
    Get list of query and recommended cases

    :param Dataset dataset: Dataset containing the cases. Mandatory to contain
           X_val, y_val and labels
    :param Keras Model model: Image descriptor model
    :param int n_imgs: Number of queries processed
    :param int n_neighbors: Top k of neighbors retrieved in the recommendation
    :param bool vae: Flag for using VAE factorization
    :param str label: If label, only use as query images of this label
    :return: list of lists with the query and recommended labels
    """
    i = 0
    results = []
    lat_rep = model.predict(dataset.X_val)
    
    # Use the z mean for VAE models, in other case use all z
    if vae:
        lat_rep = lat_rep[0]
    
    pbar = tqdm(total=n_imgs, position=0, leave=True)
    while i<n_imgs and i<len(dataset.X_val):
        # If label is specified, search cases with this label
        if label:
            img_label = None
            while img_label == label and i<len(dataset.X_val):
                img_label = dataset.labels[np.argmax(dataset.y_val[i])]
                if img_label != label:
                    i+=1
                
        img_label = dataset.labels[np.argmax(dataset.y_val[i])]
        neighbor_list = get_nn_padchest(lat_rep, img_index=i)

        labels_neigbor = []
        for j in range(n_neighbors):
            labels_neigbor.append(dataset.labels[np.argmax(dataset.y_val[neighbor_list[j][1]])])

        results.append([img_label, labels_neigbor])
        pbar.update(1)

        i+=1
        
    pbar.close()
    return results


def get_distance_results(data, model, n_imgs, n_neighbors, verbose=False, vae=False):
    """
    Get list of distances between query and recommendations

    :param Dataset dataset: Dataset containing the cases. Mandatory to contain
           X.val, y_val and labels
    :param Keras Model model: Image descriptor model
    :param int n_imgs: Number of queries processed
    :param int n_neighbors: Top k of neighbors retrieved in the recommendation
    :param bool vae: Flag for using VAE factorization
    :return: list of lists with distances between the query and recommendations
    """
    
    i = 0
    results = []    
    lat_rep = model.predict(data.X_val)
    
    # Use the z mean for VAE models, in other case use all z
    if vae:
        lat_rep = lat_rep[0]
    
    pbar = tqdm(total=n_imgs, position=0, leave=True)
    while i<n_imgs and i<len(data.X_val):
        neighbor_list = get_nn_padchest(lat_rep, img_index=i)
            
        distances_neigbor = []
        for j in range(n_neighbors):
            distances_neigbor.append(wass_distance(data.X_val[i],
                                                   data.X_val[neighbor_list[j][1]]))

        results.append(distances_neigbor)
        pbar.update(1)
        
        i+=1
        
    pbar.close()
    return results


def get_recom_metric(results, k):
    """
    Get accuracy and standard deviation from the recommendation list using the k
    nearest neighbors

    :param list results: List containing query and recommendations labels
           for different cases
    :param int k: Number of neighbors used in the calculation
    :return: tuple with precision@k and standard deviation of the recommendations
    """
    pred_acc = []

    for elem in results:
        y_real = elem[0]
        y_pred = elem[1][:k]

        correct_pred = y_pred.count(y_real)
        pred_acc.append((correct_pred / len(y_pred)) * 100)

    return np.mean(pred_acc), np.std(pred_acc)


def get_dist_metric(results, k):
    """
    Get mean and standard deviation from the distances list

    :param list results: List containing distances between different k neighbors
    :param int k: Number of neighbors used in the calculation
    :return: tuple with mean distance and standard deviation of the distances
    """
    pred_acc = []

    for elem in results:
        pred_acc.append(mean(elem[:k]))

    return mean(pred_acc), stdev(pred_acc)


def sliced_wasserstein(X, Y, num_proj):
    # Code adapted from https://gist.github.com/smestern/ba9ee191ca132274c4dfd6e1fd6167ac
    """
    Gets sliced Wasserstein distance between two arrays
    :param ndarray X: 2d (or nd) histogram
    :param ndarray Y: 2d (or nd) histogram
    :param int num_proj: Number of random projections to compute the mean over
    :return: mean_emd_dist'''
    """
    
    dim = X.shape[1]
    ests = []
    for _ in range(num_proj):
        # sample uniformly from the unit sphere
        dir_proj = np.random.rand(dim)
        dir_proj /= np.linalg.norm(dir_proj)

        # project the data
        X_proj = X @ dir_proj
        Y_proj = Y @ dir_proj

        # compute 1d wasserstein
        ests.append(wasserstein_distance(X_proj, Y_proj))
    return np.mean(ests)


def wass_distance(img1, img2):
    """
    Get Sliced Wasserstein distance between two images after equalizing them

    :param ndarray img1: Image to be compared
    :param ndarray img2: Image to be compared
    :return: distance between img1 and img2 after applying histogram equalization
             to both images
    """
    width, height = np.shape(img1)[0], np.shape(img1)[1]
    
    img1 = cv2.resize(img1, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    img1_eq = exposure.equalize_hist(img1)
    
    img2 = cv2.resize(img2, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    img2_eq = exposure.equalize_hist(img2)
    
    return sliced_wasserstein(img1_eq[:,:], img2_eq[:,:], 100)
