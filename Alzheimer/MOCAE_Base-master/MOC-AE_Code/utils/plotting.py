# -*- coding: utf-8 -*-
# +
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from utils.recommendation import *


# -

def plot_nneighbors(dataset, model, n_neighbors=5, n_cases=5, vae=False,
                    label=None, save_path=None, show_age=False, show_sex=False,
                    show_ids=False, show_dist=False, dpi=40):
    lat_rep = model.predict(dataset.X_train)
    """
    Plot n_neighbors neareast neighbors for the first n_cases patients of train partition

    :param Dataset dataset: Dataset containing the cases. Mandatory to contain
           X_train, y_train and labels. Optional to include study_id_train, study_id_train,
           age_train and sex_train
    :param int n_neighbors: Number of neighbors shown in the plot columns
    :param int n_cases: Number of cases plotted in the rows
    :param bool vae: Whether the model has Variational Autoencoder backbone
    :param str label: If specified, only query cases of the specified label are shown
    :param str save_path: If specified, The plot is saved in the path
    :param bool show_age: Whether to show the age of the patients
    :param bool show_sex: Whether to show the sex of the patients
    :param bool show_ids: Whether to show the ids for the patient and study (last 3 digits)
    :param bool show_dist: Whether to show the Sliced Wasserstein distance between the
           and the cases
    :param int dpi: Dots per inch of the plot
    """
    
    f, ax = plt.subplots(n_cases, n_neighbors + 1, figsize=(9*n_neighbors, 9*n_cases), dpi=dpi)
    
    cont=0
    i=0
    while cont < n_cases and i<len(dataset.X_train):
        # If label is specified, search cases with this label
        if label:
            img_label = None
            while img_label != label and i<len(dataset.X_train):
                img_label = dataset.labels[np.argmax(dataset.y_train[i])]
                if img_label != label:
                    i+=1
                
        # Use the z mean for VAE models, in other case use all z
        if vae:
            neighbor_list = get_nn_padchest(lat_rep[0], img_index=i)
        else:
            neighbor_list = get_nn_padchest(lat_rep, img_index=i)
                
        plot_case(ax[cont], neighbor_list, dataset, i, n_neighbors,
                  show_age, show_sex, show_ids, show_dist)
        cont+=1
        i+=1
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_case(ax, neighbors, dataset, img_idx, n_neighbors,
              show_age=False, show_sex=False, show_ids=False, show_dist=False):
    """
    Get one row with one query and its corresponding set of recommendations

    :param axes ax: Row of the complete plot to fill with the case
    :param Dataset dataset: Dataset containing the cases. Mandatory to contain
           X_train, y_train and labels. Optional to include study_id_train, study_id_train,
           age_train and sex_train    :param int n_cases: Number of cases plotted in the rows
    :param int img_idx: Position of the query image in the training split
    :param int n_neighbors: Number of neighbors shown in the plot columns
    :param bool show_age: Whether to show the age of the patients
    :param bool show_sex: Whether to show the sex of the patients
    :param bool show_ids: Whether to show the ids for the patient and study (last 3 digits)
    :param bool show_dist: Whether to show the Sliced Wasserstein distance between the
           and the cases
    :return: Axes filled with the query and recommendations
    """
    
    title = ''
    if show_ids==True:
        title += 'Study ID: ' + str(dataset.study_id_train[img_idx])[:3]
        title += '\nPatient ID: ' + str(dataset.patient_id_train[img_idx])[:3]
    ax[0].set_title(title + '\nQuery image')
    ax[0].imshow(dataset.X_train[img_idx,:,:], cmap='gray', vmin=-1, vmax=1)
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_xticks([])
    
    x_label_str = dataset.labels[np.argmax(dataset.y_train[img_idx])]
    if show_age != False:
        x_label_str = x_label_str + '\nAge: ' + str(dataset.age_train[img_idx])
    if show_sex != False:
        x_label_str = x_label_str + '\nSex: ' + dataset.sex_train[img_idx]
    ax[0].set_xlabel(x_label_str)

    for i in range(n_neighbors):
        ax[i+1].imshow(dataset.X_train[neighbors[i][1],:,:], cmap='gray', vmin=-1, vmax=1)
        ax[i+1].get_yaxis().set_visible(False)
        ax[i+1].set_xticks([])
        
        title = ''
        if show_ids==True:
            title += 'Study ID: ' + str(dataset.study_id_train[neighbors[i][1]])[:3]
            title += '\nPatient ID: ' + str(dataset.patient_id_train[neighbors[i][1]])[:3]
            
        if show_dist==True:
            title += '\nDist: '+ str(wass_distance(dataset.X_train[img_idx,:,:],
                                                   dataset.X_train[neighbors[i][1],:,:]))
        
        ax[i+1].set_title(title + '\nk= ' + str(i) + ' neighbor')
        
        x_label_str = dataset.labels[np.argmax(dataset.y_train[neighbors[i][1]])]
        if show_age != False:
            x_label_str = x_label_str + '\nAge: ' + str(dataset.age_train[neighbors[i][1]])
        if show_sex != False:
            x_label_str = x_label_str + '\nSex: ' + dataset.sex_train[neighbors[i][1]]
            
        ax[i+1].set_xlabel(x_label_str)
