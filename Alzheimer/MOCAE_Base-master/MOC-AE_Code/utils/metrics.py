# -*- coding: utf-8 -*-
# +
import os, statistics, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

from PIL import Image


# -

def plot_reconstruction(n_plots, original_imgs, reconstructed_img_predictions, batch, epoch, out_path):
    """
    Plot reconstruction results for a train step

    :param int n_plots: Number of cases
    :param list original_imgs: Original images
    :param list reconstructed_img_predictions: Reconstructed images
    :param int batch: Number of batch of the training
    :param int epoch: Number of epoch of the training
    :param string out_path: Output path to save the plot
    """
    
    # Define figure
    f, axarr = plt.subplots(n_plots, 2, figsize=(8, 3*n_plots))
    plt.suptitle('Epoch: ' + str(epoch) + ', Batch: ' + str(batch), fontsize=16)

    axarr[0,0].set_title('Source image')
    axarr[0,1].set_title('Reconstructed image')

    # Plot each reconstruction
    for i in range(n_plots):
        axarr[i,0].imshow(original_imgs[i,:,:,0], cmap='gray')
        axarr[i,1].imshow(reconstructed_img_predictions[i,:,:,0], cmap='gray')

        axarr[i,0].get_xaxis().set_visible(False)
        axarr[i,0].get_yaxis().set_visible(False)
        axarr[i,1].get_xaxis().set_visible(False)
        axarr[i,1].get_yaxis().set_visible(False)

    # Save figure
    plt.savefig(out_path + 'reconstructions/e' + str(epoch).zfill(3) + 'b'
                + str(batch).zfill(5) + '_reconstructed_image.png')
    plt.show()


def plot_train_val(history, loss_name='', out_path=None):
    """
    Plot evolution of a loss during the training

    :param dict history: Historic dictionary of the losses
    :param str loss_name: Prefix of the loss, it will be appended to 'loss'
    :param string out_path: Output path to save the plot
    """
    
    # X axis definition
    x_axis = range(len(history[loss_name + 'loss']))
    x_axis_mean = range(0, len(history[loss_name + 'loss_mean']))

    # Define figure
    plt.rcParams['figure.figsize'] = [20, 5]
    f, ax = plt.subplots(1, 2, sharex=False, sharey=False)
    f.suptitle(loss_name)

    # Plot complete loss figure
    ax[0].plot(x_axis, history[loss_name + 'loss'], label='Train Loss')
    ax[0].plot(x_axis, history['val_' + loss_name + 'loss'], label='Validation Loss')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('iteration')
    ax[0].legend(loc="upper right")
    ax[0].grid()

    # Plot boxplot for each epoch figure
    ax[1].plot(x_axis_mean, history[loss_name + 'loss_mean'], label='Mean loss')
    ax[1].plot(x_axis_mean, history[loss_name + 'loss_val_mean'],
               label='Validation mean loss')
    ax[1].boxplot(history[loss_name + 'loss_epoch'], positions=x_axis_mean,
                  showfliers=False)
    ax[1].set_title('Mean Loss')
    ax[1].set_xlabel('epoch')
    ax[1].legend(loc="upper right")
    ax[1].grid()

    # Save figure
    plt.savefig(out_path + loss_name + 'train_losses.png')
    plt.show()


def get_mean_loss(loss_name, n_batches, history):
    """
    Update historic mean value of losses in divided by epochs

    :param str loss_name: Complete name of the loss
    :param int n_batches: Number of batches in each epoch
    :param dict history: Historic dictionary of the losses
    """
    
    epoch_loss = np.split(np.array(history[loss_name]),
                          int(len(np.array(history[loss_name]))/n_batches))
    mean_loss = get_mean(history[loss_name], n_batches)
    val_mean_loss = get_mean(history['val_'+ loss_name], n_batches)

    history[loss_name + '_epoch'] = epoch_loss
    history[loss_name + '_mean'] = mean_loss
    history[loss_name + '_val_mean'] = val_mean_loss


def get_mean(loss, n_batches):
    """
    Get mean value of a loss divided by epochs

    :param list loss: Loss values during the training divided by batches
    :param int n_batches: Number of batches in each epoch
    :return: List of mean values of losses in each epoch
    """
    num_epochs = int(len(loss)/n_batches)
    mean_loss = []
    
    for i in range(num_epochs):
        mean_loss.append(statistics.mean(loss[i*n_batches:(i+1)*n_batches]))
        
    return mean_loss


def conf_matrix(y_real, y_pred, names, batch, epoch, out_path, best=False):
    """
    Plot confusion matrix for a train step
    
    :param list y_real: True labels
    :param list y_pred: Predicted labels
    :param list names: Label names of the classification
    :param int batch: Number of batch of the training
    :param int epoch: Number of epoch of the training
    :param string out_path: Output path to save the plot
    :param bool best: If True, the image is saved in a specific path
    """
    
    f, ax = plt.subplots(2, 1, figsize=(7, 14))
    f.suptitle('Epoch: ' + str(epoch) + ', Batch: ' + str(batch), fontsize=16)

    conf_matrix = confusion_matrix(y_real, y_pred)
    sns.heatmap(conf_matrix, annot=True,
                cmap='Blues', fmt = 'd', ax=ax[0], cbar=False)
    
    ax[0].xaxis.set_ticklabels(names)
    ax[0].yaxis.set_ticklabels(names)

    ax[1].set_xlim((0,1))
    ax[1].set_ylim((0,1))
    ax[1].text(0.5, 0.5, classification_report(y_real, y_pred, target_names=names),
               ha='center', va='center')
    ax[1].axis('off')

    # Save figure
    if best==True:
        plt.savefig(out_path + 'conf_matrix/best_confusion_matrix.png')
    else:
        plt.savefig(out_path + 'conf_matrix/e' + str(epoch).zfill(3) + 'b'
                    + str(batch).zfill(5) + '_confusion_matrix.png')
    plt.show()


def classes_conf_matrix(y_real, y_pred, names, batch, epoch, out_path, best=False):
    """
    Plot binary confusion matrices for each label for a train step

    :param list y_real: True labels
    :param list y_pred: Predicted labels
    :param list names: Label names of the classification
    :param int batch: Number of batch of the training
    :param int epoch: Number of epoch of the training
    :param string out_path: Output path to save the plot
    :param bool best: If True, the image is saved in a specific path
    """
    
    n_classes= len(y_real[0])
    # Define figure
    f, ax = plt.subplots(2, n_classes, figsize=(5*n_classes, 10))
    f.suptitle('Epoch: ' + str(epoch) + ', Batch: ' + str(batch), fontsize=16)

    # Plot conf matrix for each class
    for i in range(n_classes):
        ax[0, i].set_title(names[i])
        plot_conf_matrix(ax[0, i], np.array(y_real)[:,i], np.array(y_pred)[:,i])

        ax[1, i].set_xlim((0,1))
        ax[1, i].set_ylim((0,1))
        ax[1, i].text(0.5, 0.5, classification_report(y_real[:,i], y_pred[:,i]), ha='center', va='center')
        ax[1, i].axis('off')

    # Save figurea
    plt.savefig(out_path + 'conf_matrix/e' + str(epoch).zfill(3) + 'b'
                + str(batch).zfill(5) + '_confusion_matrix.png')
    plt.show()


def plot_conf_matrix(ax, y, y_pred):
    """
    Plot confusion matrix using seaborn
    
    :param axes ax: Axes to fill with the confusion matrix plot
    :param list y: True labels
    :param list y_pred: Predicted labels
    :param bool best: If True, the image is saved in a specific path
    :return: Confusion matrix plot
    """
    
    conf_matrix = confusion_matrix(y, y_pred)
    return sns.heatmap(conf_matrix, annot=True,
                cmap='Blues', fmt = 'd', ax=ax, cbar=False)
