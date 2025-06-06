# -*- coding: utf-8 -*-
# +
import os, json
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np

from PIL import Image, ImageFile
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.utils import to_categorical
# -

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(object):
    def __init__(self, config):
        with open(os.path.dirname(os.path.abspath(__file__)) + '/dataset_config.json', 'r') as f:
            config = json.load(f)

        # Dataset configuration
        self.dataset_path = config['dataset_path']
        self.image_shape = config['image_shape']
        self.pixel_depth = config['pixel_depth']
        self.labels = config['labels']
        self.label_names = {label: idx for idx, label in enumerate(config["labels"])}
        self.val_perc = config['val_percentage']
        self.undersampling = config['undersampling']

        self.df = None

        self.X_train = []
        self.y_train = []
        self.X_val = []
        self.y_val = []
        
        self.weight_train = []
        self.weight_val = []

    def load_alzheimer_df(self, verbose=0):
        """
        Load and filter dataframe with the dataset information

        :param int verbose: Whether to show information about the dataframe
        """
        
        print('-----GENERATING DATAFRAME-----')
        # Load data from .csv
        filename = os.path.join(self.dataset_path, 'alzheimer_dataset_undersampling.csv')
        self.df = pd.read_csv(filename)

        # Select images within de label list and only one occurrence
        self.df = self.df[self.df['Labels'].notna()]

        self.df['Labels'] = self.df['Labels'].apply(lambda x: filter_values(x, self.labels))
        self.df = self.df[self.df['Labels'].apply(len) == 1]
        self.df['Labels'] = self.df['Labels'].apply(lambda x: str(x))
        
        # Undersampling labels
        if self.undersampling:
            min_class = np.min(self.df['Labels'].value_counts().values)
            self.df = self.df.groupby('Labels').apply(lambda x: x.sample(n=min_class)).reset_index(drop=True)

        # Randomize list
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

        if verbose == 1:
            print('\n---IMAGE LABELS---\n' + str(self.df['Labels'].value_counts(dropna=False)))
            
    def load_data(self, verbose=0):
        """
        Load dataset images and labels in the Dataset object

        :param int verbose: Whether to show information about the process
        """

        self.load_alzheimer_df(verbose=verbose)

        print('\n-----LOADING DATA FROM DF-----')
        # Create list of image path and labels
        imgs_list = []
        label_list = []
        weight_list = []

        print('# Loading images')
        pbar = tqdm(enumerate(self.df.itertuples()))
        total_imgs = self.df.shape[0]
        for i, row in pbar:
            img_path = os.path.join(self.dataset_path, str(row.ImageDir), row.ImageID)
            pbar.set_description("Procesando " + str(i) + "/" + str(total_imgs))
            img = load_img_from_path(img_path, self.image_shape)
            imgs_list.append(img)

            row_list = eval(row.Labels)  
            label = row_list[0] if len(row_list) > 0 else None
            
            if label in self.label_names:
                label_list.append(self.label_names[label])
            else:
                label_list.append(None)

        # Normalize images within the range [-1, 1]
        print('# Normalizing image pixels')
        imgs_list = np.array(imgs_list) / (self.pixel_depth / 2) - 1
    
        # Save class weights
        print('# Loading class weights')
        total_samples = len(self.df['Labels'])
        dict_weights = {}
        for label, n_samples in self.df['Labels'].value_counts(dropna=False).items():
            dict_weights[eval(label)[0]] = total_samples/(n_samples*2)
            
        weight_list = [dict_weights[self.labels[elem]] for elem in label_list]
        
        print('# Splitting dataset')
        self.X_train, self.X_val = split_list(imgs_list, self.val_perc)
        self.y_train, self.y_val = split_list(label_list, self.val_perc)
        self.weight_train, self.weight_val = split_list(weight_list, self.val_perc)
        
        self.X_train = np.expand_dims(self.X_train, axis=-1)
        self.X_val = np.expand_dims(self.X_val, axis=-1)
        
        # One-hot encoding labels
        num_classes = len(self.label_names)
        self.y_train = to_categorical(self.y_train, num_classes)
        self.y_val = to_categorical(self.y_val, num_classes)
        
        if verbose == 1:
            print("\tDATA PARTITION\n")
            print("X_train: " + str(np.shape(self.X_train)))
            print("y_train: " + str(np.shape(self.y_train)))
            print("X_val: " + str(np.shape(self.X_val)))
            print("y_val: " + str(np.shape(self.y_val)))


# Convert dataframe columns if None to 0
def convert_dtype(x):
    if str(x)=='None':
        return 0
    else:
        return x


# Load, resize and transform to np.array an image from its path
def load_img_from_path(path, img_shape):
    img = Image.open(path).convert('L')
    img = img.resize((img_shape[0], img_shape[1]), Image.BILINEAR)
    img_array = np.array(img)
    img.close()

    return img_array


# Delete rows that include labels not within the list
def filter_values(row, label_list):
    unique_values = set(eval(row))

    valid_values = unique_values.intersection(set(label_list))
    new_row = np.array(list(valid_values))

    return new_row


# Split a list with a specific percentage
def split_list(list_data, val_perc):
    train_split = np.array(list_data[int(val_perc * len(list_data)):])
    val_split = np.array(list_data[:int(val_perc * len(list_data))])
    
    return train_split, val_split
