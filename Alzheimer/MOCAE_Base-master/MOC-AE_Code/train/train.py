# -*- coding: utf-8 -*-
# +
import sys
sys.path.append("..")

import json, random, os

from utils.metrics import *

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow import device


# -

with open(os.path.dirname(os.path.abspath(__file__)) + '/train_config.json', 'r') as f:
    config = json.load(f) 

def train(dataset, out_path, model, model_type, gamma=None):
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    conf_mat_samples = config['conf_mat_samples']
    
    min_val_mean_loss = 999
    
    history = {
        'loss' : [],
        'val_loss' : [],
        'loss_mean' : [],
        'loss_val_mean' : [],
        'loss_epoch' : [],

        'rec_loss' : [],
        'val_rec_loss' : [],
        'rec_loss_mean' : [],
        'rec_loss_val_mean' : [],
        'rec_loss_epoch' : [],

        'class_loss' : [],
        'val_class_loss' : [],
        'class_loss_mean' : [],
        'class_loss_val_mean' : [],
        'class_loss_epoch' : [],
    }
    
    print('-----COMPILING MODEL-----')
    model_config = {
        "mocae": {
            "losses_fun": {"rec": "mse", "class": "categorical_crossentropy"},
            "loss_weights": {"rec": 1, "class": gamma}
        },
        "mocvae": {
            "losses_fun": {"rec": "mse", "class": "categorical_crossentropy"},
            "loss_weights": {"rec": 1, "class": gamma}
        },
        "ae": {
            "losses_fun": {"rec": "mse"},
            "loss_weights": {"rec": 1}
        },
        "classifier": {
            "losses_fun": {"class": "categorical_crossentropy"},
            "loss_weights": {"class": 1}
        }
    }

    if model_type not in model_config:
        raise Exception("Model type not recognized")
    
    model.mocae.compile(
        loss=model_config[model_type]['losses_fun'],
        loss_weights=model_config[model_type]['loss_weights'],
        optimizer=Adam(learning_rate=config['learning_rate'])
    )
                    
    print('-----TRAIN START-----')
    with device('/GPU:0'):
        for epoch in range(n_epochs):
            dataset.X_train, dataset.y_train, dataset.weight_train = \
            shuffle_three_arrays(dataset.X_train, dataset.y_train, dataset.weight_train)
            dataset.X_val, dataset.y_val, dataset.weight_val = \
            shuffle_three_arrays(dataset.X_val, dataset.y_val, dataset.weight_val)

            n_batches = int(len(dataset.X_train) / batch_size)
            for batch in range(n_batches):
                X_imgs = dataset.X_train[batch*batch_size:(batch+1)*batch_size]
                y_labels = dataset.y_train[batch*batch_size:(batch+1)*batch_size]
                sample_weight = dataset.weight_train[batch*batch_size:(batch+1)*batch_size]

                if model_type == "mocae" or model_type == "mocvae":
                    losses = model.mocae.train_on_batch(x=X_imgs,
                                                     y={"rec": X_imgs, "class": y_labels},
                                                     return_dict=True,
                                                     sample_weight={"class": sample_weight})
                elif model_type == "classifier":
                    losses = model.mocae.train_on_batch(x=X_imgs,
                                                     y={"class": y_labels},
                                                     return_dict=True,
                                                     sample_weight={"class": sample_weight})
                elif model_type == "ae":
                    losses = model.mocae.train_on_batch(x=X_imgs,
                                                     y={"rec": X_imgs},
                                                     return_dict=True)
                
                history['loss'].append(losses['loss'])
                if 'rec_loss' in losses:
                    history['rec_loss'].append(losses['rec_loss'])
                if 'class_loss' in losses:
                    history['class_loss'].append(losses['class_loss'])

                # Get a random validation batch
                idx = np.random.randint(0, len(dataset.X_val) / batch_size)
                X_imgs = dataset.X_val[idx*batch_size:(idx+1)*batch_size]
                y_labels = dataset.y_val[idx*batch_size:(idx+1)*batch_size]
                sample_weight = dataset.weight_val[idx*batch_size:(idx+1)*batch_size]

                if model_type == "mocae" or model_type == "mocvae":
                    val_losses = model.mocae.test_on_batch(x=X_imgs,
                                                           y={"rec": X_imgs, "class": y_labels},
                                                           return_dict=True,
                                                           sample_weight={"class": sample_weight})
                
                elif model_type == "classifier":
                    val_losses = model.mocae.test_on_batch(x=X_imgs,
                                                           y={"class": y_labels},
                                                           return_dict=True,
                                                           sample_weight={"class": sample_weight})
                elif model_type == "ae":
                    val_losses = model.mocae.test_on_batch(x=X_imgs,
                                                           y={"rec": X_imgs},
                                                           return_dict=True)
                    
                history['val_loss'].append(val_losses['loss'])
                if 'rec_loss' in val_losses:
                    history['val_rec_loss'].append(val_losses['rec_loss'])
                if 'class_loss' in val_losses:
                    history['val_class_loss'].append(val_losses['class_loss'])

            # Reconstruction evaluation
            n_plots = 3
            if model_type == "ae" or model_type == "mocae" or model_type == "mocvae":
                original_imgs = np.expand_dims(dataset.X_val[:n_plots], axis=-1)
                plot_reconstruction(n_plots, original_imgs,
                                    model.autoencoder.predict(dataset.X_val[:n_plots]),
                                    batch, epoch, out_path)

            # Confusion matrix evaluation
            if model_type in ["classifier", "mocae", "mocvae"]:
                    y_real = dataset.y_val[:conf_mat_samples]
                    lat_space = model.encoder.predict(dataset.X_val[:conf_mat_samples])

                    # Ajustar la salida del encoder según el tipo de modelo
                    if model_type == "mocvae":
                        z_mean, z_log_sigma, z = lat_space
                        lat_space = z  

                    # Predicción con el clasificador
                    y_pred = model.classifier.predict(lat_space)

                    # Convertimos las etiquetas reales y predichas en índices
                    if len(y_real.shape) > 1:
                        y_real = np.argmax(y_real, axis=1)
                    else:
                        y_real = y_real  

                    if len(y_pred.shape) > 1:
                        y_pred = np.argmax(y_pred, axis=1)
                    else:
                        y_pred = y_pred  

                    # Generación de la matriz de confusión
                    conf_matrix(y_real, y_pred, dataset.labels, batch, epoch, out_path, best=True)

            get_mean_loss('loss', n_batches, history)
            plot_train_val(history, '', out_path)
            if model_type == "mocae" or model_type == "mocvae":
                get_mean_loss('rec_loss', n_batches, history)
                plot_train_val(history, 'rec_', out_path)
                get_mean_loss('class_loss', n_batches, history)
                plot_train_val(history, 'class_', out_path)

            # Save best model and conf matrix (validation total loss)
            if history["loss_val_mean"][-1] < min_val_mean_loss:
                min_val_mean_loss = history["loss_val_mean"][-1]
                
                model.encoder.save(out_path + 'models/e_best_encoder.h5')
                if model_type == "ae" or model_type == "mocae" or model_type == "mocvae":
                    model.decoder.save(out_path + 'models/e_best_decoder.h5')
                    model.autoencoder.save(out_path + 'models/e_best_autoencoder.h5')
                if model_type == "classifier" or model_type == "mocae" or model_type == "mocvae":
                    model.classifier.save(out_path + 'models/e_best_classifier.h5')
                if model_type == "mocae" or model_type == "mocvae":
                    model.mocae.save(out_path + 'models/e_best_mocae.h5')
                
                if model_type in ["classifier", "mocae", "mocvae"]:
                    y_real = dataset.y_val[:conf_mat_samples]
                    lat_space = model.encoder.predict(dataset.X_val[:conf_mat_samples])

                    # Ajustar la salida del encoder según el tipo de modelo
                    if model_type == "mocvae":
                        z_mean, z_log_sigma, z = lat_space
                        lat_space = z  

                    # Predicción con el clasificador
                    y_pred = model.classifier.predict(lat_space)

                    # Convertimos las etiquetas reales y predichas en índices
                    if len(y_real.shape) > 1:
                        y_real = np.argmax(y_real, axis=1)
                    else:
                        y_real = y_real  

                    if len(y_pred.shape) > 1:
                        y_pred = np.argmax(y_pred, axis=1)
                    else:
                        y_pred = y_pred  

                    # Generación de la matriz de confusión
                    conf_matrix(y_real, y_pred, dataset.labels, batch, epoch, out_path, best=True)

            # Saving models
            if epoch%config['model_save_pace'] == 0 and epoch!=0:
                model.encoder.save(out_path + 'models/e' + str(epoch).zfill(3) + '_encoder.h5')
                if model_type == "ae" or model_type == "mocae" or model_type == "mocvae":
                    model.decoder.save(out_path + 'models/e' + str(epoch).zfill(3) + '_decoder.h5')
                    model.autoencoder.save(out_path + 'models/e' + str(epoch).zfill(3) + '_autoencoder.h5')
                if model_type == "classifier" or model_type == "mocae" or model_type == "mocvae":
                    model.classifier.save(out_path + 'models/e' + str(epoch).zfill(3) + '_classifier.h5')
                if model_type == "mocae" or model_type == "mocvae":
                    model.mocae.save(out_path + 'models/e' + str(epoch).zfill(3) + '_mocae.h5')


def shuffle_three_arrays(a, b, c):
    combined = list(zip(a, b, c))
    random.shuffle(combined)
    a_permuted, b_permuted, c_permuted = zip(*combined)
    
    return np.array(a_permuted), np.array(b_permuted), np.array(c_permuted)
