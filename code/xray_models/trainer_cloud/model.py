import os
import shutil

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Softmax)
import tensorflow_hub as hub

from .config import config
from .import preprocessing

def get_layers(model_type, 
               nclasses=15, 
               hidden_layer_1_neurons=400,
               hidden_layer_2_neurons=100,
               dropout_rate=0.25,
               num_filters_1=64,
               kernel_size_1=3,
               pooling_size_1=2,
               num_filters_2=32,
               kernel_size_2=3,
               pooling_size_2=2):
    """
    Get model layers for a specific model
    """
    model_layers = {
        'cnn':[
            Conv2D(num_filters_1, kernel_size=kernel_size_1,
                  activation='relu', input_shape=(config.IMG_WIDTH, config.IMG_HEIGHT, 1)),
            MaxPooling2D(pooling_size_1),
            Conv2D(num_filters_2, kernel_size=kernel_size_2,
                  activation='relu'),
            MaxPooling2D(pooling_size_2),
            Flatten(),
            Dense(hidden_layer_1_neurons, activation='relu'),
            Dense(hidden_layer_2_neurons, activation='relu'),
            Dropout(dropout_rate),
            Dense(nclasses, activation='sigmoid')
        ],
        'vision_transformer':
        [hub.KerasLayer("https://tfhub.dev/sayakpaul/vit_b16_fe/1", trainable=False),
        Dense(nclasses, activation='sigmoid')
        ],
        'inception_resnet':[
        hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/5",trainable=False),
        Dense(nclasses, activation='sigmoid')]
        
    }
    return model_layers[model_type]

def label_weighted_cross_entropy(y_true, y_pred):
    """
    Loss 
    """
    P = tf.reduce_sum(y_true)
    N = -1 * tf.reduce_sum(y_true - 1)
    
    beta_P = tf.cast((P + N) / P, dtype=tf.float64)
    beta_N = tf.cast((P + N) / N, dtype=tf.float64)
    
    y_true = tf.cast(y_true, dtype=tf.float64)
    y_pred = tf.cast(y_pred, dtype=tf.float64)
    
    epsilon = tf.constant(1e-7, dtype=tf.float64) #avoid nans
    loss = (beta_P*tf.math.log(y_pred+epsilon)*y_true + beta_N*tf.math.log((1-y_pred)+epsilon) * (1-y_true))*-1.0
    tf.debugging.assert_all_finite(loss, 'There are nan values')
    return tf.reduce_sum(tf.reduce_mean(loss, axis = 0))


class ClassImbalanceSparsityAdjustedLoss(tf.keras.losses.Loss):
    def __init__(self, inverse_class_weights):
        """
        Initialization of inverse class weights
        """
        super().__init__(name = 'ClassImbalanceSparsityAdjustedLoss')
        self.inverse_class_weights = inverse_class_weights
    
    def call(self, y_true, y_pred):
        """
        Cross entropy loss adjusted for class imabalance and one-hot encoding sparsity
        """
        P = tf.reduce_sum(y_true)
        N = -1 * tf.reduce_sum(y_true - 1)

        beta_P = tf.cast((P + N) / P, dtype=tf.float64)
        beta_N = tf.cast((P + N) / N, dtype=tf.float64)

        y_true = tf.cast(y_true, dtype=tf.float64)
        y_pred = tf.cast(y_pred, dtype=tf.float64)

        epsilon = tf.constant(1e-7, dtype=tf.float64) #avoid nans
        loss = (beta_P*tf.math.log(y_pred+epsilon)*y_true + beta_N*tf.math.log((1-y_pred)+epsilon) * (1-y_true))*-1.0
        tf.debugging.assert_all_finite(loss, 'There are nan values')
        return tf.reduce_sum(tf.reduce_mean(loss, axis = 0)*self.inverse_class_weights) 


def build_model(layers, output_dir,inverse_class_weights, loss_class_weighted):
    """
    Compiles keras model for image classification/
    """    
    recall = tf.keras.metrics.Recall()
    precision = tf.keras.metrics.Precision()
    
    #original_loss_func - label_weighted_cross_entropy
    model = Sequential(layers)
    if loss_class_weighted: 
        model.compile(optimizer='adam',
                     loss=ClassImbalanceSparsityAdjustedLoss(inverse_class_weights),
                     metrics=[recall, precision, 'accuracy'])
    else: 
        model.compile(optimizer='adam',
                     loss=label_weighted_cross_entropy,
                     metrics=[recall, precision, 'accuracy'])
    return model

def train_and_evaluate(model_layers,num_epochs, steps_per_epoch, output_dir, nrows, loss_class_weighted, model_type):
    """
    Compile the model and load data for training into it.
    """
    if nrows: 
        train_nrow=int(nrows*.8)
        val_nrow=int(nrows*.2)
    else: 
        train_nrow=None
        val_nrow=None
    
    # Training Dataset 
    train_filenames, train_labels  = preprocessing.get_filenames_and_labels(config.meta_data_train_processed_path, 
                                                                            nrow_ind=train_nrow)
    # CALCULATE WEIGHTS HERE 
    labels_df = pd.DataFrame(train_labels,columns=config.response_variables)
    total = labels_df.sum(axis=0).sum()
    inverse_props = 1/labels_df.sum(axis=0).div(total)
    normalized_inverse_props = inverse_props.div(inverse_props.sum())
    inverse_class_weights=np.array(normalized_inverse_props)
    
    # Validation Dataset
    val_filenames, val_labels  = preprocessing.get_filenames_and_labels(config.meta_data_val_processed_path,
                                                                            nrow_ind=val_nrow)
    
    model = build_model(model_layers, output_dir=output_dir, inverse_class_weights=inverse_class_weights, loss_class_weighted=loss_class_weighted)

    print(f'Number of Training Records:{len(train_filenames)}')
    print(f'Number of Validation Records:{len(val_filenames)}')
    
    #create training and validation data
    train_ds = preprocessing.load_dataset(train_filenames, train_labels, config.BATCH_SIZE, model_type)
    val_ds = preprocessing.load_dataset(val_filenames, val_labels, config.VAL_BATCH_SIZE, model_type,training=False)
    
    callbacks = []
    if output_dir:
        tensorboard_callback = TensorBoard(log_dir=output_dir)
        callbacks = [tensorboard_callback]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=2,
        callbacks=callbacks)
    
    if output_dir:
        export_path = os.path.join(output_dir, 'keras_export')
        model.save(export_path, save_format='tf')
    
    return history
