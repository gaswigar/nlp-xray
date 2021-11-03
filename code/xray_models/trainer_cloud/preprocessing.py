import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from .config import config

AUTOTUNE = tf.data.experimental.AUTOTUNE

def rescale_image_tensor(t,domain_interval,range_interval):
    a=domain_interval[0]
    b=domain_interval[1]
    c=range_interval[0]
    d=range_interval[1]
    rescaled_t=c+((d-c)/(b-a))*(t-a)
    return(rescaled_t)

def decode_image(img, reshape_dims, num_channels, pixel_min, pixel_max):
    """
    Decode an image
    """
    img = tf.image.decode_png(img, channels=num_channels)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, reshape_dims)
    image_min = tf.math.reduce_min(img)
    image_max = tf.math.reduce_max(img)    
    if pixel_min!=image_min and pixel_max!=image_max:
        #img=img.map(lambda x: tf.py_function(func=rescale_image_tensor, inp=[x, [image_min,image_max],[pixel_min,pixel_max]], Tout=tf.float32))
        #img = img.map(lambda x: rescale_image_tensor(x,[image_min,image_max],[pixel_min,pixel_max]))
        img = rescale_image_tensor(img, [image_min,image_max],[pixel_min,pixel_max])
    img = tf.cast(img, dtype=tf.float32)
    return(img)

def decode(filename, label):
    """
    Decode file names.
    """
    image_bytes = tf.io.read_file(filename=filename)
    return image_bytes, label


def get_filenames_and_labels(data_path,nrow_ind):
    """
    Get filenames and labels
    """
    meta_data = pd.read_csv(data_path)
    all_files = meta_data['gs_path'].tolist()
    response_vars = config.response_variables    
    all_labels = np.array(meta_data[response_vars].values.tolist())
    if nrow_ind:
        if nrow_ind<= meta_data.shape[0]:            
            zip_lists=list(zip(all_files, all_labels))
            random.shuffle(zip_lists)
            files, labels = zip(*zip_lists)
            files=list(files)[:nrow_ind]
            labels=np.array(labels)[:nrow_ind]
            return(files, labels)
        else:
            return(all_files, all_labels)
    else: 
        return(all_files, all_labels)

        

def read_and_preprocess(image_bytes, label,model_type='cnn', random_augment=False):
    """
    Function which performs data augmentation.
    """
    pp=config.processing_parameters[model_type]
    if random_augment:
        img = decode_image(image_bytes, [pp['image_width']+10,pp['image_width']+10], pp['channels'], pp['pixel_scale_min'], pp['pixel_scale_max'])
        img = tf.image.random_crop(img, [config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, config.MAX_DELTA)
        img = tf.image.random_contrast(img, config.CONTRAST_LOWER, config.CONTRAST_UPPER)
    else:
        img = decode_image(image_bytes, [pp['image_width'],pp['image_width']], pp['channels'], pp['pixel_scale_min'], pp['pixel_scale_max'])
    return img, label

def read_and_preprocess_with_augment(image_bytes, label, model_type='cnn'):
    """
    Data augmentation for the training set.
    """
    return read_and_preprocess(image_bytes, label, model_type, random_augment=True)


# UPDATE HERE
def load_dataset(filenames, labels, batch_size,model_type, training=True):
    """
    This functions load the dataset from the GCS bucket.
    Inputs include:
    filenames: list of gcs locations for image files
    labels: numpy array of one hot encoded multi labels
    batch_size: batch size
    training: boolean entry specifying if training data is needed. False for test data.
    """
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels)).map(decode) #numpy array of filenames and numpy array of labels    
    if training:
        dataset = dataset.map(lambda f,l: read_and_preprocess_with_augment(f,l,model_type=model_type)).cache().shuffle(config.SHUFFLE_BUFFER).repeat(count=None)
    else:
        dataset = dataset.map(lambda f,l: read_and_preprocess(f,l,model_type=model_type)).repeat(count=1)
    
    return dataset.batch(batch_size=batch_size).prefetch(buffer_size=AUTOTUNE) 
