
#model parameters
IMG_HEIGHT = 400
IMG_WIDTH = 400
IMG_CHANNELS = 1

BATCH_SIZE = 1
SHUFFLE_BUFFER = 20*BATCH_SIZE

VALIDATION_IMAGES = 100
VALIDATION_STEPS = VALIDATION_IMAGES//BATCH_SIZE

#augmentation parameters
MAX_DELTA = 63.0/255.0 #adjusting brightness
CONTRAST_LOWER = 0.2
CONTRAST_UPPER = 1.8

#response variable columns in the mete_data_processed.csv
response_variables = ['atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'effusion', 'emphysema', 'fibrosis', 'hernia',
                 'infiltration', 'mass', 'no_finding', 'nodule', 'pleural_thickening', 'pneumonia', 'pneumothorax']

meta_data_train_processed_path = "gs://nlp-xray-dataset/meta_data/meta_data_processed_train.csv" #location to the meta data file - train 
meta_data_val_processed_path = "gs://nlp-xray-dataset/meta_data/meta_data_processed_val.csv" #location to the meta data file - val
meta_data_test_processed_path = "gs://nlp-xray-dataset/meta_data/meta_data_processed_test.csv" #location to the meta data file - test
meta_data_processed_path = "gs://nlp-xray-dataset/meta_data/meta_data_processed.csv" #location to the meta data file


