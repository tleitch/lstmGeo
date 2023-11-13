# initialize tf/kera and/or whatver else you need here
import os as os
import pandas as pd
import numpy as np
import tensorflow as tf
tf.config.list_physical_devices("GPU")
from tensorflow.python import keras
from keras.layers import LSTM, Bidirectional
from keras.layers import Dense, Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
import string
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.utils import to_categorical
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTEN
from imblearn.over_sampling import SMOTENC
from collections import Counter
from sklearn.model_selection import KFold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import load_model
from io import StringIO
import chardet
import csv
from datetime import datetime
import xgboost as xgb
import ast  # Import the ast module for literal evaluation


#import tensorflow.keras as keras
#print(tensorflow.keras.__version___)
# print(tf.__version__)

#using only one GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the 3rd GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

#setting the directory
os.chdir('/home/lstm/voter')

# Process assumes this is run from directory where there is an "./experiemnts" and
# "data" file. I could automate their creation but have not as yet. The experiemtns directory is for
# outputing models, data is for input files which were included in this zip

# os.chdir("rethnicity")

os.chdir('/home/lstm')
# directory containing  dataset file
data_directory = os.listdir('/home/lstm')


#getting a list of data files in the directorys
dataFiles = [file for file in data_directory if file.endswith('.csv')]
print("experiment files available: ", dataFiles)

# # ----- Any Pre-Processing goes here ----

# create ASCII dictionary
chars = ['E'] + [chr(i) for i in range(97,123)] + [' ', 'U'] 
id2char = {i:j for i,j in enumerate(chars)}
char2id = {j:i for i,j in enumerate(chars)}


def name2id(name, l = 10):
    ids = [0] * l
    for i, c in enumerate(name):
        if i < l:
            if c.isalpha():
                ids[i] = char2id.get(c, char2id['U'])
            elif c in string.punctuation:
                ids[i] = char2id.get(c, char2id[' '])
            else:
                ids[i] = char2id.get(c, char2id['U'])
    return ids

# ---- Creating a file that captures the stats as they come off the models ----
dataFiles = ["votersGeo.csv"]      ###CHECK RELEVANT FILE NAME BEFORE RUNNING ANYTHING 
dFile=dataFiles[0]
for dFile in dataFiles:
    #copying the experiment file to test directory with the new name (change with every test run)
    fileOut=["/home/lstm/test_results/", dFile.split(".")[0],"_testLr3_prob_race5.csv"] #or with xgboost
    fileOut = "".join(fileOut)
    # creating model file name by stripping input data file name
    modelOut = ["/home/lstm/models/", dFile.split(".")[0], "_Lr3_prob_race5.h5"]  # tracked name lengths in file name
    modelOut = "".join(modelOut)
    #copying the experiment file to validate directory with the new name (changes with every test run)
    vFile = ["/home/lstm/validate_results/", dFile.split(".")[0],"_validateLr3_prob_race5.csv"] #or with xgboost
    vFile = "".join(vFile)
    # Build data file input name
    dFile = ["/home/lstm/", dFile]  #already reading in near data directory when prompting user
    dFile = "".join(dFile)

    # votersGeo.csv is not comma separated
    # Try different encodings, such as 'utf-8', 'latin1', 'ISO-8859-1', etc.
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            df = pd.read_csv(dFile, delimiter=';', encoding=encoding)
            break  # Exit the loop if successful
        except UnicodeDecodeError:
            pass
    df.head()
    df.shape[0]
    #mapping race5 values to integer
    print(df['race5'].unique())
    
    # Sample dataframe with a column 'race5'
    race5_unique = {'race5': ['white5', 'afrAmer5', 'hisp5', 'asian5', 'other5', 'multi', 'unkown', 'hispUnk']}
    #df = pd.DataFrame(race5_unique)
    # Define the mapping dictionary
    mapping = {'white5': 1,'afrAmer5': 2,'hisp5': 3,'asian5': 4,'other5': 5,'multi': 6,'unkown': 7,'hispUnk': 8}  # Map 'hispUnk' to the same integer as 'unkown'
    # Map the values in the 'race5' column to integers
    df['race5_map'] = df['race5'].map(mapping)
    df.head()
    print(df['race5_map'].unique())
    print(df['gender'].unique())
    print(df['gender'].value_counts())
    print(df)
    
    df = df.astype({"lname" : str,"fname" : str})
    
    #remove this subset of dataset
    subset_condition1 = ((df['race5_map'] == 6) | 
    ((df['race5_map'] == 7) | (df['race5_map'] == 8 ))) #subset to predict on later
    
    #also drop the gender where the value is U
    subset_condition2 = (df['gender'] == 'U')| (df['gender'] == ' ')
    
    #build a new dataframe without the subset
    df = df[~(subset_condition1 | subset_condition2)]
    df.shape

    #MODEL TRAINING - LSTM
    # Convert to numeric representation (converting to scalar)
    X = np.array([name2id(fn.lower(), l=12) + name2id(ln.lower(), l=16)
              for fn, ln in zip(df['fname'], df['lname'])], dtype=object)
    y = np.array([int(i) for i in (df['race5_map']-1).tolist()]) #subtracting -1 from y due to value error into the model
    
    # No need for k-fold or resampling
    X_train_all_folds = X
    y_train_all_folds = y
    # Convert lists to numpy arrays
    X_train_all_folds = np.vstack(X_train_all_folds)
    # Convert y_train_all_folds, y_test_all_folds, and y_validate_all_folds to numpy arrays
    y_train_all_folds = np.array(y_train_all_folds)
        
    #TRAIN THE MODEL
    # cut texts after this number of words (among top max_features most common words)
    num_words = len(id2char)
    feature_len = 28
    batch_size = 512
    print(len(X_train_all_folds), 'train sequences')
    
    print('Pad sequences (samples x time)')
    X_train_all_folds = sequence.pad_sequences(X_train_all_folds, maxlen=feature_len)
    #X_test_all_folds = sequence.pad_sequences(X_test_all_folds, maxlen=feature_len)
    #X_validate_all_folds=sequence.pad_sequences(X_validate_all_folds, maxlen=feature_len)
    
    print('X_train shape:', X_train_all_folds.shape)
    #print('X_test shape:', X_test_all_folds.shape)
    #print('X_validate shape:', X_test_all_folds.shape)
    print(np.max(y_train_all_folds) + 1)

    num_classes = 5  # np.max(y_train) + 1 --> only for binary
    print(num_classes, 'classes')

    print('Convert class vector to binary class matrix ''(for use with categorical_crossentropy)')
    y_train_all_folds = to_categorical(y_train_all_folds, num_classes)
    #y_test_all_folds = to_categorical(y_test_all_folds, num_classes)
    #y_validate_all_folds = to_categorical(y_validate_all_folds, num_classes)
    print('y_train shape:', y_train_all_folds.shape)
    # simple train-test
    # first build
    model = Sequential()
    model.add(Embedding(num_words, 256, input_length=feature_len))
     # try out bi-directional LSTM
    model.add(Bidirectional(LSTM(512, return_sequences=True, dropout=0.2)))
    model.add(Bidirectional(LSTM(512, return_sequences=True, dropout=0.2)))
    model.add(Bidirectional(LSTM(512, return_sequences=True, dropout=0.2)))
    model.add(Bidirectional(LSTM(512, dropout=0.2)))
    model.add(Dense(num_classes, activation='softmax'))

    # choose between learning rates
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10 ** -3),
        loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    callback = tf.keras.callbacks.EarlyStopping(mode='min', monitor='loss', patience=1, min_delta=.001) #earlier delta was 0.0015 

    # train model
    model.fit(X_train_all_folds, y_train_all_folds, batch_size=batch_size, epochs=10, verbose=1, callbacks=[callback])

    #for the train data
    y_pred_train = model.predict(X_train_all_folds, batch_size=batch_size, verbose=1)
    
    # get predicitions on test data
    #y_pred = model.predict(X_test_all_folds, batch_size=batch_size, verbose=1)
    #y_pred_bool = np.argmax(y_pred, axis=1)
    #print captured performance versus training set
    print("train results", file=open(fileOut, "a"))
    y_pred_bool_train = np.argmax(y_pred_train, axis=1)
    print(classification_report(y_train_all_folds.argmax(axis=1), y_pred_bool_train),  file=open(fileOut, "a"))
    print(confusion_matrix(y_train_all_folds.argmax(axis=1), y_pred_bool_train), file=open(fileOut, "a"))
    
    #getting the probabilities for the race5 category so they can be added into features for xgboost
    # Add the flattened probabilities as a single column in the existing DataFrame
    df['probabilities_race5_cat'] = y_pred_train.tolist() 

    output_directory = '/home/lstm/'
    
    # Define the full path of the output file
    output_file = os.path.join(output_directory, 'votersGeo_probabilities_race5.csv')

    #Save the DataFrame with probabilities to a new CSV file
    df.to_csv(output_file, index=False)

    #once the probabilities are done, they are sent into xgboost
    #MODEL TRAINING - XGBOOST
    