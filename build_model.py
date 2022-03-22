"""
Build the model using the best parameter and then save the model into external file

Reference:
    * https://www.tensorflow.org/ 
    * https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
    * https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/
    * https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
"""

import pandas as pd
import sklearn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.backend import clear_session
import pickle

import helper

BATCHSIZE = 512
EPOCHS = 500


NGRAM_RANGE = (2, 2)
DENSE_UNIT = 256
DROPOUT_RATIO = 0.3
LEARNING_RATE = 1e-4

if __name__ == "__main__":

    ## Clear clutter from previous Keras session graphs.
    clear_session()

    ## load the dataset
    names = pd.read_csv("data/name_gender.csv")

    ## preprocess the dataset
    names["name"] = names["name"].apply(lambda x: x.lower())
    names["name"] = names.apply(lambda row: helper.remove_punctuation(row["name"]), axis=1)
    names["name"] = names.apply(lambda row: helper.remove_number(row["name"]), axis=1)
    names["name"] = names.apply(lambda row: helper.normalize_text(row["name"]), axis=1)

    ## convert name into its feature representation
    count_vectorizer = sklearn.feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=NGRAM_RANGE)
    cv_features = count_vectorizer.fit_transform(names["name"])
    X = pd.DataFrame(data=cv_features.toarray(), columns=count_vectorizer.get_feature_names())

    ## convert string label into int label
    y = names["gender"].apply(lambda x: 1 if x == "M" else 0)

    ## define the keras model
    model = Sequential()

    input_dim = len(count_vectorizer.get_feature_names())
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))

    model.add(keras.layers.Dense(
        units=DENSE_UNIT,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-2),
        bias_regularizer=regularizers.l2(1e-2),
        activity_regularizer=regularizers.l2(1e-2)
    ))

    model.add(Dropout(DROPOUT_RATIO))
    model.add(Dense(64,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(1e-2),
                    bias_regularizer=regularizers.l2(1e-2),
                    activity_regularizer=regularizers.l2(1e-2)
                    ))

    model.add(Dense(1, activation='sigmoid'))

    ## set the training optimizer, loss, and evaluation metric
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    ## add early stopping to reduce overfitting
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10)

    ## fit the keras model on the dataset
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCHSIZE,
              validation_split=0.05, verbose=True, callbacks=[stop_early])

    model.save("model")
    pickle.dump(count_vectorizer, open('model/count_vectorizer.pickle', 'wb'))

