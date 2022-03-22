"""

Reference:
    * https://www.tensorflow.org/ 
    * https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
    * https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/
    * https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
"""

import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

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


def compute_roc_auc(y_prob, y):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, y_prob)
    auc_score = sklearn.metrics.auc(fpr, tpr)
    return fpr, tpr, auc_score


def compute_score(y_pred, y):
    acc = sklearn.metrics.accuracy_score(y, y_pred)
    f1 = sklearn.metrics.f1_score(y, y_pred)
    return acc, f1

if __name__ == "__main__":

    # Clear clutter from previous Keras session graphs.
    clear_session()

    # load the dataset
    names = pd.read_csv("data/name_gender.csv")
    names["name"] = names["name"].apply(lambda x: x.lower())
    names["name"] = names.apply(
        lambda row: helper.remove_punctuation(row["name"]), axis=1)
    names["name"] = names.apply(
        lambda row: helper.remove_number(row["name"]), axis=1)
    names["name"] = names.apply(
        lambda row: helper.normalize_text(row["name"]), axis=1)

    count_vectorizer = CountVectorizer(
        analyzer='char', ngram_range=(2, 2))

    cv_features = count_vectorizer.fit_transform(names["name"])

    X = pd.DataFrame(data=cv_features.toarray(),
                     columns=count_vectorizer.get_feature_names())

    y = names["gender"].apply(lambda x: 1 if x == "M" else 0)

    # define the keras model
    model = Sequential()

    input_dim = len(count_vectorizer.get_feature_names())
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))

    model.add(keras.layers.Dense(
        units=256,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-2),
        bias_regularizer=regularizers.l2(1e-2),
        activity_regularizer=regularizers.l2(1e-2)
    ))

    model.add(Dropout(0.3))
    model.add(Dense(64,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(1e-2),
                    bias_regularizer=regularizers.l2(1e-2),
                    activity_regularizer=regularizers.l2(1e-2)
                    ))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10)

    # fit the keras model on the dataset
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCHSIZE,
              validation_split=0.05, verbose=True, callbacks=[stop_early])

    model.save("model")
    filename = 'model/count_vectorizer.pickle'
    pickle.dump(count_vectorizer, open(filename, 'wb'))

