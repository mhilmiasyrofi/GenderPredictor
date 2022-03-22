# first neural network with keras make predictions
from numpy import loadtxt
import pandas as pd

import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
from sklearn.model_selection import train_test_split


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers


import helper


def countvect(data):
    count_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))

    emb = count_vectorizer.fit_transform(data)

    return emb, count_vectorizer


def compute_roc_auc(y_prob, y):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, y_prob)
    auc_score = sklearn.metrics.auc(fpr, tpr)
    return fpr, tpr, auc_score


def compute_score(y_pred, y):
    acc = sklearn.metrics.accuracy_score(y, y_pred)
    f1 = sklearn.metrics.f1_score(y, y_pred)
    return acc, f1


# load the dataset
names = pd.read_csv("data/name_gender.csv")
names["name"] = names["name"].apply(lambda x: x.lower())
names["name"] = names.apply(lambda row: helper.remove_punctuation(row["name"]), axis=1)
names["name"] = names.apply(lambda row: helper.remove_number(row["name"]), axis=1)

cv_features, count_vectorizer = countvect(names["name"])

X = pd.DataFrame(data=cv_features.toarray(),
                 columns=count_vectorizer.get_feature_names())

y = names["gender"].apply(lambda x: 1 if x == "F" else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# define the keras model
model = Sequential()
model.add(Dense(256, input_dim=618, activation='relu'))
model.add(Dropout(0.3, input_shape=(512,)))
model.add(Dense(512,
                activation='relu',
                kernel_regularizer=regularizers.l2(1e-3),
                bias_regularizer=regularizers.l2(1e-3),
                activity_regularizer=regularizers.l2(1e-3)
                ))
model.add(Dropout(0.4, input_shape=(512,)))
model.add(Dense(64,
                activation='relu',
                kernel_regularizer=regularizers.l2(1e-3),
                bias_regularizer=regularizers.l2(1e-3),
                activity_regularizer=regularizers.l2(1e-3)
                ))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=300, batch_size=512, verbose=1)
# make class predictions with the model
probs = model.predict(X_test)
predictions = ((probs) > 0.5).astype(int)

acc, f1 = compute_score(predictions, y_test)
fpr, tpr, auc_score = compute_roc_auc(probs, y_test)


print("Accuracy\t: ", acc)
print("F1 Score\t: ", f1)
print("AUC\t\t: ", auc_score)
