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
model.add(Dropout(0.4))
model.add(Dense(512,
                activation='relu',
                kernel_regularizer=regularizers.l2(1e-2),
                bias_regularizer=regularizers.l2(1e-2),
                activity_regularizer=regularizers.l2(1e-2)
                ))
model.add(Dropout(0.4))
model.add(Dense(64,
                activation='relu',
                kernel_regularizer=regularizers.l2(1e-2),
                bias_regularizer=regularizers.l2(1e-2),
                activity_regularizer=regularizers.l2(1e-2)
                ))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=500, batch_size=512, verbose=1)
# make class predictions with the model
probs = model.predict(X_test)
predictions = ((probs) > 0.5).astype(int)

acc, f1 = compute_score(predictions, y_test)
fpr, tpr, auc_score = compute_roc_auc(probs, y_test)


print("Accuracy\t: ", acc)
print("F1 Score\t: ", f1)
print("AUC\t\t: ", auc_score)

# Performance of MLP
#    Acc Train  Acc Test  F1 Train   F1 Test  AUC Train  AUC Test
# 0   0.998382  0.865036  0.998725  0.894266   0.999975  0.932539
# 1   0.998119  0.867614  0.998518  0.896563   0.999956  0.935964
# 2   0.998237  0.869613  0.998612  0.897873   0.999971  0.934790
# 3   0.998158  0.867614  0.998550  0.896205   0.999962  0.936098
# 4   0.998198  0.866772  0.998580  0.895225   0.999951  0.935521
