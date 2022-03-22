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
import optuna

import helper


BATCHSIZE = 512
EPOCHS = 500

N_JOBS = 1
N_TRIALS = 72
MAX_TIMEOUT = 600

def compute_roc_auc(y_prob, y):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, y_prob)
    auc_score = sklearn.metrics.auc(fpr, tpr)
    return fpr, tpr, auc_score


def compute_score(y_pred, y):
    acc = sklearn.metrics.accuracy_score(y, y_pred)
    f1 = sklearn.metrics.f1_score(y, y_pred)
    return acc, f1


def objective(trial):
    # Clear clutter from previous Keras session graphs.
    clear_session()

    # load the dataset
    names = pd.read_csv("data/name_gender.csv")
    names["name"] = names["name"].apply(lambda x: x.lower())
    names["name"] = names.apply(lambda row: helper.remove_punctuation(row["name"]), axis=1)
    names["name"] = names.apply(lambda row: helper.remove_number(row["name"]), axis=1)
    names["name"] = names.apply(lambda row: helper.normalize_text(row["name"]), axis=1)

    ngram_range_trial = trial.suggest_categorical(
        'ngram_feature', ["(2,2)", "(3,3)"])

    ngram_range_dict = {
        "(2,2)" : (2, 2),
        "(2,3)" : (2, 3),
        "(3,3)" : (3, 3)
    }

    count_vectorizer = CountVectorizer(
        analyzer='char', ngram_range=ngram_range_dict[ngram_range_trial])

    cv_features = count_vectorizer.fit_transform(names["name"])


    X = pd.DataFrame(data=cv_features.toarray(),
                    columns=count_vectorizer.get_feature_names())

    y = names["gender"].apply(lambda x: 1 if x == "M" else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    
    # define the keras model
    model = Sequential()

    input_dim = len(count_vectorizer.get_feature_names())
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))

    # Tune the number of units in the Dense layer
    # Choose an optimal value between 256-512
    dense_units = trial.suggest_categorical(
        'dense_unit', [256, 384, 512])

    model.add(keras.layers.Dense(
        units=dense_units,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-2),
        bias_regularizer=regularizers.l2(1e-2),
        activity_regularizer=regularizers.l2(1e-2)
    ))

    dropout_units = trial.suggest_categorical(
        'dropout', [0.3, 0.4, 0.5, 0.6])

    model.add(Dropout(dropout_units))
    model.add(Dense(64,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(1e-2),
                    bias_regularizer=regularizers.l2(1e-2),
                    activity_regularizer=regularizers.l2(1e-2)
                    ))

    # model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    learning_rate_units = trial.suggest_categorical(
        'learning_rate', [1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate_units),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10)

    # fit the keras model on the dataset
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCHSIZE,
              validation_split=0.05, verbose=False, callbacks=[stop_early])
    
    # make class predictions with the model
    probs = model.predict(X_test, verbose=False)
    predictions = ((probs) > 0.5).astype(int)

    acc, f1 = compute_score(predictions, y_test)
    fpr, tpr, auc_score = compute_roc_auc(probs, y_test)

    # print("Accuracy\t: ", acc)
    # print("F1 Score\t: ", f1)
    # print("AUC\t\t: ", auc_score)
    # print(f"Parameters: {}")
    return acc


if __name__ == "__main__" :

    # Create a new study.
    study = optuna.create_study(study_name="Trial", direction="maximize")
    
    # Invoke optimization of the objective function.
    study.optimize(objective, n_jobs=N_JOBS, n_trials=N_TRIALS,
                   timeout=MAX_TIMEOUT, catch=())

    
    print("Number of finished trials: {}".format(len(study.trials)))

    logs = []
    for trial in study.trials :
        trial_log = f"Accuracy: {trial.value:.3f}; Params: {trial.params.items()}"
        logs.append(trial_log)
    logs = sorted(logs)
    for log in logs: print(log)
        

    trial = study.best_trial
    
    print("Best trial:")
    print("  Accuracy: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Performance of MLP
    #    Acc Train  Acc Test  F1 Train   F1 Test  AUC Train  AUC Test
    # 0   0.998382  0.865036  0.998725  0.894266   0.999975  0.932539
    # 1   0.998119  0.867614  0.998518  0.896563   0.999956  0.935964
    # 2   0.998237  0.869613  0.998612  0.897873   0.999971  0.934790
    # 3   0.998158  0.867614  0.998550  0.896205   0.999962  0.936098
    # 4   0.998198  0.866772  0.998580  0.895225   0.999951  0.935521


    # TODO:
    # - Preprocessing -> normalize greek character etc
    # - Count Vectorizer -> fit add dummy to handle out of vocabulary
    # X Count Vectorizer -> tune bigram , bigram-trigram, tigram
    # X Trainer -> add evaluation in val data per 10 epoch
    # X Trainer -> hyper parameter tuning
