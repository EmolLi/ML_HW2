from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from src.data_loading import load_data
from config import test_data_path, dev_data_path, train_data_path, output_csv_path
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold
import csv

kFold_n = 8


# =========================================
def split_data(data):
    X_train, X_test, Y_train, Y_test = train_test_split(data['data'], data['target'], train_size=0.8,
                                                        test_size=0.2)
    return [X_train, Y_train, X_test, Y_test]


def tfidf_pipeline(classifier):
    return Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('norm', Normalizer()),
                     classifier])


def binary_occurrences_pipeline(classifier):
    return Pipeline([('vect', CountVectorizer(binary=True)),
                     classifier])





# ==========================================
test_data = load_data(test_data_path)
data = load_data(train_data_path)

# ================output====================
def output_prediction(Y_pred, test_data, test_index=None):
    output = []
    for i, y in enumerate(Y_pred):
        index = i if test_index is None else test_index[i]
        output.append([test_data['file_names'][index], y])

    def sortFiles(x):
        try:
            return int(x[0])
        except ValueError:
            return x[0]

    output.sort(key=sortFiles)
    with open(output_csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        for x in output:
            w.writerow(x)



def experiment(pipeline, classifier, model_name, X_train, Y_train, X_test, Y_test):
    model = pipeline(classifier)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print(metrics.classification_report(Y_test, Y_pred,
                                            target_names=["Negative", "Positive"]))
    f1 = metrics.f1_score(Y_test, Y_pred, average=None)[1]
    return [f1, Y_pred, model]


def experiment_basic_validation():
    [X_train, Y_train, X_test, Y_test] = split_data(data)
    experiment(tfidf_pipeline, ('clf', MultinomialNB()), "mnb+tfidf", X_train, Y_train, X_test, Y_test)
    # experiment(binary_occurrences_pipeline, ('clf', MultinomialNB()), "mnb+binary", X_train, Y_train, X_test, Y_test)


def experiment_k_fold():
    best_f1 = 0
    best_Y_pred = []
    best_test_index = []
    best_model = None
    kf = KFold(n_splits=kFold_n, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(data['data']):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train = [data['data'][i] for i in train_index]
        X_test = [data['data'][i] for i in test_index]
        Y_train = [data['target'][i] for i in train_index]
        Y_test = [data['target'][i] for i in test_index]
        [f1, Y_pred, model] = experiment(tfidf_pipeline, ('clf', MultinomialNB()), "mnb+tfidf", X_train, Y_train, X_test, Y_test)
        if (f1 > best_f1):
            best_f1 = f1
            best_Y_pred = Y_pred
            best_test_index = test_index
            best_model = model
        # experiment(binary_occurrences_pipeline, ('clf', MultinomialNB()), "mnb+binary", X_train, Y_train, X_test, Y_test)
    print("Best f1: ", best_f1)
    Y_pred_test = best_model.predict(test_data['data'])
    output_prediction(Y_pred_test, test_data)



experiment_k_fold()
# experiment_basic_validation()
