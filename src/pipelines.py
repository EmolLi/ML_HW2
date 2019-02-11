from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from src.data_loading import load_data
from config import test_data_path, dev_data_path, train_data_path
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold

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
data = load_data(train_data_path)


def experiment_basic_validation():
    [X_train, Y_train, X_test, Y_test] = split_data(data)
    # experiment(tfidf_pipeline, ('clf', MultinomialNB()), "mnb+tfidf", X_train, Y_train, X_test, Y_test)
    # experiment(binary_occurrences_pipeline, ('clf', MultinomialNB()), "mnb+binary", X_train, Y_train, X_test, Y_test)




def experiment(pipeline, classifier, model_name, X_train, Y_train, X_test, Y_test):
    model = pipeline(classifier)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print(metrics.classification_report(Y_test, Y_pred,
                                            target_names=["Negative", "Positive"]))


def experiment_k_fold():
    kf = KFold(n_splits=kFold_n, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(data['data']):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train = [data['data'][i] for i in train_index]
        X_test = [data['data'][i] for i in test_index]
        Y_train = [data['target'][i] for i in train_index]
        Y_test = [data['target'][i] for i in test_index]
        experiment(tfidf_pipeline, ('clf', MultinomialNB()), "mnb+tfidf", X_train, Y_train, X_test, Y_test)
        # experiment(binary_occurrences_pipeline, ('clf', MultinomialNB()), "mnb+binary", X_train, Y_train, X_test, Y_test)


experiment_k_fold()