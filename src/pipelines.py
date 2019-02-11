from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from src.data_loading import load_data
from config import test_data_path, dev_data_path, train_data_path
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics


# ================= tfidf =================
def split_data(data):
    X_train, X_test, Y_train, Y_test = train_test_split(data['data'], data['target'], train_size=0.8,
                                                        test_size=0.2)
    return [X_train, Y_train, X_test, Y_test]


def tfidf_pipeline(classifier):
    return Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('norm', Normalizer()),
                     classifier])

# ==========================================


data = load_data(train_data_path)
[X_train, Y_train, X_test, Y_test] = split_data(data)


def experiment(pipeline, classifier, model_name):
    model = pipeline(classifier)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print(metrics.classification_report(Y_test, Y_pred,
                                        target_names=["Negative", "Positive"]))




experiment(tfidf_pipeline, ('clf', MultinomialNB()), "mnb+tfidf")
