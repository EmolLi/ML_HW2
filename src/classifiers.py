from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# ================== PARAMETERS =======================
randomState = None

# ========== Classifiers ==========
logistic_regression = LogisticRegression(penalty="l2", random_state=randomState)
decision_tree_classifier = DecisionTreeClassifier(criterion="entropy", random_state=randomState)
svm = LinearSVC(penalty="l2", random_state=randomState, max_iter=1000)
MultinomialNB = MultinomialNB()
