from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# ================== PARAMETERS =======================
randomState = None

# ========== Classifiers ==========
logistic_regression = LogisticRegression(penalty="l2", random_state=randomState)
decision_tree_classifier = DecisionTreeClassifier(criterion="entropy", random_state=randomState)
MultinomialNB = MultinomialNB()
