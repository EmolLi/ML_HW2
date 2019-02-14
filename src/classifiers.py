from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.svm import LinearSVC

# ================== PARAMETERS =======================
randomState = None

# ========== Classifiers ==========
logistic_regression = LogisticRegression(penalty="l2", random_state=randomState)
decision_tree_classifier = DecisionTreeClassifier(criterion="entropy", random_state=randomState)
MultinomialNB = MultinomialNB()
svc = svm.SVC(gamma='scale')
lsvc = LinearSVC(random_state=551, tol=1e-5)