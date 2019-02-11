from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# ================== PARAMETERS =======================
X = []
Y = []
trainSize = 0.8
randomState = None
# =====================================================

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=trainSize, test_size=1-trainSize)

# ========== Logistic Regression ==========
lg = LogisticRegression(penalty="l2", random_state=randomState)
lg.fit(x_train, y_train)
y_pred_lg = lg.predict(x_test)
print("Mean accuracy is ", lg.score(x_test, y_test))
print(metrics.classification_report(y_test, y_pred_lg))

# ========== Decision Trees ==========
dtc = DecisionTreeClassifier(criterion="entropy", random_state=randomState)
dtc.fit(x_train, y_train)
y_pred_dtc = dtc.predict(x_test)
print("Mean accuracy is ", dtc.score(x_test, y_test))
print(metrics.classification_report(y_test, y_pred_dtc))
