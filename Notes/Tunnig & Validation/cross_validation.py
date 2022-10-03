from sklearn.model_selection import cross_val_score
from sklearn import svm
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X.values, y[0].values, cv=5)
# 95% confidence interval of the score
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))