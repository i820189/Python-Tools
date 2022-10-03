from sklearn.model_selection import GridSearchCV
svr = svm.SVC()
parameters = {'kernel':('linear','rbf'), 'C':[1, 10]}
clf = GridSearchCV(svr, parameters,n_jobs = -1,cv = 5)
clf.fit(X, y[0])
clf.cv_results_
clf.best_estimator_