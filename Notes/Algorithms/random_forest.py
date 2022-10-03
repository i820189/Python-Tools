from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
rf = RandomForestClassifier(n_estimators=8000,n_jobs=-1,oob_score=True)
rf.fit(train,res)
rf.oob_score_
# feature importance
feature_importances = pd.Series(rf.feature_importances_,index = train.columns)
feature_importances.sort(inplace = True)
feature_importances.plot(kind = 'barh')