# (0,1) scaling 
# (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
cols_to_norm = ['PassengerId','SibSp']
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: scaler.fit_transform(x))

# Standardization: Zero mean and unit variance
from sklearn.preprocessing import scale
cols_to_norm = ['Age','SibSp']
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: scale(x))

# Normalization: scaling individual observation (row) to have unit norm.
# if you plan to use a quadratic form such as the dot-product or any other kernel to quantify the similarity of any pair of samples, like KNN. 
from sklearn.preprocessing import normalize
df_normalized = pd.DataFrame(normalize(df._get_numeric_data(),norm = 'l2'),columns=df._get_numeric_data().columns,index=df._get_numeric_data().index)
df_normalized.apply(lambda x: np.sqrt(x.dot(x)), axis=1) # check results