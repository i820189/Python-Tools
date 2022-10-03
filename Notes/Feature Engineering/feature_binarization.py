# thresholding numerical features to get boolean values
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=30)
df['Age'] = df['Age'].apply(lambda x: binarizer.fit_transform(x)[0][0])