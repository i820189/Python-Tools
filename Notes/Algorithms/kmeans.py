from sklearn.cluster import KMeans
estimator = KMeans(n_clusters=3)
estimator.fit(filtered_scaled)
labels = estimator.labels_