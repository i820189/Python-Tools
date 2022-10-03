from sklearn import linear_model
# Create linear regression object
regr = linear_model.LinearRegression(fit_intercept=True)
# Train the model using the training sets
regr.fit(df, y)
# The coefficients
print('Coefficients:', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(df) - y) ** 2))
# Explained variance score: 1 is perfect prediction
print "R Squared score:";regr.score(df, y)

# Coef_ check
plt.figure(figsize=(12,8))
plt.barh(range(len(regr.coef_)),regr.coef_,height=0.2,tick_label = df.columns)
plt.title('Regression Coefficients')

# Residuals Check
res = regr.predict(df) - y
plt.axhline(0)
plt.scatter(range(len(res)),res.values,color = 'r')
plt.title('Residual Plot')