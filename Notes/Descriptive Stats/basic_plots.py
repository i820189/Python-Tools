# line chart
fig = plt.figure(figsize=(12,6))
plt.plot(data.dateTime,data.volume_out)
plt.title('title')

# hist: numerical feature distribution
df.Age.hist()
# categorical feature distribution  
df.Survived.value_counts().plot(kind = 'bar')
# Basic box plot
sns.boxplot(consum.instantaneous,orient='v')
plt.title('instantaneous consumption value distribution')
# Box plot with hue
sns.boxplot(x="Sex", y="Age",hue = 'Survived', data=df, palette="Set3")

# Scatter
plt.scatter(df.Fare,df.Survived)
plt.xlabel('Fare')
plt.ylabel('Survived?')

# Regression chart
sns.jointplot(x="duration", y="usage", kind = 'reg', data=filtered)
plt.title('title')
