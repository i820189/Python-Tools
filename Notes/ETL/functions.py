# axis=0: column-wise; axis=1: row-wise
df.apply(func,axis = )

# apply to every element
df.applymap(lambda x: )

# Applies a function across each column
df.apply(np.mean) 

# Applies a function across each row
df.apply(np.max, axis=1) 