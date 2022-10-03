# Drop them
df = df.dropna()
# Fill them
df.fillna(0) # fill by a number
df.fillna(method='ffill') # propagates last valid observation forward to next valid
df.fillna(method='bfill')