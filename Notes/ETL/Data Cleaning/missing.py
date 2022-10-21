# Drop them
df = df.dropna()
# Fill them
df.fillna(0) # fill by a number
df.fillna(method='ffill') # propagates last valid observation forward to next valid
df.fillna(method='bfill')


misiones_sku_[ (
  misiones_sku_['sku'].isnull()) | (misiones_sku_['grupo_mision'].isnull()) | 
  (len(misiones_sku_['sku'])<1)  | (len(misiones_sku_['grupo_mision'])<1)
]

#######################################################################################################

>>> df['colB'].isna().sum()
2
>>> df['colA'].isna().sum()
0

>>> df['colB'].isnull().sum()
2

>>> df.isna().sum()
colA    0
colB    2
colC    3
colD    1
dtype: int64

>>> df[['colA', 'colD']].isna().sum()
colA    0
colD    1
dtype: int64


# Count rows having only NaN values in the specified columns
>>> df[['colC', 'colD']].isna().all(axis=1).sum()
1

# Count rows containing only NaN values in every column
>>> df.isnull().all(axis=1).sum()
0

# Count the NaN values within the whole DataFrame
>>> df.isna().sum().sum()
6

#######################################################################################################


df.groupby(['No', 'Name'], dropna=False, as_index=False).size()

df[["No", "Name"]].value_counts(dropna=False)