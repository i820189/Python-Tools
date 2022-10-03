# concat along rows
df_new = pd.concat([df_a, df_b])

# join
df = df1.join(df2, how='left', lsuffix='', rsuffix='', sort=False)

# Adds the rows in df1 to the end of df2 (columns should be identical)
df1.append(df2) 

# Adds the columns in df1 to the end of df2 (rows should be identical)
pd.concat([df1, df2],axis=1) 

# SQL-style joins the columns in df1 with the columns on df2 where the rows for col have identical values. 
# how can be one of 'left', 'right', 'outer', 'inner' 
df1.join(df2,on=col1,how='inner') 