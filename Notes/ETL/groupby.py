df.groupby(by = 'Sex').mean()

# Returns a groupby object for values from one column
df.groupby(col) 

# Returns a groupby object values from multiple columns
df.groupby([col1,col2])

# Returns the mean of the values in col2, grouped by the values in col1 
# (mean can be replaced with almost any function from the statistics section)
df.groupby(col1)[col2].mean() 

# Creates a pivot table that groups by col1 and calculates the mean of col2 and col3
df.pivot_table(index=col1, values= col2,col3], aggfunc=mean) 

# Finds the average across all columns for every unique column 1 group
df.groupby(col1).agg(np.mean) 


# Crear dataframe desde otro dataframe
df_brand = pd.DataFrame(df_misiones__['brandto'].unique(),  columns = ['brandto'])