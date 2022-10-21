# axis=0: column-wise; axis=1: row-wise
df.apply(func,axis = )

# apply to every element
df.applymap(lambda x: )

# Applies a function across each column
df.apply(np.mean) 

# Applies a function across each row
df.apply(np.max, axis=1) 





# Función para limpiar Strings
def format_clean_str(dataframe,lista):
  for i in lista:
    dataframe[i] = dataframe[i].str.strip()
    dataframe[i] = dataframe[i].astype({i: 'object'})
    print("Columna formateado STR: ", i)
    
    
# Función para limpiar Ints    
def format_clean_int(dataframe,lista):
  for i in lista:
    if dataframe[i].dtypes == object:
      dataframe[i] = dataframe[i].str.strip()
      dataframe[i] = dataframe[i].fillna(0)
      dataframe[i] = dataframe[i].astype({i: 'int64'})
    else:
      dataframe[i] = dataframe[i].fillna(0)
      dataframe[i] = dataframe[i].astype({i: 'int64'})  
    print("Columna formateado INT: ", i)
    
# Formateo de columnas:
format_clean_str( misiones_ejecucion_, ['mision','grupo_mision'] 