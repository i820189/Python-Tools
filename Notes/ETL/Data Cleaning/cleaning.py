df.columns = ['a','b','c'] # Renames columns
pd.isnull() # Checks for null Values, Returns Boolean Array
pd.notnull() # Opposite of s.isnull()
df.dropna() # Drops all rows that contain null values
df.dropna(axis=1) # Drops all columns that contain null values
df.dropna(axis=1,thresh=n) # Drops all rows have have less than n non null values
df.fillna(x) # Replaces all null values with x
s.fillna(s.mean()) # Replaces all null values with the mean (mean can be replaced with almost any function from the statistics section)
s.astype(float) # Converts the datatype of the series to float
s.replace(1,'one') # Replaces all values equal to 1 with 'one'
s.replace([1,3],['one','three']) # Replaces all 1 with 'one' and 3 with 'three'
df.rename(columns=lambda x: x + 1) # Mass renaming of columns
df.rename(columns={'old_name': 'new_ name'}) # Selective renaming
df.set_index('column_one') # Changes the index
df.rename(index=lambda x: x + 1) # Mass renaming of index


def remove_spaces(dataframe):  
  for i in dataframe.columns:
    if dataframe[i].dtype == 'object':
      dataframe[i] = dataframe[i].map(str.strip)
    else:
      pass
  

----------------------

import re

def formatbrand(text):
  text_ = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,']", " ", text)
  lineNumber = []
  for line in text_.split(" "):
      line = line.strip()  # Strip trailing spaces and newline
      lineNumber.append(line.capitalize())

  txt = "".join(lineNumber)
  return txt


X = data.drop('quality', axis=1)




To find all the values from the series that starts with a pattern "s":
SQL - WHERE column_name LIKE 's%' 
Python - column_name.str.startswith('s')

To find all the values from the series that ends with a pattern "s":
SQL - WHERE column_name LIKE '%s'
Python - column_name.str.endswith('s')

To find all the values from the series that contains pattern "s":
SQL - WHERE column_name LIKE '%s%'
Python - column_name.str.contains('s')