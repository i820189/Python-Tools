import datetime
import matplotlib.pyplot as plt
import glob
from stat import FILE_ATTRIBUTE_ARCHIVE
import pandas as pd
import numpy as np


# Python

############################################################################################################
##### ENTORNOS ####################################################################
############################################################################################################

>python - m pip install - -upgrade pip  # actualizar el PIP

> pip install virtualenv (instalar en el path de python38 de archivos de programa, sino agregar la carpeta scripts al path de entorno global)
> virtualenv env_name_entorno  # creo el entorno
> source env/bin/activate  # activar para linux / mac
> source env/Scripts/activate  # activarpara gitbash
> D: / projcts/ejemplo/env/Scripts > activate   # activar para command (CMD)

> pip freeze  # Lista los paquetes instalados en la virtual con sus versiones
> pip freeze > requeriments.txt   # pasar

> deactivate  # desactivar


**** segundo caso ** ***

python - m venv env
source env/bin/activate(mac)

pip list(ver versión de pip)
pip install - -upgrade pip(actualizo pip)

pip install flask

touch app.py


export FLASK_APP = app.py(setear variables)
export FLASK_ENV = development / export FLASK_ENV = production

flask run(inicializar el app o tmb python app.py, previa seteada)

pip freeze > requirements.txt / pip install - r requirements.txt


**** Streamlit ** *

source ~/.bash_profile
pip install streamlit
streamlit run myapp.py


############################################################################################################
##### CARPETS / FILES ####################################################################
############################################################################################################
for i in glob.glob("./*"): print(i)

filename = r'.\ds_consultora_202006.json'

os.getcwd()  # detect the current working directory and print it
os.mkdir('/carpeta1/carpeta2')

try:
	os.mkdir(path)
except OSError:
	print('error')
else:
	print('creado')

path = r'/Users/javierdiaz/Dropbox/Belcorp/python/prepedidos'  # MAC
all_files = glob.glob(path + "/*.csv")

# forecast : https://facebook.github.io/prophet/docs/quick_start.html

# source : https://elitedatascience.com/python-cheat-sheet

# https://developers.google.com/machine-learning/crash-course/framing/ml-terminology

############################################################################################################
 IMPORTING DATA
############################################################################################################

pd.read_csv(filename)  # From a CSV file
pd.read_table(filename)  # From a delimited text file (like TSV)
pd.read_excel(filename)  # From an Excel file
pd.read_sql(query, connection_object)  # Reads from a SQL table/database
pd.read_json(json_string)  # Reads from a JSON formatted string, URL or file.
# Parses an html URL, string or file and extracts tables to a list of dataframes
pd.read_html(url)
pd.read_clipboard()  # Takes the contents of your clipboard and passes it to read_table()
# From a dict, keys for columns names, values for data as lists
pd.DataFrame(dict)


#######
Carga de datos a través de la función open
######

data3 = open(mainpath + "/" +
             "customer-churn-model/Customer Churn Model.txt", 'r')

cols = data3.readline().strip().split(",")
n_cols = len(cols)

counter = 0
main_dict = {}
for col in cols:
    main_dict[col] = []

for line in data3:
    values = line.strip().split(",")
    for i in range(len(cols)):
        main_dict[cols[i]].append(values[i])
    counter += 1


print("El data set tiene %d filas y %d columnas" % (counter, n_cols))

df3 = pd.DataFrame(main_dict)

df3.head()


############################################################################################################
EXPLORING DATA
############################################################################################################

# SET LIMIT ROWS
pd.options.display.max_rows = 999


# Once you have imported your data into a Pandas dataframe, you can use these methods to get a sense of what the data looks like:

df.shape()  # Prints number of rows and columns in dataframe
df.head(n)  # Prints first n rows of the DataFrame
df.tail(n)  # Prints last n rows of the DataFrame
df.info()  # Index, Datatype and Memory information
df.describe()  # Summary statistics for numerical columns
s.value_counts(dropna=False)  # Views unique values and counts
df['pais'].unique()  # Ver valores unicos como array
df['pais'].unique().tolist()  # Ver valores unicos como lista
len(df['pais'].unique())  # Ver cuantos países hay

df['sex'].value_counts(ascending=True)  # rank ascending conunt values
df['fare'].value_counts(bins=7)  # bins rank

df.apply(pd.Series.value_counts)  # Unique values and counts for all columns
df.describe()  # Summary statistics for numerical columns
df.mean()  # Returns the mean of all columns
df.corr()  # Returns the correlation between columns in a DataFrame
df.count()  # Returns the number of non-null values in each DataFrame column
df.max()  # Returns the highest value in each column
df.min()  # Returns the lowest value in each column
df.median()  # Returns the median of each column
df.std()  # Returns the standard deviation of each column
df.columns
df.shape
df.info
# contar vaores unicos, contabilizar lso nulls
df['pais'].value_counts(dropna=False)
df['pais'].value_counts(dropna=False).head()

df.describe()  # statistics


############################################################################################################
DATA VISUALIZATION
############################################################################################################
# BARPLOT / Continuous / frequencies / Discrete
# puedo ver la distribución de los totales, y ver outliers
df['totales'].plot('hist')
# puedo encontrar ya los outliers, mayores a algo anormal
df[df['totales'] > 100000]

# BOXPLOT / Outliers / in/Max / 25,50,75 % percentiles
df.boxplot(column='totales', by='pais')
plt.show()

# SCATTERPLOTS / relation 2 variables / Flag potencial bad data (erros not found by looking at 1 variable)
# ...no dice buscar...

.apply(lambda x: str(x))


######################################### MATPLOTLIB #######################################################
year = [202007, 202008, 202009, 202010]
pop = [2.6, 3.67, 1.7, 2.2]

year = [202001, 202002, 202003] + year  # agregar mas valores a los años
pop = [3.3, 4.4, 5.1] + pop  # agregar mas valores a los pop

plt.plot(year, pop)  # lineas unidas

# penalizo mas el pop, para que las burbujas se vean mas grandes y se pueda detallar los pesos (s)
np_pop = np.array(pop) * 3

# puntos en el plano, la variable S es tamaño de los punts del scatter plot, si no va simplemente no tiene tamaño es uniforme
plt.scatter(year, pop, s=np_pop)
# C = colores por categoria, alpha = transparency
plt.scatter(x=gdp_cap, y=life_exp, s=np.array(pop) * 2, c=col, alpha=0.8)

plt.hist(values, bins=3)  # distribución de datos en un rango

# Labels Axis
plt.xlabel('Campaña')
plt.ylabel('Totales')

# Title
plt.title('Titulo de campaña')

# Ticks
# Etiquetas para las escalas por ejemplo en el axis "y"
plt.yticks([0, 2, 4, 6, 8, 10], ['0', '2B', '4B', '6B', '8B', '10B'])

# Escalas los valores, ejemplo logaritmica
plt.xscale('log')  # poner en escala logaritmica el axis "X"

# Agregar texto en un punto determinado del grafico
plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')

# Agregar una grilla para que se visualice mas exacto los puntos
plt.grid(True)

plt.show()  # Imprime
plt.clf()  # limpiar plots para mostrar siguiente plot consecutivo / puedo comprara coportamiento de dos muestras en dos planos


plt.plot([1, 2, 3, 4, 5, 6, 7])
plt.show()


############################################################################################################
SELECTING DATA
############################################################################################################

df[col]  # Returns column with label col as Series
df[[col1, col2]]  # Returns Columns as a new DataFrame
s.iloc[0]  # Selection by position (selects first element)
s.loc[0]  # Selection by index (selects element at index 0)
df.iloc[0, :]  # First row
df.iloc[0, 0]  # First element of first column

# MELTING / unificar dos columnas en una sola y los valores en una tercera:
pd.melt(
    frame=df,
    id=vars='name',
    value_vars=['campo_1', 'campo_2']
    var_name='campo',
    value_name='result'
)  # organizar por NAMES los valores del campo_1 y campo_2

# PIVOT / opuesto del melting, separar el diferentes columnas, ayuda a la reporteria lo hace mas amigable
weather.pivot(
    index='date',
    columns='element',  # esta es la columna a desagregar
    values='value'
)

tmp11 = df_testops_final.groupby(['key2','group']).size().reset_index(name='tot').pivot(
  index='key2',
  columns='group',
  values='tot'
).reset_index()

tmp11['porc_control'] =  tmp11['Control'] / (tmp11['Test']+tmp11['Test sin Cambios']+tmp11['Control'])
tmp11.sort_values('porc_control', ascending=False).head(20)


-----------------------


DICTIONARYS

# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

# Get index of 'germany': ind_ger
ind_ger = countries.index("germany")

# Use ind_ger to print out capital of Germany
print(capitals[ind_ger])

world = {"afganistan": 30.55, "albania": 2.77, "algeria": 39.21}
world['albania']

del (world['albania']) 			# borra un registro del diccionario
# agreggar u nuevo valor al diccionario ó actualizar un valor al diccionario
europe["italy"] = "rome"

europe['spain']['capital']


************ PANDAS ** ********************************************************************************
Dictionary -> DataFrame

# Crear un dataframe y agregarle campos, puede ser en un for
smc = pd.DataFrame(columns=["campania", "codpais", "total"])
smc = smc.append({"campania": campania, "codpais": codpais,
                 "total": total}, ignore_index=True)

df = pd.DataFrame(dict)
brics = pd.read_csv("path/to/brics.csv", index_col=0 				# poner como incide la columna 0
					)

cars.index = ['CL', 'MX', 'GT', 'CO', 'PE']  		# agregar el index al dataframe

# DATAFRAME 1D+ - seleccionar dos campos completos, sigue siendo un dataframe      (varias dimensiones 2D +)
type(df[['campo1', 'campo2']])
# SERIES 1D- seleccionamos solo los datos, son los valores del campo nada mas (una dimension 1D)
type(df['campo1'])
df[1:4]  # FILAS - seleccionar campos de los indices 1 al 4

np_array[rows, columns]  # Algo similar a esto en PANDAS es los LOC / ILOC
# En Pandas obtendría algo asi, pero con indices conocidos
df.loc[["ind2", "ind3"], ["col1", "col2"]]
df.loc[:, ["col1", "col2"]]  # acceder a todas las columnas
df.loc[[1, 4, 6], [4, 8, 9]]  # acceder a filas y columnas especificas

# LOC -> puedes llamar por su nombre, tanto columnas como filas
df.loc[['row1', 'row2'], ["col1", "col2"]]
# ILOC -> puedes llamar por su indice,  tanto columnas como filas
df.iloc[[0, 2], [1, 3]]


************ NUMPY ** ********************************************************************************
# convertir a array para oeprar mejor
arrayy = np.array[(12, 345, 56, 32, 54, 654,)]

np.logical_and(bmi > 21, bmi < 22)  # resultado como booleans
bmi[np.logical_and(bmi > 21, bmi < 22)]		# resutado del array
np.logical_or(bmi > 21, bmi < 22)
logical_not()

cpc = cars['cars_per_cap']
between = np.logical_and(cpc > 10, cpc < 80)
medium = cars[between]

** RANDOM **
np.random.rand()  # random entre 0 y 1 (generado pseudoaleatorio, en base a una semilla)
np.random.seed(123)  # definir una semilla por ejemplo para pruebas
np.random.randint(0, 2)  # entero que puede ser 0 o 1 (el 2 no se incluye)


# Transpose np_aw: np_aw_t
np_aw_t = np.transpose(np_aw)

# Plot np_aw_t and show
plt.plot(np_aw_t)
plt.show()


************ LOOPS / FOR / WHILE ** ********************************************************************************
x = 1
while x < 4:
    print(x)
    x = x + 1

areas = [11.25, 18.0, 20.0, 10.75, 9.50]
for index, a in enumerate(areas):  # Con enumerate obtengo el indice del DF
    print("room " + str(index) + " : " + str(a))

for k in np.nditer(np_baseball):
    print(k)

for lab, row in brics.iterrows():
	brics.loc[lab, "name_length"] = len(row['country'])


******** DATETIME ** *************************************************************+

datetime.datetime.now()  # format date-time
datetime.datetime.now().date()  # solo obtengo al fecha de hoy

date_time_str = '2018-06-29 08:15:27.243860'
date_time_obj = datetime.datetime.strptime(
    date_time_str, '%Y-%m-%d %H:%M:%S.%f')

print('Date:', date_time_obj.date())
print('Time:', date_time_obj.time())
print('Date-time:', date_time_obj)

date = datetime.strptime('fecha_string', '%d %b %Y %H:%M:%S')
fecha = datetime.strptime('fecha_string', '%Y-%m-%d').date()
.apply(lambda x: datetime.strptime(x, '%d %b %Y %H:%M:%S').time())


############################################################################################################
CLEANING DATA
############################################################################################################
If you’re working with real world data, chances are you’ll need to clean it up. These are some helpful methods:

df.columns = ['a', 'b', 'c']  # Renames columns
pd.isnull()  # Checks for null Values, Returns Boolean Array
pd.notnull()  # Opposite of s.isnull()
df.dropna()  # Drops all rows that contain null values
df.dropna(axis=1)  # Drops all columns that contain null values
# Drops all rows have have less than n non null values
df.dropna(axis=1, thresh=n)
df.fillna(x)  # Replaces all null values with x
# Replaces all null values with the mean (mean can be replaced with almost any function from the statistics section)
s.fillna(s.mean())
s.astype(float)  # Converts the datatype of the series to float
s.replace(1, 'one')  # Replaces all values equal to 1 with 'one'
# Replaces all 1 with 'one' and 3 with 'three'
s.replace([1, 3], ['one', 'three'])
df.rename(columns=lambda x: x + 1)  # Mass renaming of columns
df.rename(columns={'old_name': 'new_ name'})  # Selective renaming
df.set_index('column_one')  # Changes the index
df.rename(index=lambda x: x + 1)  # Mass renaming of index


##### Filter, Sort and Group By ####################################################################
Methods for filtering, sorting and grouping your data:

df[df[col] > 0.5]  # Rows where the col column is greater than 0.5
df[(df[col] > 0.5) & (df[col] < 0.7)]  # Rows where 0.5 < col < 0.7
df.sort_values(col1)  # Sorts values by col1 in ascending order
# Sorts values by col2 in descending order
df.sort_values(col2, ascending=False)
# Sorts values by col1 in ascending order then col2 in descending order
df.sort_values([col1, col2], ascending=[True, False])
dogs.sort_values(["columna", "columnb"], acending=[True, False])
top_3 = sorted(poke_list_np, key=lambda x: x[1], reverse=True)[
               :3]  # encontrado en datacamp
df.groupby(col)  # Returns a groupby object for values from one column
# Returns a groupby object values from multiple columns
df.groupby([col1, col2])
# Returns the mean of the values in col2, grouped by the values in col1 (mean can be replaced with almost any function from the statistics section)
df.groupby(col1)[col2].mean()
df.pivot_table(index=col1, values=col2, col3], aggfunc=mean)  # Creates a pivot table that groups by col1 and calculates the mean of col2 and col3
# Finds the average across all columns for every unique column 1 group
df.groupby(col1).agg(np.mean)
df.apply(np.mean)  # Applies a function across each column
df.apply(np.max, axis=1)  # Applies a function across each row
df['texto'].apply(len)  # Longitud de caracteres

# All of our three examples used exactly the same groupby() call to begin with:
df.groupby('day')['total_bill'].mean()
df.groupby('day').filter(lambda x: x['total_bill'].mean() > 20)
df.groupby('day')['total_bill'].transform(lambda x: x/x.mean())


##### Joining and Combining ####################################################################
Methods for combining two dataframes:

# Adds the rows in df1 to the end of df2 (columns should be identical)
df1.append(df2)
# Adds the columns in df1 to the end of df2 (rows should be identical)
pd.concat([df1, df2], axis=1)
# SQL-style joins the columns in df1 with the columns on df2 where the rows for col have identical values.
df1.join(df2, on=col1, how='inner')
									# how can be one of 'left', 'right', 'outer', 'inner'



EXPORT FILE

##### Writing Data ####################################################################
And finally, when you have produced results with your analysis, there are several ways you can export your data:

df.to_csv(filename)  # Writes to a CSV file
df.to_excel(filename)  # Writes to an Excel file
df.to_sql(table_name, connection_object)  # Writes to a SQL table
df.to_json(filename)  # Writes to a file in JSON format
df.to_html(filename)  # Saves as an HTML table
df.to_clipboard()  # Writes to the clipboard


##### Machine Learning ####################################################################
The Scikit-Learn library contains useful methods for training and applying machine learning models. Our Scikit-Learn tutorial provides more context for the code below.
For a complete list of the Supervised Learning, Unsupervised Learning, and Dataset Transformation, and Model Evaluation modules in Scikit-Learn, please refer to its user guide.

# Import libraries and modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
# Load red wine data.
dataset_url='http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data=pd.read_csv(dataset_url, sep=';')
# Split data into training and test sets
y=data.quality
X=data.drop('quality', axis=1)
X_train, X_test, y_train, y_test=train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 123,
                                                    stratify = y)
# Declare data preprocessing steps
pipeline=make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))
# Declare hyperparameters to tune
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}
# Tune model using cross-validation pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, y_train)
# Refit on the entire training set
# No additional code needed if clf.refit == True (default is True)
# Evaluate model pipeline on test data
pred = clf.predict(X_test)
print r2_score(y_test, pred)
print mean_squared_error(y_test, pred)
# Save model for future use
joblib.dump(clf, 'rf_regressor.pkl')
# To load: clf2 = joblib.load('rf_regressor.pkl')







****** clases de patrones : ********
creacionales
estructurales
de comportamiento


############################################## funcion : CONTAR CUANTOS VALORES EN UNA COLUMNA HAY PEOR EN UN DICTIONARY ############################
# Define count_entries()
def count_entries(df, col_name):
    """Return a dictionary with counts of 
    occurrences as value for each key."""

    # Initialize an empty dictionary: langs_count
    langs_count = {}
    
    # Extract column from DataFrame: col
    col = df[col_name]
    
    # Iterate over lang column in DataFrame
    for entry in col:

        # If the language is in langs_count, add 1
        if entry in langs_count.keys():
            langs_count[entry] += 1
        # Else add the language to langs_count, set the value to 1
        else:
            langs_count[entry] = 1

    # Return the langs_count dictionary / tuplas / multiples valores
    return langs_count

# Call count_entries(): result
result = count_entries(tweets_df,'lang')

# Print the result
print(result)


############################################## funcion : CONTAR CUANTOS VALORES EN UNA COLUMNA HAY PEOR EN UN DICTIONARY ############################

# Define count_entries()
def count_entries(df, col_name = 'lang'):
    """Return a dictionary with counts of
    occurrences as value for each key."""

    # Initialize an empty dictionary: cols_count
    cols_count = {}

    # Extract column from DataFrame: col
    col = df[col_name]
    
    # Iterate over the column in DataFrame
    for entry in col:

        # If entry is in cols_count, add 1
        if entry in cols_count.keys():
            cols_count[entry] += 1

        # Else add the entry to cols_count, set the value to 1
        else:
            cols_count[entry] = 1

    # Return the cols_count dictionary
    return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df)

# Call count_entries(): result2
result2 = count_entries(tweets_df,'source')

# Print result1 and result2
print(result1)
print(result2)




############################################## funcion : CONTAR CUANTOS VALORES EN UNA COLUMNA HAY PEOR EN UN DICTIONARY ############################

# Define count_entries()
def count_entries(df, *args):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    # Initialize an empty dictionary: cols_count
    cols_count = {}
    
    # Iterate over column names in args
    for col_name in args:
    
        # Extract column from DataFrame: col
        col = df[col_name]
    
        # Iterate over the column in DataFrame
        for entry in col:

            # If entry is in cols_count, add 1
            if entry in cols_count.keys():
                cols_count[entry] += 1
    
            # Else add the entry to cols_count, set the value to 1
            else:
                cols_count[entry] = 1

    # Return the cols_count dictionary
    return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang')

# Call count_entries(): result2
result2 = count_entries(tweets_df, 'lang', 'source')

# Print result1 and result2
print(result1)
print(result2)


#################################################### FUNCTION : MAP ####################################################################

# Map aplica la función a toda la secuencia :
map(func, seq)

nums = [48,12,5,6,2]
df = map(lambda x: x**2, nums)



############################################# LAMBDA FUNCTION ###########################################################################

# Define echo_word as a lambda function: echo_word
echo_word = (lambda word1,echo : word1 * echo)

# Call echo_word: result
result = echo_word('hey',5)

# Print result
print(result)


##################################################  Contar cuantas veces se repite una letra en un codigo ######################################################################
******** ALCANCE A FUNCIONES : 
* Alcance global (global scope)- > definido en el cuerpo del script, o en el programa
* 


####################
Contar cuantas veces se repite una letra en un codigo : 
####################

def count_letter(content, letter):
  """Count the number of times `letter` appears in `content`.

  # Add a Google style arguments section
  Args:
    content (stri): The string to search.
    letter (str): The letter to search for.
  """
  if (not isinstance(letter, str)) or len(letter) != 1:
    raise ValueError('`letter` must be a single character string.')
  return len([char for char in content if char == letter])


###############################################  Contar cuantos se repiten en una lista  'Counter' #########################################################################
from collections import Counter

# Collect the count of primary types
type_count = Counter(primary_types)
print(type_count, '\n')

############################################### Contar cuantos de una lista inicializan con una letra en especifico ###############################################
from collections import Counter

starting_letters = [name[0] for name in names]    # obtengo las primeras letras en una LISTA

starting_letters_count = Counter(starting_letters) # luego le hago un Counter para ver los totales

print(starting_letters_count)

############################################## COMBINATIONS ITERTOOLS : HACER COMBINACIONES de elementos de una lista ##############################################

from itertools import combinations
combos_4 = [*combinations(pokemon,4)]
print(combos_4)

############################################## SETS (like IN) : COMPARAR, RESTAR, ver que elementos están en otra lista ##############################################

# Convert both lists to sets
ash_set = set(ash_pokedex)
misty_set = set(misty_pokedex)

# Find the Pokémon that exist in both sets
both = ash_set.intersection(misty_set)
print(both)

# Find the Pokémon that Ash has and Misty does not have
ash_only = ash_set.difference(misty_set)
print(ash_only)

# Find the Pokémon that are in only one set (not both)
unique_to_set = ash_set.symmetric_difference(misty_set)
print(unique_to_set)

Probar la velocidad :

%timeit 'Psyduck' in ash_pokedex
%timeit 'Psyduck' in brock_pokedex_set


##############################################  SET :  VALORES UNICOS EN UNA LISTA  ##############################################

The below function was written to gather unique values from each list:

def find_unique_items(data):
    uniques = []

    for item in data:
        if item not in uniques:
            uniques.append(item)

    return uniques

---
testing : 

%timeit find_unique_items(names)
> 1.79 ms +- 116 us per loop (mean +- std. dev. of 7 runs, 1000 loops each)

%timeit set(names)
> 11 us +- 509 ns per loop (mean +- std. dev. of 7 runs, 100000 loops each)


##############################################  Eliminating LOOPS / MAP  ##############################################

%%timeit
totals = []
for row in poke_stats:
    totals.append(sum(row))
> 140 µs ± 1.94 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit totals_comp = [sum(row) for row in poke_stats]
> 114 µs ± 3.55 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%timeit totals_map = [*map(sum, poke_stats)]
> 95 µs ± 2.94 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)   # Here is better

----------
Exercise : 

# Collect Pokémon that belong to generation 1 or generation 2
gen1_gen2_pokemon = [name for name,gen in zip(poke_names, poke_gens) if gen < 3]

# Create a map object that stores the name lengths
name_lengths_map = map(len, gen1_gen2_pokemon)

# Combine gen1_gen2_pokemon and name_lengths_map into a list
gen1_gen2_name_lengths = [*zip(gen1_gen2_pokemon, name_lengths_map)]

print(gen1_gen2_name_lengths[:5])

##############################################  Eliminating LOOPS / NUMPY  ##############################################
import numpy as np

%timeit avgs = poke_stats.mean(axis=1)
> 23.1 µs ± 235 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

%%timeit
avgs = []
for row in poke_stats:
    avg = np.mean(row)
    avgs.append(avg)

> 5.54 ms ± 224 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)



############################################## Writing better ############################################## 

# Moving calculations above a loop

# Import Counter
from collections import Counter

# Collect the count of each generation
gen_counts = Counter(generations)

# Improve for loop by moving one calculation above the loop
total_count = len(generations)

for gen,count in gen_counts.items():
    gen_percent = round(count / total_count * 100, 2)
    print('generation {}: count = {:3} percentage = {}'
          .format(gen, count, gen_percent)
    )

-------

# Collect all possible pairs using combinations()
possible_pairs = [*combinations(pokemon_types, 2)]

# Create an empty list called enumerated_tuples
enumerated_tuples = []

# Append each enumerated_pair_tuple to the empty list above
for i,pair in enumerate(possible_pairs, 1):
    enumerated_pair_tuple = (i,) + pair
    enumerated_tuples.append(enumerated_pair_tuple)

# Convert all tuples in enumerated_tuples to a list
enumerated_pairs = [*map(list, enumerated_tuples)]
print(enumerated_pairs)

----------

# Nueva columna

# Create an empty list to store run differentials
run_diffs = []

# Write a for loop and collect runs allowed and runs scored for each row
for i,row in giants_df.iterrows():
    runs_scored = row['RS']
    runs_allowed = row['RA']
    
    # Use the provided function to calculate run_diff for each row
    run_diff = calc_run_diff(runs_scored, runs_allowed)
    
    # Append each run differential to the output list
    run_diffs.append(run_diff)

giants_df['RD'] = run_diffs
print(giants_df)




############################################################################################################
SCKIMAGE - SCKIT IMAGE
############################################################################################################

# Import the modules from skimage
from skimage import data, color

# Load the rocket image
rocket = data.rocket()

# Convert the image to grayscale
gray_scaled_rocket = color.rgb2gray(rocket)

# Show the original image
show_image(rocket, 'Original RGB image')

# Show the grayscale image
show_image(gray_scaled_rocket, 'Grayscale image')

######    NUMPY FOR IMAGES    ###########################################################

Con Numpy podemos procesar imágenes con ´tecnicas de:
    - Voltear imágenes
    - Extraer características
    - Analizar

######    CARGAR IMAGEN    ###########################################################
# Loading the image using Matplotlib / vemos que una imagene s una matriz multimensional
madrid_image = plt.imread('/madrid.jpeg')
type(madrid_image)
> <class 'numpy.ndarray''>)

######    COLORES CApAS RGB   ###########################################################
# Colors with Python
# Obtaining the red values of the image
red = image[:,:,0]
# Obtaining the green values of the image
green = image[:,:,1]
# Obtaining the blue values of the image
blue = image[:,:,2]

######    ESCALA DE GRISES   ###########################################################
Podemos plotearlos en escala de grises usando el atributo CMAP de la función IMSHOW, y es posible ver las diferentes intensidades de la imágen para cada color.
plt.imshow(red, cmap="gray")
plt.tittle('Red')
plt.axis('off')
plt.show()

######    TAMAÑO DE IMAGEN    ###########################################################
Podemos ver el TAMAÑO de cada imagen / matriz
madrid_image.shape
> (426, 640, 3)

Podemos ver el total número de PIXELES
madrid_image.size
> 817920

######    ROTAR IMAGEN    ###########################################################
Podemos voltear ima imagen con la función FLIPUD VERTICALMENTE
# Flip the image in up direction
vertically_flipped = np.flipud(madrid_image)
show_image(vertically_flipped, 'Vertically flipped image')

Podemos voltear ima imagen con la función FLIPUD HORIZONTALMENTE
# Flip the image in left direction
horizontally_flipped = np.fliplr(madrid_image)
show_image(horizontally_flipped, 'Horizontally flipped image')

######   HISTOGRAMAS    ###########################################################
HISTOGRAMS
Se usan para 
- Analizar
- Establecer Umbrales(un tema importante den computer visión)
- Alterrar Brillo
- Contraste
- Igualar una imagen (bien usado)

MATPLOTLIB tiene un método de HISTOGRAMMM / PLOTEAR UN HISTOGRAMA DE IMAGEN
- Toma una MATRIZ de entrada(Feecuencia) y BINS como parámetros
- RAVEL devuelve una matriz aplanada continua a partir de la matriz
- Definimos 256, porque mostraremos todos los números de pixeles existentes, de 0 a 255
# Red color of the image
red = image[:, :, 0]
# OBtain the red Histogram
plt.hist(red.ravel(), bins = 256)


######   UMBRALES / THRESHOLDING    ###########################################################
Usado usualmente para  dividir el fondo y el primer plano en imágenes en escala de grises, Conviriendolas en blanco y Negro.
Toma cada pixel y lo compara en un UMBRAL y define BLANCO o NEGRO
Aplicaciones:

- Object Detection
- Face Detection
- Etc...

# Definimos el MEDIO
# Obtain the optimal threshold value
thresh = 27

# Apply  thresholding to the image
inverted_binary = image < thresh

# Show  the original and thresholded
show_image(image, 'Original')
show_image(inverted_binary, 'Inverted Thresholded')


CATEGORÍAS DE UMBRASLES
- GLOBAL / HISTOGRA BASED
    Es bueno usandolo en imagenes con fondos relativamrnte uniformes

- LOCAL / ADAPTATIVE
    Usado para Fondos que nos e diferencian, con iluminación de fondos desigual
    Es un proceso más LENTO que el GLOBAL

- ALGORITMOS GLOBALES / Propone el mejor
    Es una función que evalúa diferentes algoritmos globales
    Evalua 7 algoritmos
    from skimage.color import rgb2gray
    from skimage.filters import try_all_threshold
    image = rgb2gray(image_color)
    # Obtain all the resulting images
    fig, ax = try_all_threshold(image, verbose=False)   #Ponemos False en verbose para que no imprima el nombre de la función para cada método.
    # Showing resulting plots
    show_plot(fig, ax)

DEFINIENDO UN VALOR OPTIMO GLOBAL
image = rgb2gray(image_color)
# Import the outsu threshold function
from skimage.filters import threshold_otsu   # Creo que elegimos OUTSU porque eso fue lo que nos botó el 'try_all_threshold' del algoritmo anterior
# Obtain the optimal threshold value
thresh = threshold_otsu(image)
# Apply thresholding to the image
binary_global = image > thresh
# Show the original image  and binarized image / IMAGEN BINARIZADA
show_image(image, 'Original')
show_image(binary_global, 'Global thresholding')


DEFINIENDO UN VALOR OPTIMO LOCAL
tex_image = rgb2gray(tex_image_color)
# Import the outsu threshold function
from skimage.filters import threshold_local   # Creo que elegimos OUTSU porque eso fue lo que nos botó el 'try_all_threshold' del algoritmo anterior
# Definimos el tamaño del bloque que va a calular umbrales de cada pixel, barrios locales
block_size = 35
# Obtiene el mejor umbral local, el OFFSET es el desplazamiento, OFFSET OPCIONAL, es una constante restada de la media de los bloques para calcular el valore del umbral local
local_tresh = threshold_local(tex_image, block_size, offset= 10)
# Aplicamos el umbral local
binary_local = text_image > local_tresh

show_image(image, 'Original')
show_image(binary_local, 'Local thresholding')


######    jump into FILTERING    ###########################################################
En este capitulo veremos FILTRADO, CONTRASTE, TRANSFORMACIONES y MORFOLOGÍA

######   FILTROS   ########################
Es una técnica par MODIFICAR o MEJORAR una imagen
Un filtro es una FUNCION MATEMÁTICA que se aplica a un filtro
Se puede usar para ENFATIZAR o ELIMINAR ciertas características, como bordes, suavizado, afilado y detecciónd e bordes.
Acá veremos SUAVIZADO y DETECCIÓN de borders
el FILTRADO es una operación de Neighboarhood

- FILTRADO, podemos detectar bordes, técnica para ubicar los límites de los objetos dentro de las imágenes
    además de segmentar y extraer informaciónd e cuantas monedas hay en la imágen.
    La detecciónd e bordes funciona detectando discontinuidades en el brillo
- ECUALIZACIÓN DE HISTOGRAMA  para contraste realce!
- FUNCIONES MORFOLOGICAS

VEREMOS FILTROSSSSS:

- FUNCION SOBEL, es muy utilizada, requiere de una imágene escala de GRISES bidimensional.
from skimage.filters import sobel
# Apply edge detection filter
edge_sobel = sobel(image_coins)
plot_comparasion(image_coins, edge_sobel, "Edge with Sobel")

- FILTRO GAUSSIANO / GAUSSIAN SMOOTHING
Usado para enfocar una imagen o para REDUCIR el RUIDO
Diosminuye los BORDES y reducirá el CONTRASTE
Se usa en otras técnicas como el filtrado ANTI-ALIASING FILTERING.
from skimage.filters import gaussian
# Apply edge detection filter
gaussian_image = gaussian(amsterdam_pic, multichannel=True)  # Multicanal en True si es de COLOR, sino False
# Show original and resulting image to compare
plot_comparison(amsterdam_pic, gaussian_image, "Blurred with Gaussian filter")


######   "CONTRASTE" / MEJORAMIENTO DE IMAGENES   ########################
* CONTRAST ENHANCEMENT (Aplicaciones de HISTOGRAMAS para MEJORA DE IMAGENES)
- Cuando se aumenta el CONTRASTE las imágenes se vuelven más visibles, por ejempllo una radiografía
- El contraste, puede verse como LA MEDIDA DE SU RANGO DINAMICO, o como la EXTENSION de su histograma
Va- lores frecuenes de intensidad de HISTOGRAMA utilizando la DISTRIBUCION DE PROBABILIDAD

* ECUALIZACION DEL HISTOGRAMA - Types:
- Histogram Equalization (ESTANDAR)
    Distribuye los valores de intensidad más frecuentes
    from skimage import exposure #accedemos a todas las funciones de ecualización
    # Obtain the equalized image ESTANDAR
    image_eq = exposure.equalize_hist(image)

- Adaptative Histogram Equalization (ADAPTATIVO)
    Calcula varios histogramas, cada uno correspondiene a una parte distinta de la imagen y lo utiliza para redistribuir los valores de luminosidad del hsitograma de la imagen

- Contrast Limited Adaptive Histogram Equalization CLAHE (ADAPTATIVO LIMITADO)
    Es parecido al Adaptativo, pero menos intenso.
    Desarrollado para evitar la sobreamplificación del ruido que puede generar ala ecualización del histograma ADAPTATIVO
    Opera en regiones pequeñas llamadas MOSAICOS, o VECINDARIOS para el hsitograma.
    image_adapteq = exposure.equalize_adapthist(image, clip_limit=0.03)
    Ese LIMITE DE RECORTE(Clip_Limit) se normaliza enrte 0 y 1 (valores ma´s altos dan más contraste)

The contrast is:
In [4]: show_image(clock_image)
In [5]: np.max(clock_image) - np.min(clock_image)
Out[5]: 148

PLoteando el HISTOGRAM
# Import the required module
from skimage import exposure
# Show original x-ray image and its histogram
show_image(chest_xray_image, 'Original x-ray')
plt.title('Histogram of image')
plt.hist(chest_xray_image.ravel(), bins=256)
plt.show()

######   TRANSFORMATIONS / TRANSFORMACIONES  ################################
A veces necesitamos girar o escalar una imagen.
Aplicaciones:
- Reducir una imagen para procesarla más rapido
- Escalar todas las imagenes al mismo tamaño, antes de procesarlas 

* ROTATING:
from skimage.transform import rotate
image_rotated = rotate(image, -90) # Izquierda 90
show_image(image, 'Original')

* RESCALING: Dowrngrading - 4 veces más pequeña:
from skimage.transform import rescale
image_rescale = rescale(image, 1/4, anti_aliasing=True, multichannel=True) #Anti Aliasing, flag apra aplicar un suavisado antes de reducir la escala. MUltichannel True is color

    ¿¿ QUE ES ALIASING ??
    El  ALIASING es un patrón o un efecto de ondulación
    El ALIASING hace que laimagen parezca que tiene ondas y ondulaciones que irradian desde una determinada parte

* RESIZING:
Tiene el mismo propósito que REESCALAR, pero permite espcificar una forma de imagen de salida en lugar de un factor de escala.
Definimos una tupla(alto y ancho)deseada para cambiar el tamaño
from skimage.transorm import resize
height = 400
width = 500
image_resized = resize(image, (heught,width), anti_aliasing=True)

    * RESIZING PROPORTIONALLY
    from skimage.transorm import resize
    height = image.shape[0] / 4
    width = image.shape[1] / 4
    image_resized = resize(image, (heught,width), anti_aliasing=True)


######   MORPHOLOGY / MORFOLOGIA  ################################
Para detecciónde  OBjetos, se detecta por sus CARACTERISTICAS como la FORMA.
Las regiones BINARIAS (0,1 - BLANCO-NEGRO) producidas por un simple UMBRAL, puede distorsionarse por el RUIDO y la TEXTURA.
Las operaciones de FILTRADO MORFOLOGICO intentan eliminar estos imperfecciones teniendo en cuenta la forma y estrictura de los ojetos en la imagen.
Estas operaciones son especialmente para imagenes escalas binarias, hasta grayscale.
OPERACIONES BÄSICAS:
- DILATACION
    Agrega pixeles a los límites de los objetos en una imagen.
    from skimage import morphology
    # Usamos el por DEFAULT que es la CRUZ y probamos 
    dilated_image = morphology.binary_dilation(image_horse)
- EROSION
    Elimina PIxeles en lso límites de los objetos
    EROSION BINARIA
    from skimage import morphology
    # Con esto podemos configurar opcionalmente un elemento de estructuración para usar la operación:
    selem = rectangle(12,6) #Utilizamos Rectangulo porque tiene algo de forma del caballo de la imagen que queremos obtener.
    # Si no definimos el "elemento estructurado" utilizara por default la CRUZ
    enroded_image = morphology.binary_erosion(image_horse, selem=selem)
    # Con el efecto por default, se ve mejor solo en esta imagen
     enroded_image = morphology.binary_erosion(image_horse)

** La cantidad de pixeles agregados o eliminados de los objetos en una imagne, dependen del tamaño y la forma de un elemento estructurante(pequeña imagen binaria para sondear la imagen de entrada) utilizado para procesar la imagen.
    Elementos estructurantes:
    - Cuadrado 5x5
    - Diamond 5x5
    - Cross-Shaped 5x5 (como una cruz)
    - Square 3x3
    from skimage import morphology
    square = morphology.square(4)
    rectangle = morphology.rectangle(4,2)


######   IMAGE RESTORATION  /RESTAURACION DE IMAGENES ##########################
Aprenderemos sobre RESTAURACION , SEGMENTACION, RUIDO y como encontrar CONTORNOS en imágenes. 
UTILIZADO para ARREGLAR imagenes dañadas, eliminar TEXTOS, Eiminar LOGOS, eliminar objetos pequeños, como tatuajes.

- INPAINTING (ECUACION BIARMONICA)
    Recuperaicón de partes perdidas o deterioradas de imágenes
    Recupera Automaticamente explotando la información presentada en las regiones NO DAÑADAS de la imagen.
    COn la función BIHARMONIC INPAINT, necesitamos ubicar los pixeles DAÑADOS, como una imagen de MASCARA en la parte superior de la imagen para trabajar.
    from skimage.restoration import inpaint
    # Obtain the mask
    mask = get_mask(defect_image) # OSea este get_mask() es una función en duro que definimos que pixeles vamos a corregir, se llenan con 1
    # Apply
    restored_image = inpaint.inpaint_biharmonic(defect_image, mask, multichannel=True)

    Para definir la MASCARA AUTOMATICAMENTE(segmento de pixeles negros, osea faltantes, errados = 1) se necesitará utilizar THRESHOLDING o SEGMENTACION para hacerlo.

    # Initialize the mask
    mask = np.zeros(image_with_logo.shape[:-1])
    # Set the pixels where the logo is to 1  /estos son pixeles ya definidos de un logo en una imagen
    mask[210:290, 360:425] = 1

- NOISE / RUIDO
    Las imágenes son señales, y las señales del mundo real generalmente contienen DESVIACIONES de la señal ideal.
    Estas desviaciones se denominan RUIDO, es el resultado de errores en la adquisición de la imagen. Resultan estos pixeles que no reflejan las verdaderas INTENSIDADES de la escena real.
    
    * COMO CREAR RUIDO
        Mas conocido como sal y pimienta (generado aleatoriamente)
        Mientras más resolución sea la imagen, más tardarña en eliminar el ruido.
        from skimage.util import random_noise
        # Add noise to the image
        noisy_image = random_noise(dog_image)
        show_image(noisy_image)

    * COMO ELIMINARLO
        * TIPOS DE ALGORITMO de eliminaciónd e ruido
            - TOTAL VARIATION (TV) - Filtro de variación Total
                Intenta minimizar la variación total de la imagen.
                Produe imagenes parecida a los dibujos, es decir, imágenes constantes por partes.
                Se utiliza la FUNCION de eliminación de ruido del CANAL DE TELEVISIÓN del módulo de restauración
                from skimage.restoration import denoise_tv_chambolle
                # WEIGHT -> Peso de eliminación de ruido / mas peso tenga mejor eliminad, pero a su vez suaviza massss
                denoised_image = denoise_tv_chambolle(noise_image, weight=0.1, multichannel=True)
                show_image(denoised_image)

            -BILATERAL - Filtrado Bilateral
                Suaviza las imágenes conservando bordes
                Reemplaza la INTENSIDAD de cada Píxel con im PROMEDIO PONDERADO de los valores de intensidad de los píxeles cercanos.
                Se utiliza la FUNCION de eliminación de RUIDO BILATERAL desde el package de restauración
                from skimage.restoration import denoise_bilateral
                denoised_image = denoise_bilateral(noise_image, multichannel=True) # Vemos que conserva más los BORDES, no tan suaviadoo como el anterior.

            - WAVELET DENOISING - Eliminación de ruido de ondas (opcional no visto en el tema.)

- SUPERPIXEL  / SEGMENTATION
    La idea es dividir la imagen en SEGMENTOS o REGIONES paracambiar la representación en algo más significativo y mas facil de analizar.
    Applications:
        - Por ejemplo antes de analizar un tumor en una radiografía computarizada, se debe aislar el TUMOR
        - Antes de reconocer una CARA debe seleccionarse de su FONDO
        - Podemos utilizar UMBRALES que es el simple métodod e segmentación, separando el primer plano del fondo
        - Aprenderemos mas que separar el fondo.
        - Un solo pixel no es representación natural, así que podemos explorar significados más lógicos en una imagen formada por REGIONES mas grandes o PIXELES AGRUPADOS, llamado como SUPERPIXELS.
        - SUPERPIXEL e sun grupo de pixeles conectados con colores o niveles de gris similares
        
        SEGMENTACION DE SUPERPIXELES 
        D- ivide una imagen en superpíxeles, aplicado a muchas tareas de COmputer Visión, como el seguimiento visual y la clasificación de imágenes.
        - Puede calcular ENTIDADES en regiones más significativas
        - Reduce una imágen de miles de pixeles a algunas REGIONES para ALGORITMOS POSTERIORES, reduciendo gran nivel de procesamiento computacional.

        DOS TIPOS DE SEGMENTACION:
        - SUPERVISED
            
           +`Ñ.LPL, ,K9IKM MJI9IJN NHU7YHB VGT6TGVVF5RFCCFR5TGREWRTYUYTREWQ azxaASDFGHJKLÑ´Ç
           
            Donde se utiliza algunos conocimientos previos para guiar el algoritmo.
            Como el tipo de umbral en el que especificamos el valor umbral nosotros mismos/

        - UNSUPERVISED
            Donde no se requieren conocimientos previos
            Estos algoritmos intenta subdividir imagenes en regiones significatiuvas AUTOMATICAMENTE
            Es posible realizar cietas configuraciones para obtener el resultado deseado, como el umbral de OTSU que usamos en el priemr capitulo.

            SIMPLE LINEAR ITERATIVE CLUSTERING (SLIC) (utiliza KMEANS)
            Enfoquemos en una técnica de SEGMENTACIO en SUPERPIXELES, denominada CLUSTERING ITERATIVO LINEAL SIMPLE (SLIC).
            Segmenta la imagen utilizando un algoritmo de aprendizaje automatico, llamado AGRUPACION DE K-MEANS , intenta separarlos en un numero predefinido de SUBREGIONES.
            Usamos el método LABEL2RGB del package de COLOR para devolver una imagen donde los segmentos obtenidos del SLIC METHOD, se resaltará em método en rodajas, ya sea con colores aleatorios o con el color promedio del segmento de superpixeles

            from skimage.segmentation import slic
            from skimage.color import label2rgb
            # Obtain the segments
            segments = slic(image, n_segments=300) # SEGMENT OR LABELS, SI QUEREMOS 300 segmentos, 100 DEFAULT
            # Put segments on top of original image to compare
            segmented_image = label2rgb(segments, image, kind='avg') # USAMOS UN COLOR PROMEDIO
            show_image(segmented_image)


- FINDING CONTOURS
    Aprenderemos a encontrar CONTORNOS de los objetos en una imagen
    Un contorno e suna forma cerrada de puntos o segmentos de línea, que representan los límites de estos objetos
    Podemos:
        - Medir el tamaño
        - Clasificar Formas
        - Determinar la cantidad de objetos en una imagen

    La entrada a una FUNCION DE BUSQUEDA de contorno debe ser un binario, que podemos producir, aplicando primero el UMBRAL.
    En tal imagen ya BINARIA, los objetos que deseamos detectar deben ser BLANCOS, mientras que el fondo NEGRO.

    1. PREPARING THE IMAGE:
        - CONVERTIR LA IMAGEN A 2D GRAYSCALE
        from skimage.filters import threshold_otsu
            # Make the image grayscale
            cimage = color.rgb2gray(image)
        - BINARIXE THE IMAGE
            # Obtain the thresh value
            thresh = threshold_otsu(image)
            # Apply Thresholding
            thresholded_image = image > thresh
    2. CONTAMOS CONTORNOS
        Esta función encuentra las CURVAS de nivel o une puntos(pixeles) de una elevación igual o brillo en una matriz 2D, por encima de un valor de nivel dado.
            from skimage import measure
            # Find contours at a constant value of 0.8 (valor de nivel constante, definiremos más adelante eso)
            contours = measure.find_contours(thresholded_image, 0.8) # thresholded_image: imagen del umbral
            # Retorna una lista con -.,MNBVCXZ<odos los contornos de la IMAGEN, con las coordenadas a lo largo del contorno*.*-+
            +-JHBV
            for contour in contours:
                print(contour.shape) # CADA CONTORNO ES UN ARRAY DE FORMA (N, 2) N FILAS y COLUMNAS de coordinadas a lo largo del contorno.
            # Un contorno es la union de varios PUNTOS unidos entre si. (433, 2), ese 433 son la cantidad de puntos unidos, puede ser la figura más grande, ojo almacena CONTORNOS, osea hay contornos fuera y dentro del objeto

            * VALOR DE NIVEL CONSTANTE: es el valor que define que tan SENSIBLE va a detectar los objetos, de 0 a 1(mas sensible), puede detectar contornos mas COMPLEJOS, tenemos que encontrar el mejor.

    EJERCICIO, trae el TOTAL DE PUNTOS DE LOS DADOS:
    # Create list with the shape of each contour
    shape_contours = [cnt.shape[0] for cnt in contours]

    # Set 50 as the maximum size of the dots shape
    max_dots_shape = 50

    # Count dots in contours excluding bigger than dots size
    dots_contours = [cnt for cnt in contours if np.shape(cnt)[0] < max_dots_shape]

    # Shows all contours found 
    show_image_contour(binary, contours)

    # Print the dice's number
    print("Dice's dots number: {}. ".format(len(dots_contours)))

######   FINDING THE EDGE WITH CANNY ##########################
Veremos:
    - Detección de Bordes (Usado para dividir la imagen en áreas de objetos)
    - Esquinas
    - Caras de Personas

* BORDES
Con funciones de construcción, CANNY(astuta XD) es el método estandar más usado, por su precisión y menor tiempode ejecución comparado con el de SOBEL
- Se necesita una matriz Bidimensional para la función.(escala de grises)
- Se aplica la Función
- Luego se aplica un algoritmo Gausssiano para eliminar el ruido en la imagen, visto anteriormente. Esto está dentro de la misma función CANNY
    from skimage.feature import canny
    coins = color.rgb2gray(coins)
    canny_edges_0_5 = canny(coins, sigma=0.5) #Sigma, intensidad del filtro gaussiano mas cerca de 0 tiene más bordes(porque se elimina el ruido antes de continuar con los siguientes pasos del algoritmo). Default is 1
    show_image(canny_edges)

* ESQUINAS - Right around the corner
Utilizado apra extraer ciertas características e inferir en el contenido de una imagen. usado apara:
    - Detección de movimiento
    - Registro de imágenes
    - Seguimiento de video
    - unión e panorámicas
    - Modelado 3D
    - Reconocimiento de objetos

Finalmente es detectar un tipo de puntos de interes en una imagen, por ejemplo los bordes con canny, también son un tipo de puntos de interes.
Los bordes pueden ser tmb la intersección de DOS BORDES.
    - Corners
    - Matching Corners
    - Harris Corners Detector (Algoritmo bien usado para Computer Vision)
        from image.feature import corner_harris, corner_peaks
        # Convert image to grayscale
        image = rgb2gray(image)
        # Apply the Harris corners to the image / Al aplicar esto solo se muestran algúna slineas negras candidatas a aplicar la detecciónd de esquinas
        measure_image = corner_harris(image)
        # Show the image
        show_image(measure_image)

        # Find the coordinates of corners / Devuelve las coordinadas de los picos de las posibles esquinas
        coords = corner_peaks(corner_harris(image), min_distance=5) # Distancia minima entre las esquinas en 5 pixeles
        # Print the total corners
        print(len(coords))
        # Show the image with marks ind etected corners
        show_image_with_detected_corners(image, coords) #Función precargada de Matplotlib, parecida a la de show_image

        # Función para trazar las coordenadas de las esquinas  por marcas rojas cruzadas
        def show_image_with_corners(image, coords, title="Corners detected"):
            plt.imshow(image, interpolation='nearest', cmap='gray')
            plt.title(title)
            plt.plot(coords[:,1], coords[:,0], '+r', markersize=15)
            plt.axis('off')
            plt.show()

######  FACE DETECTION ##########################
Uses cases:
    - Filters
    - Auto Focus
    - Recommendations (like facebook)
    - Blur for privacy protection
    - To recognize emotions later on (primero reconocer caras, para luego reconcoer emociones)

CLASIFICADOR para detección:
    - Método ML: Cascada de Clasificadores (cascade classiider)

    from skimage.feature import Cascade
    # Load the trained file from the module root
    # Este marco necesita un archivo XML, desde el cual se pueden leer los datos entrenados
    # En este caso usaremos archivos de cara frontal que incluye la librería scikit-image
    trained_file = data.lbp_frontal_face_cascade_filename()
    # Initialize the detector cascade
    detector = Cascade(trained_file)

    Para aplicar el método de imágenes, necesitamos el método detect_multi_Scale, de la misma Cascade
    Lo que hace es detectar el objeto(cara) mediante unos cuadros mapeando toda la imágen, de diferentes escalas de tamaño(multiscale)
    # Apply detector on the image
        # scale_factor :multiplicación de la ventana de busqueda
        # step_ratio : 1 representa una busqueda exhaustiva(más lenta), + Alto peor busqueda, pero mayor tiempo de busqueda(INtervalos de 1,1 dan buenos resultados siempre)
        # min_size : Tamaño mínimo de la ventana
        # max_size : Tamaño máximo de la ventana
    detected = detector.detect_multi_scale(img=image, scale_factor=1.2, step_ratio=1, min_size=(10,10), max_size=(200,200))
    # El detector devolverá las coordenadas de la caja que contiene la cara
    print(detected)
        # 'r' : representa la fila posiciónd e la esquina superior izquieda de la ventana detectada
        # 'c' : posiciónd e la columna píxel
        # ancho de la ventana
        # alto de la ventana
    > Detected face: [{'r':115,'c':210, 'width':167, 'height':167}]

    def show_detected_face(result, detected, title='face image'):
        plt.show(result)
        img_desc = plt.gca()
        plt.set_cmap('gray')
        plt.title(title)
        plt.axis('off')

        for patch in detected:
            img_desc.add_patch(
                patches.Rectangle(
                    (patch['c'], patch['r'], patch['width'], patch['height'], fill=False, color='r', linewidth=2)
                )
            )
        plt.show()

######  REAL WORLD APPLICATIONS ##########################
- TURNING TO GRAYSCALE BEFORE DETECTING EDGES/CORNES
- REDUCING NOISE AND RESTORING IMAGES
- BLURRING FACES DETECTED
- APPROXIMATIONS OF OBJETS SIZES

* Casos: Detectar rostros en una imagen y anonimizarlos
    # Import Cascade of Classifiers and Gaussian filter
    from skimage.feature import Cascade
    from skimage.filters import gaussian
    # Detect the faces
    detected = detector.detect_multi_scale(img=image, scale_factor=1.2, step_ratio=1, min_size=(50,50), max_size=(100,100))

    # Función para cortar las caras de la imagen
    def getFace(d):
        ''' Extracts the face rectangle from the image using the coordinates of the detected'''
        # X and Y starting points of the face rectangle
        x, y = d['r'], d['c']  # d['r'] es la posición de la FILA de la ESQUINA SUPERIOR como la posición 'X', y 'C' que es la columna posición como 'Y' inciial.
        # The width and height of the face rectangle / Puntos donde agregaremos un ALTO y ANCHO para completar el rectangulo
        width, height = d['r'] + d['width'], d['c'] + d['height']
        # Extract the detected face
        face = image[X:width, y:height] # Especificamos estas dimensiones en al iamgen original
        return face

    def mergeBlurryFace(original, gaussian_image):
        # X and Y starting of the face rectangle /  ESPECIFICAMOS LOS PUNTOS X & Y INCIALES
        x, y = d['r'], d['c']
        # The width and height of the face rectangle /  ESPECIFICAMOS TMB EL ANCHO y ALTO
        width, height = d['r'] + d['width'], d['c'] + d['height']
        original[ x:width, y:height ] = gaussian_image
        return original

    # Extraemos las CARAS
    # For each detected face
    for d in detected:
*        # Obtain the face cropped from detected coordinates
        face = getFace(d)
        # Apply Gaussian filter to extracted face
        gaussian_face = gaussian(face, multichannel=True, sigma=10)
        # Merge this blurry face to our final image and show it
        resulting_image = mergeBlurryFace(image, gaussian_face)

RECORDAR que el modelo ha sido entrenado con caraas FRONTALES, se necesita cargar otras imagenes XML de librerías como OPENCV para entrenarlo por ejemplo con caras de gatos, o caras de perfil, etc....



----- CASE RESTORING

# Import the necessary modules
from skimage.restoration import denoise_tv_chambolle, inpaint
from skimage import transform

# Transform the image so it's not rotated
upright_img = transform.rotate(damaged_image, 20)

# Remove noise from the image, using the chambolle method
upright_img_without_noise = denoise_tv_chambolle(upright_img,weight=0.1, multichannel=True)

# Reconstruct the image missing parts
mask = get_mask(upright_img)
result = inpaint.inpaint_biharmonic(upright_img_without_noise, mask, multichannel=True)

show_image(result)

--------


1. Amazing work!
Amazing work! You have reached the final video. During the course you have learned many concepts and practiced a lot along the way, solving real-world problems.

2. Recap: What you have learned
    - From improving images contrast
    - restoring damaged ones with very few lines! 
    - You also applied filters
    - rotated, flipped and resized images
    - segmented using supervised and unsupervised methods
    - improved this segmentation using morphological operators! 
    - You created and reduced noise
    - detected edges, corners and faces
    - and mixed them up to solve difficult challenges! 
You will now be able to apply these techniques to many other use cases and keep extending your computer vision knowledge!

3. What's next?
So you have learned many useful methods that will let you process images with scikit-image or other image processing libraries. Like Open CV. We focused on many fundamental concepts so you can continue your progress from a practical experienced point. Some things that we didn't cover are tinting gray scale images, matching or approximation. Since it's a very large field, there are many more techniques for you to play around with!

4. Congrats!
Finally, thank you for completing the course! It's been a pleasure being with you along the way and I wish you the best of luck in your journey!

------------------

#####################################################################################################
BOTO 3 - AWS
#####################################################################################################


import json
import boto3
import pandas as pd
from io import BytesIO
import gzip
import re
# import sys
import subprocess
from datetime import datetime, timedelta
import os

# if sys.version_info[0] < 3:
#    from StringIO import StringIO # Python 2.x
# else:
#    from io import StringIO # Python 3.x

# Parametros de Conexion
S3_BUCKET_NAME_LOADED= 'belc-bigdata-landing-dlk-prd'
S3_PATH_FILES= 'lan-virtualcoach/input/data-hybris-datadeldia/'

# parte de nombre de archivo
dia = timedelta(days=1)
start_time = datetime.now()
if (start_time-dia).month < 10:
	str_month = "0"+str((start_time-dia).month)
else:
	str_month = str((start_time-dia).month)

if (start_time-dia).day < 10:
	str_day = "0"+str((start_time-dia).day)
else:
	str_day = str((start_time-dia).day)
str_year = str(start_time.year)
zipFile = str_year+"-"+str_month+"-"+str_day+".json.gz"
origin_zip_file = "belc-bigdata-landing-dlk-prd/lan-virtualcoach/input/data-hybris-datadeldia/"+zipFile
dest_zip_file = "s3://belc-bigdata-landing-dlk-prd/lan-virtualcoach/input/Register/"
remove_zip_file = "s3://belc-bigdata-landing-dlk-prd/lan-virtualcoach/input/Register/"+zipFile

# Conexion con el bucket de S3
s3 = boto3.client('s3')

# Definicion de variables
n = []
listfiles= []
NameFiles = []
# listado de archivos en el bucket
for obj in s3.list_objects(Bucket = S3_BUCKET_NAME_LOADED, Prefix = S3_PATH_FILES)['Contents']:
    listfiles.append(obj['Key'])

if len(obj['Key']) <= 46:
    no_hay_data
  

for files in listfiles[1:]:
    s3 = boto3.resource('s3')
    key= files #'lan-virtualcoach/input/data-hybris/2019-07-06.json.gz'
    s = key
    # filtro de solo archivos que están luego de los caracteres "data-hybris/"
    ss = re.findall('data-hybris-datadeldia/(.*)', s)
    for i in ss:
        # print(i, end="")
        data = i
        NameFiles.append(data)
        # print(data)
    obj = s3.Object(S3_BUCKET_NAME_LOADED, key)
    dato = obj.get()['Body'].read()
    gzipfile = BytesIO(dato)
    gzipfile = gzip.GzipFile(fileobj=gzipfile)
    content = gzipfile.read()
    json_str = content.decode("utf-8")
    # data = json.loads(json_str)
    number = json_str.count('\n')
    n.append(number)
    # print(n)
    # print(NameFiles)
    # raise e
# remove file from register
# client.Object(S3_BUCKET_NAME_LOADED,'lan-virtualcoach/input/Register/'+zipFile).delete()
# print('the zip file has been removed it from Register')

# Unir ambas listas de registros y files
d =  {'Files':NameFiles,'Registers':n}
print(NameFiles)
print(n)
# Transformar a dataframe
df = pd.DataFrame(d)
# imprimir el dataframe como csv
df.to_csv('Reporte_Registros.csv', sep='\t', encoding='utf-8', index=False)
# print
print(df)


#####################################################################################################


# PANDASSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS

------------------------------------------------------------------------ 
# CASTEAR DATOS COLUMNARES
------------------------------------------------------------------------ 
print(df.dtypes)

# Change data type of column 'Marks' from int64 to float64
df['Marks'] = df['Marks'].astype('float64')

# Change data type of column 'Age' from int64 to string i.e. object type
df['Age'] = df['Age'].astype('object')

df = df.astype({'Age': 'float64', 'Marks': 'object'})

misiones_sku_ = misiones_sku_[['sku','grupo_mision']].apply(lambda x: x.astype('object'))

try:
    df['Age'] = df['Age'].astype('abc')
except TypeError as e:
    print(e)
    
for col in ['parks', 'playgrounds', 'sports', 'roading']:
    public[col] = public[col].astype('category')
    
cols=data.select_dtypes(exclude='int').columns.to_list()
data[cols]=data[cols].astype('category')

if df[col].dtypes == object
------------------------------------------------------------------------ 


------------------------------------------------------------------------ 
# MANIPULAR VACIOS
------------------------------------------------------------------------ 
# replacing na values in college with No college
nba["College"].fillna("No College", inplace = True)

# replacing na values in college con valores de la fila anterior
nba["College"].fillna( method ='ffill', inplace = True)

# Borrar rows cuando una columna es vacía, como una llave principal
import numpy as np
df['Charge_Per_Line'] = df['Charge_Per_Line'].replace('-', np.nan) # reemplazo en caso de ser necesario
df = df.dropna(axis=0, subset=['Charge_Per_Line']) # luego lo borro

# You can assign it back to df to actually delete vs filter ing done above
df = df[(df > 0).all(axis=1)]

# This can easily be extended to filter out rows containing NaN s (non numeric entries):-
df = df[(~df.isnull()).all(axis=1)]
------------------------------------------------------------------------ 


------------------------------------------------------------------------ 
# DROP
------------------------------------------------------------------------ 

df = df.drop(df[df.score < 50].index)

# In place version (as pointed out in comments)
df.drop(df[df.score < 50].index, inplace=True)

# To remove all rows where column 'score' is < 50 and > 20
df = df.drop(df[(df.score < 50) & (df.score > 20)].index)

------------------------------------------------------------------------ 


------------------------------------------------------------------------ 
# RENAME COLUMNS / RENOMBRAR COLUMNAS
------------------------------------------------------------------------ 
df_renamed_multiple = data.rename(
    columns={
        'PassengerId': 'Id', 
        'Sex': 'Gender',
    }
)

df1 = pd.DataFrame(df1,columns=['State','Score'])
------------------------------------------------------------------------


# Selec columns by Types
------------------------------------------------------------------------
data.select_dtypes('int')
data.select_dtypes('float')
------------------------------------------------------------------------


# Select the duplicated values
------------------------------------------------------------------------
df_dup[df_dup.duplicated()]
df.drop_duplicates(subset = 'bio' & subset = 'center' )
------------------------------------------------------------------------


# GROUP BY
------------------------------------------------------------------------
data.groupby('Sex').agg({'PassengerId': 'count'})
data.groupby(['Pclass', 'Sex']).agg({'PassengerId': 'count'})

df_misiones_.groupby('Group').size().reset_index(name='row1')

# validar cuantas poc comparten el dos archivos o más
df_.groupby('poc')['file'].nunique().reset_index(name='row1').query("row1 > 0")

# Contar cuantas POC by SKU repiten registros por SKU
poc_by_sku.groupby(['poc','sku'])['sku'].count().reset_index(name='row1').query("row1 > 1")

df[['Gender','Exited']].groupby('Gender').mean()
df[['Gender','Exited']].groupby('Gender').agg(['mean','count'])
df[['Gender','Geography','Exited']].groupby(['Gender','Geography']).mean()
df[['Gender','Geography','Exited']].groupby(['Gender','Geography']).mean().sort_values(by='Exited')
df[['Gender','Geography','Exited']].groupby(['Gender','Geography']).mean().sort_values(by='Exited', ascending=False)
df[['Geography','Age','Tenure']].groupby(['Geography']).agg(['mean','max'])
df[['Exited','Geography','Age','Tenure']].groupby(['Exited','Geography']).agg(['mean','count'])
df[['Exited','Geography','Age','Tenure']].groupby(['Exited','Geography']).agg(['mean','count']).sort_values(by=[('Age','mean')])
df[['Exited','IsActiveMember','NumOfProducts','Balance']].groupby(['Exited','IsActiveMember'], as_index=False).mean()

# Incluye NULOS OJO
df[['Geography','Exited']].groupby('Geography', dropna=False).agg(['mean','count'])
------------------------------------------------------------------------


# JOINS
------------------------------------------------------------------------
# Union All
pd.concat([df1, df2, df3, df4], ignore_index=True)

df_misiones = df_misiones_tmp.merge(df_skus_tmp, how='left', left_on=['poc','sku'], right_on=['poc','sku'])
------------------------------------------------------------------------


# TRANSFOR DATA / RENAME VALUES
------------------------------------------------------------------------
data['Survived'].map(lambda x: 'Survived' if x==1 else 'Not-Survived')
data['Survived'].map({1: 'Survived', 0: 'Not-Survived'})
data['Sex'].replace(['male', 'female'], ["M", "F"])

df_misiones_tmp['poc'] = df_misiones_tmp['poc'].astype(str).str.strip()
------------------------------------------------------------------------



# Crear dataset dummy
------------------------------------------------------------------------
# data dummy
import numpy as np
df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'],
                     'value': np.random.randn(4)})


raw_data = {'first_name': ['Sheldon', 'Raj', 'Leonard', 'Howard', 'Amy'],
                'last_name': ['Copper', 'Koothrappali', 'Hofstadter', 'Wolowitz', 'Fowler'],
                'age': [42, 38, 36, 41, 35],
                'Comedy_Score': [9, 7, 8, 8, 5],
                'Rating_Score': [25, 25, 49, 62, 70]}

df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age',
                                           'Comedy_Score', 'Rating_Score'])
------------------------------------------------------------------------

# Filter Data
------------------------------------------------------------------------
df['Comedy_Score'].where(df['Rating_Score'] < 50)
re_misiones_no[re_misiones_no['sku'].isin(['NO27', 'NO19','NO27'])].head(5)
------------------------------------------------------------------------


# SQL SPARK
------------------------------------------------------------------------
spark = SparkSession.builder.appName('df_').getOrCreate()
df_sql = spark.createDataFrame(df_) 
df_sql.createOrReplaceTempView("df_sql_")
tabla = spark.sql("""
select
  poc,count(distinct file)
from df_sql_
group by poc
having count(distinct file) > 1
""").toPandas().head(10)
------------------------------------------------------------------------


# EXPORT
------------------------------------------------------------------------
pip install openpyxl
sheets_names = pd.ExcelFile('reading_excel_file.xlsx').sheet_names
sheets_names

df = pd.read_excel('reading_excel_file.xlsx', 
                   sheet_name='Purchase Orders 1',
                  usecols='C:F',
                  # usecols='A:B, H:I'
                  )

df = pd.read_excel("sales_excel.xlsx", 
                   sheet_name='Sheet1',
                  header=5)

# skip the first 2 rows when reading the file
df = pd.read_excel('reading_excel_file.xlsx', 
                   sheet_name='Purchase Orders 1',
                  skiprows=2)

# skip 3 rows from the end.
df = pd.read_excel('reading_excel_file.xlsx', 
                   sheet_name='Purchase Orders 1',
                  skipfooter=3)

data.to_csv('/path/to/save/the/data.csv', index=False)
------------------------------------------------------------------------


# COMO COMPLETAR DE COLUMNAS IGUAL A OTRO DATAFRAME, 


/***************************************************************************************************/
[ BACKUSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS ] 
/***************************************************************************************************/

df = pd.read_excel(f"/dbfs/mnt/adls_maz131/analytics_zone/MAZ/PE/POP/Pop Ouput/202211/Base_ejecucion_noviembre_balanceada_v2.xlsx",
                  sheet_name='BASE_INICIAL'
                  )
print( df.shape )
print( df.columns )

df=pd.read_csv('/dbfs/mnt/adls_maz131/analytics_zone/MAZ/PE/POP/Pop Ouput/202210/output_challengues/raw/master_table_pop_202211_v2.csv'
               ,encoding='latin1').dropna(axis=0, how="all").drop_duplicates()
print( df.shape )
print( df.columns )


df_sku.columns = ['CODIGO SKU','brand','Pack','Package']

df_all_misiones = df_all_misiones[ df_all_misiones.brand!='0' ].copy()


/***************************************************************************************************/
# REEMPLAZAR O ACTUALIZAR UN CAMPO CON CONDICION DE OTRO
df_all_misiones.loc[df_all_misiones['sku'].isin([18083,18082]),'brand']='P.Callao'
df_all_misiones.loc[df_all_misiones['sku'].isin([18083,18082]),'Pack']='355 CAN'
df_all_misiones.loc[df_all_misiones['sku'].isin([17893]),'brand']='P.Callao'
df_all_misiones.loc[df_all_misiones['sku'].isin([17893]),'Pack']='305 RB'
/***************************************************************************************************/




/***************************************************************************************************/
spark = SparkSession.builder.appName('df_mision_ejecucion').getOrCreate()
df_mision_ejecucion_ = spark.createDataFrame(df_mision_ejecucion)
df_mision_ejecucion_.write.mode("overwrite").saveAsTable("abi_lh_portfolio.lh_pop_pe_ejecucion")

%sql
DELETE FROM abi_lh_portfolio.b2b2c_audience_cat_PE WHERE periodo = '202210';
INSERT INTO abi_lh_portfolio.b2b2c_audience_cat_PE
select *  from tmp_testops WHERE periodo = '202210';

/***************************************************************************************************/


df_ = spark.sql("""
   select * from "tu tabla"
""")

df_.toPandas().to_csv('/dbfs/mnt/adls_maz131/analytics_zone/MAZ/PE/Lighthouse_BIT/tu_csv_aqui.csv')
# b2b2c_audience_pocs.toPandas().head()

/***************************************************************************************************/



df_cobertura_adic=df_cobertura_adic[(df_cobertura_adic.flag_pop==1)&
                                    (df_cobertura_adic.type!='Generate Demand')&
                                    (df_cobertura_adic.flag_prioridad==1)].copy()

df_cobertura_adic.shape


/***************************************************************************************************/


b2b2c_audience_pocs = spark.sql("""
   select 
    'PE' as country,
    poc as client,
    audience_key,
    'BottomUp' as audience,
    brand_x as subaudience,
    month,
    year,
    proba as probability,
    '' as vol_potential,
    Group as control_poc,
    StartDate-1 as timestmp
    --count(*)
  from tmp_202209
  --where sku_final <> 'mkpl'
  where Group in ('Test','Test sin Cambios')
  and length(sku_final) in (4,5)
  group by 1,2,3,4,5,6,7,8,9,10,11
""")

df_b2b2c_audience_pocs = b2b2c_audience_pocs.toPandas()
df_b2b2c_audience_pocs.head()


