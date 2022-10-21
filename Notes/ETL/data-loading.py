import pandas as pd  

# From CSV
df = pd.read_csv("path")

# From Excel
df = pd.read_excel('/path')

# From database (sqlite)
import sqlite3
conn = sqlite3.connect("foo.db")
cur = conn.cursor()
#Check how many tables are there in the database
cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
#SQL Query to pandas dataframe
df = pd.read_sql_query("select * from bar limit 5;", conn)


Python
pd.read_csv(filename) # From a CSV file
pd.read_table(filename) # From a delimited text file (like TSV)
pd.read_excel(filename) # From an Excel file
pd.read_sql(query, connection_object) # Reads from a SQL table/database
pd.read_json(json_string) # Reads from a JSON formatted string, URL or file.
pd.read_html(url) # Parses an html URL, string or file and extracts tables to a list of dataframes
pd.read_clipboard() # Takes the contents of your clipboard and passes it to read_table()
pd.DataFrame(dict) # From a dict, keys for columns names, values for data as lists




# Databricks

## Lectura de datos desde blob Storage
misiones_ejecucion_ = pd.read_csv('/dbfs/mnt/adls_maz131/analytics_zone/MAZ/PE/POP/Pop Ouput/202210/output_format_revenue/raw/misiones/misiones_ejecucion_afiches.csv')
print(misiones_ejecucion_.columns)
misiones_sku_ = pd.read_csv('/dbfs/mnt/adls_maz131/analytics_zone/MAZ/PE/POP/Pop Ouput/202210/output_format_revenue/raw/misiones/mision_sku.csv')
print(misiones_sku_.columns)
misiones_sku_aroon_ = pd.read_excel('/dbfs/mnt/adls_maz131/analytics_zone/MAZ/PE/POP/Pop Ouput/202210/output_format_revenue/raw/misiones/mision_sku_aroon.xlsx')
print(misiones_sku_aroon.columns)
