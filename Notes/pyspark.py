

# ***************************************************
# IMPORTAR import
# ***************************************************
grupos_ = spark.sql(""" select * from abi_lh_portfolio.lh_pop_pe_group_client """).toPandas()

df_ = spark.read.csv('dbfs:/mnt/adls_maz131/analytics_zone/MAZ/PE/master_tables/deal_condition/dc_sales.csv',header=True, inferSchema=True)

# de pandas
df_client=pd.read_csv('/dbfs/mnt/adls_maz131/analytics_zone/MAZ/PE/POP/Data/PEDemographics/TelefonicaClient.csv')
df_client_sparkDF=spark.createDataFrame(df_client)
df_client_sparkDF=df_client_sparkDF.select(df_client_sparkDF.Cliente,df_client_sparkDF.Latitud,df_client_sparkDF.Longitud)

# pyspark
df_mt = spark.read.\
        option("delimiter",",").\
        option("header","true").\
        option("encoding", "windows-1252").\
        option("inferSchema", "true").\
        option("header", "true").\
        csv("dbfs:/mnt/adls_maz131/analytics_zone/MAZ/PE/POP/Pop Ouput/202212/master_table_pop.csv")

df_mt = df_mt.withColumn('date_process_db_lhpe', current_timestamp())
df_mt = df_mt.dropDuplicates()
df_mt = df_mt.withColumn("poc",col("poc").cast(IntegerType())) \
        .withColumn("sku",col("sku").cast(IntegerType()))
        
        

# /***************************************************************************************************/
spark = SparkSession.builder.appName('df_mision_ejecucion').getOrCreate()
df_mision_ejecucion_ = spark.createDataFrame(df_mision_ejecucion)
df_mision_ejecucion_.write.mode("overwrite").saveAsTable("abi_lh_portfolio.lh_pop_pe_ejecucion")

%sql
DELETE FROM abi_lh_portfolio.b2b2c_audience_cat_PE WHERE periodo = '202210';
INSERT INTO abi_lh_portfolio.b2b2c_audience_cat_PE
select *  from tmp_testops WHERE periodo = '202210';

# /***************************************************************************************************/
# encoding
df_mt = spark.read.\
        option("delimiter",",").\
        option("header","true").\
        option("encoding", "windows-1252").\
        option("inferSchema", "true").\
        option("header", "true").\
        csv("dbfs:/mnt/adls_maz131/analytics_zone/MAZ/PE/POP/Pop Ouput/202212/master_table_pop.csv")

df_mt = df_mt.withColumn('date_process_db_lhpe', current_timestamp())
df_mt = df_mt.dropDuplicates()
df_mt = df_mt.withColumn("poc",col("poc").cast(IntegerType())) \
        .withColumn("sku",col("sku").cast(IntegerType()))

df_mt = df_mt.withColumn("periodo", lit(202212)) # VARIABLE EN DURO
display(df_mt)





# ***************************************************
# EXPORTAR export
# ***************************************************
spark = SparkSession.builder.appName('reporte_telesales_').getOrCreate()
reporte_telesales_spark = spark.createDataFrame(reporte_telesales_)
reporte_telesales_spark.select("*").write.format('com.crealytics.spark.excel').option("header","true").option("inferSchema","true").save(output_report, mode="overwrite")
print(output_report)


# ***************************************************
# FUNCTIONS / FUNCIONES
# ***************************************************
df.count()
df.columns
df.types
df.schema
df.printSchema()


# ***************************************************
# SELECT / SELECCIONAR
# ***************************************************
df.select("id", "name").show()


# ***************************************************
# FILTER / FILTRAR
# ***************************************************
df.filter(df['id'] == 1).show()
df.filter(df.id == 1 ).show()
df.filter(col("id") == 1).show()
df.filter("id = 1").show()



# ***************************************************
# DROP / BORRAR / ELIMINAR
# ***************************************************
newdf = df.drop("id")  # es innutable nos e eilimna, sino creo una copia
newdf.show(2)



# ***************************************************
# AGREGACIONES  /GROUP BY 
# ***************************************************
(
    df.groupBy("dept")
        .agg(
            count("salary").alias("count"),
            sum("salary").alias("sum"),
            max("salary").alias("max"),
            min("salary").alias("min"),
            avg("salary").alias("avg"), 
        )
)

from dataclasses import dataclass
from os import fsdecode
import secrets
from tkinter.ttk import Notebook
from pyspark.sql.functions import org
display(
    diamonds
        .select("color", "price")
        .groupBy("color")
        .agg( avg("price") )
        .sort("color")
)


# ***************************************************
# SORTING
# ***************************************************
df.sort("salary").show(5)
df.sort(desc("salary")).show(5)


# ***************************************************
# CASTEO de variables, tipo de datos
# ***************************************************

# withColumn() – Change Column Type
from pyspark.sql.functions import col
from pyspark.sql.types import StringType,BooleanType,DateType

df2 = df.withColumn("age",col("age").cast(StringType())) \
    .withColumn("isGraduated",col("isGraduated").cast(BooleanType())) \
    .withColumn("jobStartDate",col("jobStartDate").cast(DateType()))
    
df2.printSchema()

# selectExpr() – Change Column Type
df3 = df2.selectExpr("cast(age as int) age",
    "cast(isGraduated as string) isGraduated",
    "cast(jobStartDate as string) jobStartDate")

df3.printSchema()
df3.show(truncate=False)

# SQL – Cast using SQL expression
df3.createOrReplaceTempView("CastExample")

df4 = spark.sql("SELECT STRING(age),BOOLEAN(isGraduated),DATE(jobStartDate) from CastExample")

df4.printSchema()
df4.show(truncate=False)


# ***************************************************
# NULOS / null / missing
# *************************************************** 
# Los valores NULL siempre son difíciles de manejar independientemente del Framework o lenguaje que usemos. 
# Aquí en Spark tenemos pocas funciones específicas para lidiar con valores NULL.

# isnull()
    # Esta función nos ayudará a encontrar los valores nulos para cualquier columna dada. 
    # Por ejemplo si necesitamos encontrar las columnas donde las columnas id contienen los valores nulos.
    newdf = df.filter(df["dept"].isNull()) # filtro los valores Nulos
    newdf.show()
# isNotNull()
    newdf = df.filter(df["dept"].isNotNull())
    newdf.show()
    
# fillna()
    # Esta función nos ayudará a reemplazar los valores nulos.
    newdf = df.fillna("INVALID", ["dept"])
    newdf.show()
    
# dropna()
    # Esta función nos ayudará a eliminar las filas con valores nulos.
    newdf = df.dropna() # Remove all rows which contains any null values.
    newdf.show()
    # Elimina todas las filas que contienen todos los valores nulos.
    newdf = df.dropna(how = "all")
    newdf.show()
    # Elimina todas las filas que contiene al menos un valor nulo
    newdf = df.dropna(how = "any") # default
    newdf.show()
    # Remove all rows where columns : dept is null.
    newdf = df.dropna(subset = "dept")
    newdf.show()

# ***************************************************
# COLUMNA DERIVADA - NUEVA COLUMNA , nuevo campo - NEW COLUMN
# *************************************************** 
df.withColumn("bonus", col("salary")* .1).show(5)

# Add new constanct column
from pyspark.sql.functions import lit
df.withColumn("bonus_percent", lit(0.3)) \
  .show()

# Add New column with NULL
df.withColumn("DEFAULT_COL", lit(None)) \
  .show()

#Add column from existing column
df.withColumn("bonus_amount", df.salary*0.3) \
  .show()
#Add column by concatinating existing columns
from pyspark.sql.functions import concat_ws
df.withColumn("name", concat_ws(",","firstname",'lastname')) \
  .show()

#Add Column Value Based on Condition
from pyspark.sql.functions import when
df.withColumn("grade", \
   when((df.salary < 4000), lit("A")) \
     .when((df.salary >= 4000) & (df.salary <= 5000), lit("B")) \
     .otherwise(lit("C")) \
  ).show()

# Add Column When not Exists on DataFrame
if 'dummy' not in df.columns:
   df.withColumn("dummy",lit(None))

# Add column using select
df.select("firstname","salary", lit(0.3).alias("bonus")).show()
df.select("firstname","salary", lit(df.salary * 0.3).alias("bonus_amount")).show()
df.select("firstname","salary", current_date().alias("today_date")).show()

#Add columns to DataFrame using SQL
df.createOrReplaceTempView("PER")
df2=spark.sql("select firstname,salary, '0.3' as bonus from PER")
df3=spark.sql("select firstname,salary, salary * 0.3 as bonus_amount from PER")
df4=dfspark.sql("select firstname,salary, current_date() as today_date from PER")
df5=spark.sql("select firstname,salary, " +
          "case salary when salary < 4000 then 'A' "+
          "else 'B' END as grade from PER")

# JOINS
# **********************************************************************************************************
df.join(deptdf, df["dept"] == deptdf["id"]).show()
df.join(deptdf, df["dept"] == deptdf["id"], "left_outer").show()
df.join(deptdf, df["dept"] == deptdf["id"], "right_outer").show()
df.join(deptdf, df["dept"] == deptdf["id"], "outer").show()


# **********************************************************
# QUERY SQL
# **********************************************************

# Es necesario registrarlo a una vista temporal
# esta vida util temporal está asociada al Spark Session
df.createOrReplaceTempView("temp_table")

# Execute SQL-Lite Query
spark.sql(" select * from  temp_table where id = 1 ").show()



# TABLAS HIVE
# **********************************************************************************************************

# LEYENDO TABLA HIVE
df = spark.table("DB_NAME.TBL_NAME")

# GUARDAR DATAFRAME EN TABLA HIVE (TABLA INTERNA)
    # por defaultguardará el dataframe como una tabla interna (administrada por hive)
df.write.saveAsTable("DB_NAME.TBL_NAME")
    # También podemos seleccionar un argumento "modo" con overwrite, "append", "error", etc...
df.write.saveAsTable("DB_NAME.TBL_NAME", modo="overwrite")

# GUARDAR EL DATAFRAME COMO UNA TABLA EXTERNA HIVE
df.write.saveAsTable("DB_NAME.TBL_NAME", path=<location_of_external_table>)


# READ / CREAR DATAFRAME DESDE UN CSV
# **********************************************************************************************************
df = spark.read.csv("path_to_csv_file", sep="|", header=True, inferSchema=True) # InferSchema, infiere el tipo de dato de cada variable

spark = SparkSession.builder.appName('POP Census').getOrCreate()



# GUARDAR / GUARDAR DATAFRAME A UN CSV / EXPORTAR
# **********************************************************************************************************
df.write.csv("path_to_CSV_file", sep="|", header=True, mode="overwrite")


# LEER TABLAS DE BASE DE DATOS RELACIONAL
# **********************************************************************************************************
    # url = a JDBC URL of  the form  jdbc:subprotocol:subname
relational_df = spark.read.format('jdbc')
                    .options(url=url, dbtable=<TBL_NAME>, user=<USER_NAME>, password=<PASSWORD>)
                    .load()
                    
# GUARDAR EN UNA TABLA DE BASE DE DATOS RELACIONAL
# **********************************************************************************************************
    # url = a JDBC URL of  the form  jdbc:subprotocol:subname
relational_df.write.format('jdbc')
                    .options(url=url, dbtable=<TBL_NAME>, user=<USER_NAME>, password=<PASSWORD>)
                    .mode('overwrite')
                    .save()
                    
                    
                    
# SPARK TIENE FUNCIONES DE OPTIMIZAR EL RENDIMIENTO Y REALIZAR TRANSFORMACIONES COMPLEJAS
# Expresiones de selectExpr(), UDF, cache(), etc
# **********************************************************************************************************

    # Una de las técnicas de optimización  son los métodos cache() y persist().
    # Estos métodos se usan para almacenar un cálculo intermedio de un RDD, Dataframe y Dataset para que puedan reutilizarse en acciones posteriores.
    # Spark tiene un optimidizador de consultas Catalyst, que  tiene mejor performance

    import findspark
    findspark.init()
    
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import *
    from pyspark.sql.functions import broadcast
    from pyspark.sql.types import *
    
    # creo la spark session
    spark = SparkSession.builder.getOrCreate()
    
    df = spark.createDataFrame(emp, ["id", "name", "dept", "salary"])
    deptdf = spark.createDataFrame(dept, ["id", "name"]) 

    # Create Temp Tables
    df.createOrReplaceTempView("empdf")
    deptdf.createOrReplaceTempView("deptdf")

    # Save as HIVE tables.
    df.write.saveAsTable("hive_empdf", mode = "overwrite")
    deptdf.write.saveAsTable("hive_deptdf", mode = "overwrite")
    
    # BROADCAST JOIN
    # **********************************************************************************************************
    # Sirve para hace run join optimizado
    # El tamaño de la tabla de difusión es de 10 MB. Sin embargo, podemos cambiar el umbral hasta 8GB según la documentación oficial de Spark 2.3.
    # * Podemos verificar el tamaño de la tabla de transmisión de la siguiente manera:
    
        # Podemos verificar el tamaño de la tabla de transmisión de la siguiente manera:
        size = int(spark.conf.get("spark.sql.autoBroadcastJoinThreshold")) / (1024 * 1024)
        print("Default size of broadcast table is {0} MB.".format(size))
        
        # Podemos establecer el tamaño de la tabla de transmisión para que diga 50 MB de la siguiente manera:
        spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 50 * 1024 * 1024)
        
        # Considere que necesitamos unir 2 Dataframes.
        # small_df: DataFrame pequeño que puede caber en la memoria y es más pequeño que el umbral especificado.
        # big_df: DataFrame grande que debe unirse con DataFrame pequeño.
        join_df = big_df.join(broadcast(small_df), big_df["id"] == small_df["id"])
        
    
        # ALMACENAMIENTO EN CACHÉ
        # **********************************************************************************************************
        # Podemos usar la función de caché / persistencia para mantener el marco de datos en la memoria. Puede mejorar significativamente el 
        # rendimiento de su aplicación Spark si almacenamos en caché los datos que necesitamos usar con mucha frecuencia en nuestra aplicación.
        df.cache()
        df.count()
        print("Memory Used : {0}".format(df.storageLevel.useMemory)) # ver almacenamiento en memoria
        print("Disk Used : {0}".format(df.storageLevel.useDisk)) # ver almacenamiento en disco
        
        # Cuando usamos la función de caché, usará el nivel de almacenamiento como Memory_Only hasta Spark 2.0.2. Desde Spark 2.1.x es Memory_and_DISK.
        # Sin embargo, si necesitamos especificar los distintos niveles de almacenamiento disponibles, podemos usar el método persist( ). Por ejemplo, 
        # si necesitamos mantener los datos solo en la memoria, podemos usar el siguiente fragmento.
        from pyspark.storagelevel import StorageLevel # mantener los datos solo en memoria
        deptdf.persist(StorageLevel.MEMORY_ONLY) # mantener los datos solo en memoria
        
        deptdf.count()
        print("Memory Used : {0}".format(df.storageLevel.useMemory))
        print("Disk Used : {0}".format(df.storageLevel.useDisk))
        
        # No persistir
        # También es importante eliminar la memoria caché de los datos cuando ya no sean necesarios.
        df.unpersist() # elimina todos los datos alacenados en memoria
        sqlContext.clearCache() # elimina todos los datos alacenados en caché
        
    
    # SQL EXPRESSIONS (expr - selectExpr)
    # **********************************************************************************************************    
    # También podemos usar la expresión  SQL para la manipulación de datos. Tenemos la función expr y tmb 
    # una variante de un método de selección como selectExpr para la evaluación de expresiones SQL
    from pyspark.sql.functions import expr

    # Intentemos categorizar el salario en Bajo, Medio y Alto según la categorización a continuación.
    # 0-2000: salario_bajo
    # 2001 - 5000: mid_salary
    #> 5001: high_salary
    cond = """case when salary > 5000 then 'high_salary'
                else case when salary > 2000 then 'mid_salary'
                        else case when salary > 0 then 'low_salary'
                            else 'invalid_salary'
                                end
                            end
                    end as salary_level"""

    # Utilizando Expr
    newdf = df.withColumn("salary_level", expr(cond))
    newdf.show()
    
    # Utilizando selectExpr
    newdf = df.selectExpr("*", cond)
    newdf.show()
    
    
    # UDF ( Funciones definidas por el usuario)
    # **********************************************************************************************************    
    # aquí s epuede crear funciones Python y utilizarlas en spark
    # A menudo necesitamos escribir la función en función de nuestro requisito muy específico. Aquí podemos aprovechar las udfs. 
    # Podemos escribir nuestras propias funciones en un lenguaje como python y registrar la función como udf, luego podemos usar la función para operaciones de DataFrame.
    # Función de Python para encontrar el nivel_salario para un salario dado.
    def detSalary_Level(sal):
        level = None

        if(sal > 5000):
            level = 'high_salary'
        elif(sal > 2000):
            level = 'mid_salary'
        elif(sal > 0):
            level = 'low_salary'
        else:
            level = 'invalid_salary'
        return level
    
    # Luego registre la función "detSalary_Level" como UDF.
    sal_level = udf(detSalary_Level, StringType())
    
    # Aplicar función para determinar el salario_level para un salario dado.
    newdf = df.withColumn("salary_level", sal_level("salary"))
    newdf.show()
    
    

        
        
        
######################## DATABRICKS ########################

### UTILS
# *************************************************************************
# Los comandos utils de databricks permiten realizar potentes tareas
    # - Generar diferentes objetos en la parte del almacenameinto
    # - Encadenar Notebooks
    # - Manejar los secretos de los workspaces
    # - etc...
# Tenemos diferentes categorías de los Utils:
dbutils.help() # to interact with the rest of Databricks.
    # credentials: DatabricksCredentialUtils -> Utilities for interacting with credentials within notebooks
    # data: DataUtils -> Utilities for understanding and interacting with datasets (EXPERIMENTAL)
    # fs: DbfsUtils -> Manipulates the Databricks filesystem (DBFS) from the console
    # library: LibraryUtils -> Utilities for session isolated libraries
    # meta: MetaUtils -> Methods to hook into the compiler (EXPERIMENTAL)
    # notebook: NotebookUtils -> Utilities for the control flow of a notebook (EXPERIMENTAL)
    # preview: Preview -> Utilities under preview category
    # secrets: SecretUtils -> Provides utilities for leveraging secrets within notebooks
    # widgets: WidgetsUtils -> Methods to create and get bound value of input widgets inside notebooks
dbutils.fs.help() # for working with FileSystems.
# fsutils
    # cp(from: String, to: String, recurse: boolean = false): boolean -> Copies a file or directory, possibly across FileSystems
    # head(file: String, maxBytes: int = 65536): String -> Returns up to the first 'maxBytes' bytes of the given file as a String encoded in UTF-8
    # ls(dir: String): Seq -> Lists the contents of a directory
    # mkdirs(dir: String): boolean -> Creates the given directory if it does not exist, also creating any necessary parent directories
    # mv(from: String, to: String, recurse: boolean = false): boolean -> Moves a file or directory, possibly across FileSystems
    # put(file: String, contents: String, overwrite: boolean = false): boolean -> Writes the given String out to a file, encoded in UTF-8
    # rm(dir: String, recurse: boolean = false): boolean -> Removes a file or directory
# mount
    # mount(source: String, mountPoint: String, encryptionType: String = "", owner: String = null, extraConfigs: Map = Map.empty[String, String]): boolean -> Mounts the given source directory into DBFS at the given mount point
    # mounts: Seq -> Displays information about what is mounted within DBFS
    # refreshMounts: boolean -> Forces all machines in this cluster to refresh their mount cache, ensuring they receive the most recent information
    # unmount(mountPoint: String): boolean -> Deletes a DBFS mount point
dbutils.data.help()
    # cp(from: <from>, to: <to>, recurse: boolean = false): boolean
    # Example: cp("/mnt/my-folder/a", "s3n://bucket/b")
    
    
    ### UTILIDAD DE DATOS (dbutils.data)
    # *************************************************************************
    dbutils.data.help()
    
    # SUMMARIZE, ejemplo de estadísticos de un dataframe
    df = spark.read.format('csv').load(
        '/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv',
        header=True,
        inferSchema=True
    )
    dbutils.data.summarize(df)
    
    
    ### UTILIDAD DEL SISTEMA DE ARCHIVOS (dbutils.data)
    # *************************************************************************
    dbutils.fs.cp("/FileStore/old_file.txt", "/tmp/new/new_file.txt") # copiar archivo
    dbutils.fs.head("/tmp/my_file.txt", 25) # leer los primeros registros
    dbutils.fs.ls("/tmp") # listar archivos de la carpeta
    dbutils.fs.mkdirs("/tmp/parent/child/grandchild") # crear carpeta
    dbutils.fs.mv("/FileStore/my_file.txt", "/tmp/parent/child/grandchild") # Mover archivo a una carpeta
    dbutils.fs.put("/tmp/old_file.txt", "Hello, Databricks!", True) # escribir en un archivo
    dbutils.fs.rm("/tmp/old_file.txt") # borrar archivo
    
    
    ### UTILIDAD DE LIBRERÍAS
    # *************************************************************************
    dbutils.library.installPyPI("numpy") # instalar una librería en el entorno
    dbutils.library.restartPython() # reiniciar ppara utilizar python e importarlo
    import numpy # importamos la librería
    dbutils.library.list() # listamos las librerías instaladas
    
    import sys
    !{sys.executable} -m pip install unidecodedata
    
    ### UTILIDAD DE NOTEBOOK
    # *************************************************************************
    # los notebook son modulares, podemos tener varios notebooks
    dbutils.notebook.help()
        # exit(value: String): void -> This method lets you exit a notebook with a value
        # run(path: String, timeoutSeconds: int, arguments: Map): String -> This method runs a notebook and returns its exit value
    dbutils.notebook.run("trial", 60)
    dbutils.notebook.exit("trial")
    
    ### UTILIDAD DE SECRETOS
    # *************************************************************************
    # Provides utilities for leveraging secrets within notebooks. Databricks documentation for more info.
    dbutils.secrets.help()
    dbutils.secrets.get(scope="my-scope", key="my-key")
    my_secret = dbutils.secrets.getBytes(scope="my-scope", key="my-key")
    my_secret.decode("utf-8")
    dbutils.secrets.list("my-scope")
    
    
    ### UTILIDAD DE WIDGETS
    # *************************************************************************
    dbutils.widgets.combobox(
        name='fruits_combobox',
        defaultValue='banana',
        choices=['apple', 'banana', 'coconut', 'dragon fruit'],
        label='Fruits'
    )
    print(dbutils.widgets.get("fruits_combobox"))
    
    dbutils.widgets.dropdown(
        name='toys_dropdown',
        defaultValue='basketball',
        choices=['alphabet blocks', 'basketball', 'cape', 'doll'],
        label='Toys'
    )
    print(dbutils.widgets.get("toys_dropdown"))
    
    dbutils.widgets.get('fruits_combobox')
    
    dbutils.widgets.removeAll()
    
    dbutils.widgets.text(
        name='your_name_text',
        defaultValue='Enter your name',
        label='Your name'
    )

    print(dbutils.widgets.get("your_name_text"))
    

    
    cols_join_data=''
    tablas=['df3_1', 'cte_ageb', 'df_p','df_c','df_pib','df_idh','df_ind_edu','df_ind_salud','df_ind_ingr','pct_Marca','pct_Cupo','pct_Linea','df4','df5','df6','pct_Mix','pct_Mix2']
    d={'cve_ent':2,'cve_mun':3,'cve_loc':4}
    
    for i in tablas:
        #res=spark.sql("select * from abi_maz_lighthouse_rpt.%s limit 1"%(i))
        res=spark.sql("select * from %s limit 1"%(i))
    
    for c in res.columns:
        if c.lower()=='null':
        continue
        if c.lower()=='CodigoDeDestinatarioCorto'.lower():
        continue
        else:
        if c in d.keys():
            c = "lpad(string(%s.%s),%d,'0') %s"%(i,c,d[c],c)
        else:
            c=i+'.'+c
        if cols_join_data=='':
            cols_join_data=c
        else:
            cols_join_data=cols_join_data+','+c
    cols_join_data='cte_ageb.codigodedestinatariocorto,'+cols_join_data
    
    
### Rename columns, renombrar columna
df2 = df.withColumnRenamed("dob","DateOfBirth") \
    .withColumnRenamed("salary","salary_amount")
    
# DROP Duplicate  /borrar o eliminar duplicados

#utilitarios sql pyspark
    
spark.sql("select * from abi_maz_lighthouse_rpt.df_c0_aux").groupBy("CodigoDeDestinatarioCorto").pivot("TA_RANGO").agg(first("val")).write.saveAsTable('abi_maz_lighthouse_rpt.df_c0')

columns=spark.sql("select * from abi_maz_lighthouse_rpt.aux0 limit 1")
for i in columns.columns:
  spark.sql("update abi_maz_lighthouse_rpt.aux0 set %s=0 where %s is null"%(i,i))
  
  
%python
from pyspark.sql.functions import avg

display(diamonds.select("color","price").groupBy("color").agg(avg("price")).sort("color"))