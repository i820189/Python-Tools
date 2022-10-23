df.count()
df.columns
df.types
df.schema
df.printSchema()


# SELECT
# **********************************************************************************************************
df.select("id", "name").show()



# FILTER
# **********************************************************************************************************
df.filter(df['id'] == 1).show()
df.filter(df.id == 1 ).show()
df.filter(col("id") == 1).show()
df.filter("id = 1").show()



# DROP
# **********************************************************************************************************
newdf = df.drop("id")  # es innutable nos e eilimna, sino creo una copia
newdf.show(2)



# AGGREFATIONS
# **********************************************************************************************************
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


# SORTING
# **********************************************************************************************************
df.sort("salary").show(5)
df.sort(desc("salary")).show(5)


# COLUMNAS DERIVADAS
# **********************************************************************************************************
df.withColumn("bonus", col("salary")* .1).show(5)


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
df = spark.read.csv("path_to_csv_file", sep="|", header=True, inferSchema=True)



# GUARDAR / GUARDAR DATAFRAME A UN CSV
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