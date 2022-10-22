df.count()
df.columns
df.types
df.schema
df.printSchema()

# SELCT
df.select("id", "name").show()

# FILTER
df.filter(df['id'] == 1).show()
df.filter(df.id == 1 ).show()
df.filter(col("id") == 1).show()
df.filter("id = 1").show()

# DROP
newdf = df.drop("id")  # es innutable nos e eilimna, sino creo una copia
newdf.show(2)

# Aggregations
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
df.sort("salary").show(5)
df.sort(desc("salary")).show(5)

# COLUMNAS DERIVADAS
df.withColumn("bonus", col("salary")* .1).show(5)

# JOINS
df.join(deptdf, df["dept"] == deptdf["id"]).show()
df.join(deptdf, df["dept"] == deptdf["id"]).show()