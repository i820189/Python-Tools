###########################################################
# GROUP BY  (COUNT)

# select
#     Category,count(*) as count 
# from hadoopexam 
# where HadoopExamFee<3200  
# group by Category having count>10")
###########################################################

# SQL
select
    Category,count(*) as count 
from hadoopexam 
where HadoopExamFee<3200  
group by Category having count>10")

# Pyspark - SQL Context
sqlContext.sql("select Category,count(*) as
count from hadoopexam where HadoopExamFee<3200  
group by Category having count>10")

# Pyspark - Spark
from pyspark.sql.functions import *
df.filter(df.HadoopExamFee<3200) \
  .groupBy('Category') \
  .agg(count('Category').alias('count')) \
  .filter(col('count')>10)

# Python - 1
counties = df.groupby(['state','county'])['field1'].max()>20
counties = counties.loc[counties.values==True]

# Python  - 2 (un solo campo)
df_trx.periodo.value_counts().reset_index(name='Total').sort_values('index')

# Python - 3
g = df.groupby('A')  #  GROUP BY A
g.filter(lambda x: len(x) > 1)  #  HAVING COUNT(*) > 1

df.groupby('A').filter(lambda x: len(x) > 1)

df_poc_sku_.groupby(['poc','sku']).filter(lambda x: len(x)>1)

# Python - 4
df.groupby('team').filter(lambda x: x['points'].sum() == 48) # HAVING SUM
df.groupby('team').filter(lambda x: x['points'].mean() > 20) # HAVING MEAN

# Python - por confirmar
df_poc_sku_.groupby(['poc','sku'])['sku'].count().reset_index(name='row1').query("row1 > 1")
df_client.groupby(['Cliente']).size().reset_index(name='row1').query("row1 > 1") # Este me funcó
df_client.loc[ (df_client.Cliente.isnull()) | (df_client.Cliente == '') | (df_client.Cliente<=0) ].head() # porsilas valido si tiene el dato medioraro
df_client = df_client.drop( df_client[ df_client['Cliente'].map(lambda x: str(x).startswith('I')) ].index)


###########################################################
# GROUP BY  (MAX)

# select
#     poc,max(fecha) as max_fecha 
# from df_encuesta 
# group by poc
###########################################################

# SQL
select
    poc,max(fecha) as max_fecha 
from df_encuesta 
group by poc

# python - 1
df_encuesta_fecha_max=pd.pivot_table(df_encuesta, index='poc',aggfunc={'fecha':'max'}).reset_index(name='max_fecha')

# Python - 2
df.groupby('team').filter(lambda x: x['points'].sum() == 48) # HAVING SUM



###########################################################
# IS NULL

# select * from df
# where state is null;
###########################################################

# SQL
select * from df where state is null;

# Pyspark - Using Column.isNull()
from pyspark.sql.functions import col
df.filter(df.state.isNull()).show()
df.filter("state is NULL").show()
df.filter(col("state").isNull()).show()

# Python
df[ (df.state.isnull()) | (df.state.isna() )].head()


###########################################################
# CASE SIMPLE (like a IF ELSE)

# SELECT
#     case
#         when segmentacion_abcd="G" then "D" 
#     else segmentacion_abcd 
#     end segmentacion_abcd
# FROM demo_df
###########################################################

# SQL
SELECT
    case
        when segmentacion_abcd="G" then "D" 
    else segmentacion_abcd 
    end segmentacion_abcd
FROM demo_df

# Python
demo_df["segmentacion_abcd"] = np.where( demo_df["segmentacion_abcd"]=="G", "D", demo_df["segmentacion_abcd"])

# Pyspark
from pyspark.sql.functions import when, col
demo_df = demo_df.withColumn('segmentacion_abcd', when(col('segmentacion_abcd') == 'G', 'D').otherwise(col('segmentacion_abcd')))
demo_df.show()


###########################################################
# UPDATE

# update demo_df SET new_class_ts = 'Bodega-reja'
# where new_class_ts = 'Bodega' and tiene_reja = 'SI'
###########################################################

# SQL
update demo_df SET new_class_ts = 'Bodega-reja'
where new_class_ts = 'Bodega' and tiene_reja = 'SI'

# Python - tomo todos los registros que cumplen la condición, pero solo del campo "new_class_ts" y luego igualo el valor
demo_df.loc[
    (demo_df["new_class_ts"]== "Bodega")
    & (demo_df.tiene_reja == "SI"),
    "new_class_ts",
] = "Bodega-reja"

# Pyspark
from pyspark.sql.functions import when
demo_df = demo_df.withColumn("new_class_ts", when((demo_df.new_class_ts == "Bodega") & (demo_df.tiene_reja == "SI"),"Bodega-reja") \
      .otherwise(df.new_class_ts))
demo_df.show()

# pyspark 2
field = "sector"
conditions_all = (when(col(field) == "Finance", "Financial Services")
    .when(col(field) == "n/a", "No sector available")
    .otherwise(col(field))
)
# apply transformation rule to the column
sector_df = sector_df.withColumn(field, conditions_all)
sector_df.show(MAX_ROWS, False)



###########################################################
# LENGTH + Groupby

# select length(poc) 
# from misiones_ 
# group by 1

# select * from misiones_ 
# where length(poc)>=10
###########################################################


# Python LENGTH + Group by (ver primero que tamaños existen)
misiones_.groupby( 
  misiones_.poc.apply(lambda x: len(str(x)))
  , dropna=False
).size().reset_index()


# Python LENGTH
misiones_.loc[ 
  misiones_['poc'].apply(lambda x: len(str(x))>=10)
].head()


###########################################################
# IS IN

# select * from df
# where state in ('','')
###########################################################

# SQL
select * df_ df
where LIFECYCLE_CD in ('G','S')

# PYSPARK
df_ = df.filter( col('LIFECYCLE_CD').isin(['G','S']) )
df_

# Python
df_ = df_[ df_['LIFECYCLE_CD'].isin(['G','S'])]
df_.head()


###########################################################
# RENAME COLUMN

# select 
#   dob AS DateOfBirth
# from df
###########################################################

# Pyspark
df.withColumnRenamed("dob","DateOfBirth")
    .printSchema()
    
    
###########################################################
# CAST

# select 
#   dob AS DateOfBirth
# from df
###########################################################
    

    
df2 = df.withColumn("age",col("age").cast(StringType())) \
    .withColumn("isGraduated",col("isGraduated").cast(BooleanType())) \
    .withColumn("jobStartDate",col("jobStartDate").cast(DateType()))
    
    
    
    
###########################################################
# PERIODO  / MES

# select 
#   year(date)*100 + month(date) as periodo
# from df
###########################################################
    
from pyspark.sql.functions import date_format

datetimesDF. \
    withColumn("periodo", date_format("date", "yyyyMM")). \
    withColumn("periodo", date_format("time", "yyyyMM")). \
    show(truncate=False)

datetimesDF. \
    withColumn("date_ym", date_format("date", "yyyyMM").cast('int')). \
    withColumn("time_ym", date_format("time", "yyyyMM").cast('int')). \
    printSchema()
    
# yyyy
# MM
# dd
# DD
# HH
# hh
# mm
# ss
# SSS
+----------+-----------------------+-------+-------+
|date      |time                   |date_ym|time_ym|
+----------+-----------------------+-------+-------+
|2014-02-28|2014-02-28 10:00:00.123|201402 |201402 |
|2016-02-29|2016-02-29 08:08:08.999|201602 |201602 |
|2017-10-31|2017-12-31 11:59:59.123|201710 |201712 |
|2019-11-30|2019-08-31 00:00:00.000|201911 |201908 |
+----------+-----------------------+-------+-------+





    
###########################################################
# SPLIT_PART

# select 
#   split_part(LOCATION_STARTED,",",0) latitud_s
# from df_dataplor_op
###########################################################


# Pyspark
df_dataplor_op = df_dataplor_op \
  .withColumn('latitud_s', split(df_dataplor_op['LOCATION_STARTED'], ',').getItem(0)) \
  .withColumn('longitud_s', split(df_dataplor_op['LOCATION_STARTED'], ',').getItem(1)) \
  .withColumn('latitud_d', split(df_dataplor_op['LOCATION_COMPLETE'], ',').getItem(0)) \
  .withColumn('longitud_d', split(df_dataplor_op['LOCATION_COMPLETE'], ',').getItem(1))
display( df_dataplor_op )


    
###########################################################
# LIKE 'I%' - starwith startWith endwith

# select 
#   *
# from df_clientdata where Cliente like 'I%'
###########################################################

# Pyspark
df_clientdata.filter( col("Cliente").startswith("I") == True).display()

df_clientdata = df_clientdata.filter( col("Cliente").startswith("I") == False)

    
###########################################################
# COALESCE 


###########################################################

# SQL
select coalesce(a,b) from df

# PYTHON
df['c'] = np.where(df["a"].isnull(), df["b"], df["a"] )