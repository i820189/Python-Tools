###########################################################
# GROUP BY  (COUNT)
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



###########################################################
# GROUP BY  (MAX)
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