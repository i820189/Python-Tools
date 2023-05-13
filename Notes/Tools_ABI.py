


#############################################################################################################################
# CLIENTES
#############################################################################################################################

# LECTURA DE ARCHIVO
df_client=pd.read_csv('/dbfs/mnt/adls_maz131/analytics_zone/MAZ/PE/POP/Data/ClientData_full.csv')

# BORRO CLIENTES DUPLICADOS
df_client[ df_client.duplicated() ]
df_client.drop_duplicates(subset = 'Cliente' & subset = 'Cliente' )

# BORRO COLUMNAS QUE NO ENCESITO
df_client = df_client.drop(['zone_id','index_right'],axis=1)

# CLIENTES ÚNICOS / CLIENTES CON CÓDIGO CUPLICADO
df_client.groupby(['Cliente']).size().reset_index(name='row1').query("row1 > 1")

# CLIENTES CON CÓDIGO CORRECTO
df_client.loc[ (df_client.Cliente.isnull()) | (df_client.Cliente == '') | (df_client.Cliente<=0) ].head()

# FILTRO CLIENTES QUE NO EMPIECEN CON I
df_client = df_client.drop( df_client[ df_client['Cliente'].map(lambda x: str(x).startswith('I')) ].index)

# FILTRO POR CLIENTE
df_client.loc[ df_client.Cliente.isin([10258145,101222601,12686941,2200670001]) ].head()

# CONVIERTO COORDENADAS DE CLIENTE
pip install geopandas
import geopandas as gpd
gdf_client = gpd.GeoDataFrame(
    df_client, geometry=gpd.points_from_xy(df_client.Longitud, df_client.Latitud))
gdf_client.set_crs(epsg=4326, inplace=True)
gdf_client.to_crs(epsg=4326)




spark.sql("""
SELECT * 
FROM abi_lh_pe.pop_audience
where periodo = {} and (poc is null or poc='' or poc<=0 or (Unidad_Negocio = 'DSD' and length(poc)>8) or (Unidad_Negocio = 'DAS' and length(poc)>10) ) 
""".format(anio+mes)).display()


#############################################################################################################################
# GEOPANDAS
#############################################################################################################################


# VALIDO QUE TANTOS NULOS HAY DESPUES DE LA BUSQUEDA
join_result.zone_id.isnull().mean()
join_result.ubigeo.isnull().mean()

# VALIDAR CON UN GRAFICO COMO SE DISTRIBUYE LA DATA EN DATABRICKS






#SE CARGAN LOS CONSOLIDADOS MENSUALES MENOS FEBRERO
for fecha in ['2022-07-31','2022-08-31','2022-09-30','2022-10-31','2022-11-30','2022-12-31','2023-01-31']:
    df=spark.sql(f"""
    select date_format(fecha_venta,'yyyyMM')as month_id,cliente_id,
	   SUM(case when estratificacion IN ('Cervezas','Licores','Ready To Drink') then hl else 0 end) as HLBEER,
	   SUM(case when estratificacion IN ('Agua','Maltas','Gaseosas') then hl else 0 end) as HLNABS,
	   SUM(case when estratificacion not in ('Cervezas','Licores','Ready To Drink','Agua','Maltas','Gaseosas') then hl else 0 end) as HLMKP
    from  dm.h_venta where estado_venta =1 and tipo_material ='FERT' AND fecha_venta BETWEEN date_trunc('month','{fecha}') AND '{fecha}'
    GROUP BY 1,2 ORDER BY 1 ASC
    """)
    df.write.mode('overwrite').parquet(f'abfss://dev@commandcentercontenedor.dfs.core.windows.net/upload_files/tmp/{fecha}')
    
    
    
    
df_cliente=spark.sql("SELECT a.cliente_id,a.direccion,a.gerencia,a.canal,a.subcanal FROM dm.cliente as a where estado =1 ")
i=0
for x in ['202207','202208','202209','202210','202211','202212','202301','202302']:
    print(x)
    if i==0:
        aux=df_cliente
    else :
        aux=spark.read.parquet('abfss://dev@commandcentercontenedor.dfs.core.windows.net/upload_files/tmp/final')
    aux.createOrReplaceTempView('aux')    
    
    final=spark.sql(f"""
              SELECT a.*,b.HLBEER as HLBEER_{x},b.HLNABS as HLNABS_{x},b.HLMKP as HLMKP_{x}
              FROM aux a
              LEFT JOIN df_{x} b
              ON  b.cliente_id=a.cliente_id
              """)
    final.write.mode('overwrite').parquet(f'abfss://dev@commandcentercontenedor.dfs.core.windows.net/upload_files/tmp/final')
    i=i+1
    
    
df=spark.read.parquet(f'abfss://dev@commandcentercontenedor.dfs.core.windows.net/upload_files/tmp/final')
df.write.mode('overwrite').option('header',True).csv(f'abfss://dev@commandcentercontenedor.dfs.core.windows.net/upload_files/tmp/final_csv')
df.count()