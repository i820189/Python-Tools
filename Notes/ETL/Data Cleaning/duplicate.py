
###Valido Duplicados por llave
print( len( misiones_ejecucion_[misiones_ejecucion_.duplicated(subset=['mision','grupo_mision'])] ) )
misiones_ejecucion_ = misiones_ejecucion_.drop_duplicates(subset=['mision','grupo_mision'])
print( len( misiones_ejecucion_[misiones_ejecucion_.duplicated(subset=['mision','grupo_mision'])] ) )


# Ver duplicados:
df.duplicated(subset=['mision','grupo_mision'])

# Eliminar duplicados
df.drop_duplicates(subset='id', keep='first') --last


# Validar si hay duplicados por POC's y Grupos
tot = df_misiones__.groupby(['poc','Group']).size().reset_index(name='row1')
tot.groupby(['poc'])['Group'].size().loc[lambda x: x>1].sort_values().head()

# Valido nivel de agregación existente
df_misiones_tmp.groupby(['poc','sku']).size().loc[lambda x: x>1].reset_index(name='tot').sort_values('tot').head(10)


# re_misiones_no.columns
# re_misiones_cs = re_misiones_cs.drop_duplicates()
# re_misiones_no.groupby(['poc','sku']).size().loc[lambda x: x>1].reset_index(name='tot').sort_values('tot').head(10)
# re_misiones_no[ (re_misiones_no.poc == 12541111) & (re_misiones_no.sku == 4961) ].transpose()

# import missingno as msno
# msno.matrix(re_misiones_no)
# msno.bar(re_misiones_no)

# print( re_misiones_no.nunique() )

# import matplotlib.pyplot as plt
# plt.figure(figsize=(20,6))
# re_misiones_no.Innovation.value_counts().plot(kind = 'bar')

# df_misiones.head()
# df_misiones_tmp.groupby(['poc','sku']).size().loc[lambda x: x>1].reset_index(name='tot').sort_values('tot').head(10)
# df_misiones__[ (df_misiones__.poc == '12541111') & (df_misiones__.sku == '4961') ].transpose()