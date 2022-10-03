# Differencing
data['instantaneous'] = data.volume_out.diff()

# Cumulation
consum.loc[:,"group"] = consum["is_start_point"].cumsum()