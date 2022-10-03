# string to datetime
df.dt = pd.to_datetime(df.dt, format='%Y%m%d')

# get datetime indexes
t = pd.DatetimeIndex(df.dt)
hr = t.hour
df['HourOfDay'] = hr
month = t.month
df['Month'] = month
year = t.year
df['Year'] = year

# resample time series
df = df.set_index('datetime')
weekly_summary['speed'] = df.speed.resample('W').mean()
weekly_summary['distance'] = df.distance.resample('W').sum()
weekly_summary['cumulative_distance'] = df.cumulative_distance.resample('W').last()

# generate given format string from datetime
df['DOB1'] = df['DOB'].dt.strftime('%m/%d/%Y')