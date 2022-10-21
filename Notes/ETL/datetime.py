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

#################################################################################################################

import calendar
from datetime import datetime

df_misiones_['brandto'] = df_misiones_.brand.apply(lambda x: formatbrand(str(x)))
df_misiones_['year'] = df_misiones_.periodo.apply(lambda x: int(str(x)[0:4]))
df_misiones_['StartDate'] = df_misiones_.periodo.apply(lambda x: datetime.strptime(str(x)[4:6] + '-01-' +str(x)[0:4], '%m-%d-%Y').date() )
df_misiones_['month'] = df_misiones_.StartDate.apply(lambda x: x.strftime("%B"))

import datetime
df_misiones_['EndDate'] = df_misiones_.StartDate.apply(lambda x: datetime.date(x.year, x.month, calendar.monthrange(x.year, x.month)[1]) )