# DF Test
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=10)
    rolstd = pd.rolling_std(timeseries, window=10)

    # Plot rolling statistics:
    plt.figure(figsize=(14,8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries.SALES_IN_ML, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

------------------------------

# ARIMA
# Ordering: ACF and PACF plots
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(TS_log, lags=30)
plot_pacf(TS_log, lags=30)

# Ordering: AIC
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic
# Smaller Better
arma_order_select_ic(TS_log, max_ar=4, max_ma=0, ic='aic')

# Modeling
model = ARIMA(TS_log, order=(1, 0, 0))  
results_AR = model.fit(disp= 1)  
plt.plot(TS_log)
plt.plot(results_AR.fittedvalues, color='red')
TS_fitted = pd.DataFrame(results_AR.fittedvalues,columns=['SALES_IN_ML'])

# Residual Series Check
residuals = TS_log - TS_fitted
test_stationarity(residuals)