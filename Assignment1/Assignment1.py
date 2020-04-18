import numpy as np
import pandas as pd 
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from datetime import datetime
from pandas import ExcelWriter
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import ARIMA




df = pd.read_csv('C:/Users/Md.Anishur/Desktop/Umea University/Umea Uni _ Assignment/DataSet/Milano_WeatherPhenomena/mi_meteo_2001.csv', names=['id','Day','value'])



print(df.head())

#print (df.iloc[1:])

dataset = df[df.columns[1:]]

#print(dataset)

dataset['Day'] = pd.to_datetime(dataset['Day'], infer_datetime_format=True)

indexedDataset = dataset.set_index(['Day'])

#writer = pd.ExcelWriter('mi_meteo.xlsx', engine='xlsxwriter')
#indexedDataset.to_excel(writer, sheet_name='Sheet1')
#writer.save()

print(indexedDataset.head(5))
print(indexedDataset.tail(5))

df = indexedDataset.resample('24H').mean()

print(df)

## plot graph ##

plt.xlabel('Date')
#plt.xticks(rotation=90)
plt.ylabel('Degree in Celsius')
plt.plot(df)
plt.show()

rolmean = df.rolling(window=2).mean()
rolstd = df.rolling(window=2).std()

print(rolmean,rolstd)

## plotting rolling statistics ##

orig = plt.plot(df, color='blue', label = 'Original')
mean = plt.plot(rolmean, color='red', label = 'Rolling Mean')
std = plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standanrd Deviation')
#plt.show(block=False)
plt.show()

## Dickey-Fuller Test ##

print('Results of Dickey-Fuller Test:')

dftest = adfuller(df['value'], autolag ='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():
    dfoutput['Critical Value(%s)'%key] = value

print(dfoutput)

## Estimating Trend ##

df_logScale = np.log(df)

plt.plot(df_logScale)
plt.show()


## Moving Average ##

movingAverage = df_logScale.rolling(window=2).mean()

movingSTD = df_logScale.rolling(window=2).std()

plt.plot(movingAverage)
plt.show()
plt.plot(movingSTD)
plt.show()

dflogScaleMinusMovingAve = df_logScale - movingAverage

print(dflogScaleMinusMovingAve.head(12))

## Removing NA values ##

dflogScaleMinusMovingAve.dropna(inplace=True)

print(dflogScaleMinusMovingAve.head(12))


## adfuller -- Test stationarity ##


def test_stationarity(timeseries):

    movingAverage = timeseries.rolling(window=2).mean()
    movingSTD = timeseries.rolling(window=2).std()

    ## plot rolling statistics

    orig = plt.plot(df, color='blue', label = 'Original')
    mean = plt.plot(rolmean, color='red', label = 'Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standanrd Deviation')
    #plt.show(block=False)
    plt.show()

    ## Perform Dickey-Fuller test ##

    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(df['value'], autolag ='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value(%s)'%key] = value
    print(dfoutput)

test_stationarity(dflogScaleMinusMovingAve)


## Exponential Decay Weighted Average ##


exponentialDecayWeightedAverage = df_logScale.ewm(halflife = 12, min_periods = 0, adjust = True).mean()

plt.plot(df - df_logScale)
plt.plot(exponentialDecayWeightedAverage, color='red')
plt.show()

dfLogScaleMinusMovingExponentialDecayAverage = df_logScale - exponentialDecayWeightedAverage

test_stationarity(dfLogScaleMinusMovingExponentialDecayAverage)

dflogDiffShifting = df_logScale - df_logScale.shift()

plt.plot(dflogDiffShifting)
plt.show()

dflogDiffShifting.dropna(inplace=True)

test_stationarity(dflogDiffShifting)


#from statsmodels.tsa.seasonal import seasonal_decompose


decomposition = seasonal_decompose(df_logScale)

trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid

plt.subplot(411)
plt.plot(df_logScale, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.legend(loc='best')
plt.tight_layout()

decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)

## ACF and PACF Plots ##

#from statsmodels.tsa.stattools import acf,pacf


lag_acf = acf(dflogDiffShifting, nlags=20)
lag_pacf = pacf(dflogDiffShifting, nlags=20, method='ols')


## Plot ACF ##

plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(dflogDiffShifting)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(dflogDiffShifting)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')


## Plot PACF ##

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(dflogDiffShifting)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(dflogDiffShifting)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()



## AR Model ##

model = ARIMA(df_logScale, order=(2,2,2))
results_AR = model.fit(disp=-1)
plt.plot(dflogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues - dflogDiffShifting["value"] )**2))
plt.show()
print('Plotting AR model')

## MA Model ##

model = ARIMA(df_logScale,order=(2,2,2))
results_MA = model.fit(disp=-1)
plt.plot(dflogDiffShifting)
plt.plot(results_MA.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues - dflogDiffShifting["value"] )**2))
plt.show()
print('Plotting MA model')

## ARIMA Model ##

model = ARIMA(df_logScale,order=(2,2,2))
results_ARIMA = model.fit(disp=False)
plt.plot(dflogDiffShifting)
plt.plot(results_ARIMA.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues - dflogDiffShifting["value"] )**2))
plt.show()
print('Plotting ARIMA model')


## Prediction ARIMA Diff ##

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)

print(predictions_ARIMA_diff.head())

## Convert to Comulative sum ##

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

print(predictions_ARIMA_diff_cumsum.head())


## Prediction ARIMA log ##

predictions_ARIMA_log = pd.Series(df_logScale['value'].ix[0], df_logScale.index)

predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)

print(predictions_ARIMA_log.head())


## Prediction ARIMA ##

predictions_ARIMA = np.exp(predictions_ARIMA_log)

plt.plot(df)
plt.plot(predictions_ARIMA)
plt.show()

## Print out the Dataset

print(df_logScale)

## Prediction ##

#results_ARIMA.plot_predict(1,7)
#plt.show()
x = results_ARIMA.forecast(steps=1)
print(x)













