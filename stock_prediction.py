#importing required libraries

import eikon as ek
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import seaborn as sns

#Thomson Rueters api key for importing data
ek.set_app_key('7d5a45be53d24a3489564b9dad0597a71c9f283a')

#User input for company
Dict = {1: "BRKa", 2: "601628.SS", 3: "AIG", 4:"ALVG.DE", 5:"MET"} 
print(Dict)
inp = int(input("select a number for which company you want to predict the stock value: " ))
ccode=Dict.get(inp)
#ccode = input("Enter a valid RIC code for which company you want to predict the stock value: " )

#extracting data from Thomson Rueters using RIC values

df = ek.get_timeseries([ccode], start_date = "2003-01-01", end_date = "2019-01-01", interval="daily")
df =  df[['CLOSE']]
print(df.CLOSE.plot(figsize=(10,6), title='The correlation between variables in the data set' ))
      
#importing macro economic Indicators

#USA gdp
gdp= ek.get_timeseries('aUSXGDP/C', start_date='2003-01-01', end_date='2019-01-01', interval='yearly')
gdp=gdp.resample('D').ffill().add_suffix('_gdp')

#unemployemennt data
unemp= ek.get_timeseries('USUNR=ECI', start_date='2003-01-01', end_date='2019-01-01', interval='monthly')
unemp= unemp.resample('D').ffill().add_suffix('_debt')

#debt as percent of gdp
debt= ek.get_timeseries('USGV1YDP=SM', start_date='2003-01-01', end_date='2019-01-01', interval='daily')

#consumer credit
cred= ek.get_timeseries('aUSCREDITI/A', start_date='2003-01-01', end_date='2019-01-01', interval='monthly')
cred= cred.resample('D').ffill().add_suffix('_cred')

#Consumer prices index
cpi= ek.get_timeseries('aUSCP67F/C', start_date='2003-01-01', end_date='2019-01-01', interval='monthly')
cpi= cpi.resample('D').ffill().add_suffix('_cpi')

#10 year bond yields
yields= ek.get_timeseries('aUSEBM10Y', start_date='2003-01-01', end_date='2019-01-01', interval='monthly')
yields= yields.resample('D').ffill().add_suffix('_yileds')

#financical sector index
Findex= ek.get_timeseries('.TRXFLDUSPFIN', start_date='2003-01-01', end_date='2019-01-01', interval='daily')

#dollar index
dolind= ek.get_timeseries('.DXY', start_date='2003-01-01', end_date='2019-01-01', interval='daily')

#adding columns to the data frame 
df['gdp'], df['unemp'], df['debt'], df['cred'],df['cpi'], df['yields'], df['findex'], df['dollar_index'] = [gdp, unemp, debt, cred, cpi, yields, Findex.CLOSE, dolind.CLOSE ]     

#filling the missing values
df.interpolate(method ='slinear', limit_direction= 'both', inplace= True)   
print(df.head())


#correlation
corr = df.corr()
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=False, annot=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')  

#convertind df to an array
#creating independent dataset
x=df.drop(['CLOSE'],1)
x = x[:-10]

#creating dependent dataset
y=df['CLOSE']
y = y[:-10]

#creating train test data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

#Machine learning models
#support vector machines using regression
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001)
svr_rbf.fit(x_train, y_train)
svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence is ", svm_confidence)

#linear regression
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_confidence= lr.score(x_test, y_test)
print("lr confidence of the model is ", lr_confidence)

#forecasting the value
x_forecast=df.drop(['CLOSE'],1)[-10:]     
print(df.tail(10))

#forecasting using linear regression     
lr_prediction= lr.predict(x_forecast)
print("\n the prediction values using linear regression are \n", lr_prediction)

#forecasting using SVR      
svm_prediction=svr_rbf.predict(x_forecast)
print("\n the prediction values using SVM are \n", svm_prediction)
