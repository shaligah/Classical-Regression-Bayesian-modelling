#data manipulation and processing
import numpy as np 
import pandas as pd
import missingno as ms

#data visualization
import matplotlib.pyplot as plt
import seaborn as sns

#variable selection
from statsmodels.stats.outliers_influence import variance_inflation_factor as vi
from sklearn.metrics import r2_score
from scipy import stats

#model building & tuning
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
import pymc3 as pc
import statsmodels.api as sma

#model evaluation
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tools.eval_measures import rmse

#load the data
data = pd.read_csv('weekly_media_sample.csv')

#removing irrelevant columns
data.drop('X', inplace= True, axis=1)

#checking for missing values
ms.matrix(data)

#checking data types
data.dtypes.value_counts()

#splitting the data into training and testing
train,test = data.iloc[0:200,:], data.iloc[200:,:]

#splitting the data in predictor and response for training
X,y = train.iloc[:,2:],train.loc[:,'revenue']

#checking the relationship between predictor and response variables using scatterplots
x = X.to_numpy()
plt.rcParams["figure.figsize"] = (11,9) 
plt.rcParams["font.size"] = 6
for i in range(x.shape[1]):
    plt.subplot(3, 3, i+1)
    plt.scatter(x[:,i],y)
    plt.title(X.columns[i])

##checking for multicollinearity
#using correlation plots
heatmap = sns.heatmap(train.corr(), annot=True)
heatmap.set_title('Correlation heatmap')

#using the variance inflation factor
vif = pd.DataFrame()
vif['features'] = X.columns
vif['VIF'] = [vi(X.values,i) for i in range(X.shape[1])]

### OLS LINEAR REGRESSION

#creating a design matrix
Xv1 = sma.add_constant(X)

#building an OLS Regression model and fitting the data
a = sma.OLS(y,Xv1)
b = a.fit()
print(b.summary())

#predicting the results using the test set
test1 = test.copy()
test1.drop(columns=['DATE','revenue'], inplace=True)
b.predict(sma.add_constant(test1))

#evaluating the model
rmse(test['revenue'],b.predict(sma.add_constant(test1)))

print(b.params)

#creating a residual plot to analyse the prediction results
residuals  = test['revenue']-predicted
plt.figure(figsize=(6,4))
plt.scatter(predicted, residuals)
plt.axhline(y=0, color='r', linestyle='-')

#developing a prediction interval for the predicted variables
n = len(X)
p = 5
se_res = np.sqrt(np.sum(residuals**2)/(n-p-1))
alpha = 0.05
tcrit = np.abs(stats.t.ppf(1- alpha/2,n-p-1))
upperbound = predicted+(tcrit*se_res)
lowerbound = predicted-(tcrit*se_res)

results = test.copy()
results.drop('DATE',axis=1, inplace=True)
results['p_revenue'] = predicted

results['pi_upper'] = upperbound
results['pi_lower'] = lowerbound

#visualizing the predictions within the prediction intervals
plt.scatter(results['revenue'], results['p_revenue'])
plt.errorbar(results['revenue'], results['p_revenue'], yerr=tcrit*se_res, fmt='o', color='r')
plt.plot([results['revenue'].min(), results['revenue'].max()], [results['revenue'].min(), results['revenue'].max()], 'k--')
plt.xlabel('True values')

### ARIMA

#splitting the data in training and test data
X2,test2 = data.iloc[:200,:2], data.iloc[200:,:2]

#conversion of the date column to datetime and setting it as the index variable
X2['DATE'],test2['DATE'] =pd.to_datetime(X2['DATE']),pd.to_datetime(test2['DATE'])

X2 =X2.set_index('DATE')
test2 = test2.set_index('DATE')

#visualizing the data in a timeseries plot
X2.plot()

#checking for stationarity of the data
result = adfuller(X2)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

#tuning the model appropriately for the data(slecting values for p,d,q)
model = pm.auto_arima(X2, start_q=0, start_p=0,max_q=15, max_p=15, error_action='ignore', suppress_warnings=True, stepwise=True)
model.summary()

#building the ARIMA time series model
model= ARIMA(X2,order=(3,0,0))
results = model.fit()
print(results.summary())

#predicting the results of the ARIMA model on the test data
predictions = results.predict(start=len(X2), end=len(X2)+len(test2)-1, dynamic=True)
predictions
test2['forecasts'] = predictions

#evaluating the model
rmse(test2['forecasts'],test2['revenue'])

#visualization of the forecasts against the original
test2[['revenue','forecasts']].plot(figsize =(8,6))

###BAYESIAN MODELLING

#building the model
with pc.Model() as model:
    
    #define the non informative priors
    sigma = pc.HalfNormal('sigma',sd=1)
    beta0 = pc.Normal('beta0', mu=0, sd=10)
    beta = pc.Normal('beta', mu = 0, sd =10, shape=5)
    
    #define the likelihood 
    y_obs = pc.Normal('y_obs', mu=beta0 + pc.math.dot(X,beta), sigma=sigma, observed=y)
    step= pc.Metropolis()
    
    #defining the MCMC sampler
    trace = pc.sample(1500, chains=4, tune=500, step=step)
    ppc = pc.sample_posterior_predictive(trace)

#using trace plots to check for convergence
pc.plot_trace(trace)
plt.show()

#summary of parameters 
pc.summary(trace)


#building the model with expert opinion on the prior distribution of media3_S
X3 = X.copy()
X3.drop('media3_S', axis=1,inplace=True)
m3 = X['media3_S']

with pc.Model() as model1:
    
    #define the priors with expert opinion on media3_S
    sigma = pc.HalfNormal('sigma',sd=1)
    beta0 = pc.Normal('beta0', mu=0, sd=10)
    beta = pc.Normal('beta', mu = 0, sd =10, shape=4)
    media3_S = pc.Normal('media3_S', mu=0, sd = 0.2)

    #define the likelihood 
    y_obs = pc.Normal('y_obs', mu=beta0 + media3_S*m3 + pc.math.dot(X3,beta), sigma=sigma, observed=y)
    step= pc.Metropolis()
    
    #defining the MCMC sampler
    trace1 = pc.sample(1500, chains=4, tune=500, step=step)
    ppc1 = pc.sample_posterior_predictive(trace1)

#summary of new parameters
pc.summary(trace1)

with pc.Model() as model2:

#building the model with several prior information available   
    #define the priors with expert opinions on various predictor varaibels
    sigma = pc.HalfNormal('sigma',sd=1)
    media3_S = pc.Normal('media3_S', mu=0, sd = 0.2)
    newsletter = pc.Lognormal('newsletter', mu = 0, sd=1) # strict positive relationship with revenue 
    media1_S = pc.Lognormal('media1_S', mu=np.log(2), sd=0.2) # unit increase results in 2x change in revenue
    media2_S = pc.Lognormal('media2_S', mu = np.log(8), sd = 0.2)# impact is 4x that of media1_S
    c_sales = pc.Uniform('c_sales', lower = 0, upper = 0.3)# change in revenue is between 0 and 0.3
    beta0 = pc.Uniform('beta0', lower=-10, upper=10)
    
    #define the likelihood 
    y_obs = pc.Normal('y_obs', mu=beta0 + (media1_S*m1) +(media2_S*m2) +(media3_S*m3)+(newsletter*news)+(c_sales*co_sales) , sigma=sigma, observed=y)
    step= pc.Metropolis()
    
    #defining the MCMC sampler
    trace2 = pc.sample(1500, chains=4, tune=500, step=step)
    ppc2 = pc.sample_posterior_predictive(trace2)

#summary of new parameters
pc.summary(trace2)

#checking for convergence
pc.plot_trace(trace2)
plt.show()
