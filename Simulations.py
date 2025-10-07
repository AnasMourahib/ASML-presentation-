# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 15:46:40 2025

@author: 20254817
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import math 
from sklearn.linear_model import LinearRegression




months =["Jan" , "Feb" , "Mar" , "Apr" , "May" , "Jun" , 
         "Jul" , "Aug" , "Sep" , "Oct" , "Nov" , "Dec"]

Temperatures = np.random.normal(loc = 22 , scale = 1 , size = 24 * 30 * 12)
Temperatures_matrix = Temperatures.reshape(720  , 12)

df = pd.DataFrame( Temperatures_matrix , columns = months )

##Plot the whole time series

plt.plot(Temperatures)
plt.title("Hourly Temperature Time Series (Jan–Dec) of an ASML Component")
plt.xlabel("Hour index")
plt.ylabel("Temperature")
plt.show()



#### Divide the time series into months and then plot 

axes = df.plot(subplots=True, layout=(4,3), figsize=(12,10), sharex=True, color = 'black')
plt.suptitle("Hourly temperature time series of an ASML component divided by months with maximum temperature each month in red", y=1.02)
axes = axes.flatten()

for i,m in enumerate(df.columns):
    idx_max = df[m].idxmax()
    val_max = df[m].max()
    axes[i].plot(idx_max , val_max , 'ro')

plt.show()



##Define the maximum for each month

max_per_month = Temperatures_matrix.max(axis= 0)



###########
import pyreadr

# Load the RDS file
result = pyreadr.read_r("C:/Users/20254817/Desktop/Githib/Simulations/st1.rds")

df = list(result.values())[0]  # extract the DataFrame
print(df.head())

Temp = []
for t in df.values:
    Temp.append(t[0])

years = np.arange(1850 , 2020 , step = 20)

plt.plot(Temp)
plt.title("Daily temperature time series (1850-2020) ")
plt.xlabel("Time index")
plt.xticks(np.linspace(0, len(Temp)-1, len(years)) , labels = years)
plt.ylabel("Temperature")
plt.show()


Temp = np.array(Temp)

blocks = np.arange(0 , 171)
max_per_year_ind = []
for block in blocks:
    print(block)
    index = np.arange(block * 92 , ((block+1) * 92)  )
    val_per_year = Temp[ index  ]
    idx = np.where(val_per_year == val_per_year.max())[0][0]  
    max_per_year_ind.append(block * 92 + idx)


plt.plot(Temp, color="blue")
plt.scatter(max_per_year_ind , Temp[max_per_year_ind], color='red')
plt.title("In red, the maximum temperature of each year")
plt.xlabel("Years")
plt.xticks(np.linspace(0, len(Temp)-1, len(years)) , labels = years)
plt.ylabel("Temperature")
plt.show()




### Histogram of Annual maxima temperature 

plt.hist(Temp[max_per_year_ind])
plt.title("Histogram of annual maxima")
plt.xlabel("Annual maxima")
plt.show() 

#### Part 1 : Fitting a Gumbel distribution 
   ####Part 1.1 QQ plot for a standard Gumbel distribution, meaning location = 0 and scale = 1
n_years = len(Temp)/92

def Gumbel_quantile_fun(p) :
    x = -math.log(-math.log(p))
    return(x)

emp_quantile  = np.sort(Temp[max_per_year_ind])
Gumbel_quantile_standard =[]
for  i in np.arange(1, n_years+1):
    val = Gumbel_quantile_fun(i/(n_years+1))
    Gumbel_quantile_standard.append(val)

plt.plot(Gumbel_quantile_standard , emp_quantile)
plt.title("Q-Q plot")
plt.xlabel("Standard Gumbel quantiles")
plt.ylabel("Annual maxima")
plt.plot([min(Gumbel_quantile_standard), max(Gumbel_quantile_standard)], [min(Gumbel_quantile_standard), max(Gumbel_quantile_standard)], color='red')  # x = y line


   ###Part 1.2 regress Gumbel quantiles on Empirical quantiles to estimate the shape and location parameters

Gumbel_quantile_standard = np.array(Gumbel_quantile_standard)


model = LinearRegression()

y = emp_quantile
x =  np.array(Gumbel_quantile_standard).reshape((-1 , 1))
lm = LinearRegression().fit(x, y)

slope = lm.coef_ ### This is the scale parameter
intercept = lm.intercept_ ### This is the location parameter 


Gumbel_quantile = (Gumbel_quantile_standard * slope) + intercept

plt.plot(Gumbel_quantile , emp_quantile)
plt.title("Q-Q plot")
plt.xlabel("Gumbel quantiles")
plt.ylabel("Annual maxima")
plt.plot([min(Gumbel_quantile), max(Gumbel_quantile)], [min(Gumbel_quantile), max(Gumbel_quantile)], color='red')  # x = y line



##### Part 2 : Fitting a generalized extreme value distribution  

from scipy.stats import genextreme as gev

###

def main(rvs):
    shape , loc , scale = gev.fit(rvs)
    return - shape , loc , scale

shape, loc, scale = main(Temp[max_per_year_ind])



###### QQ plot 

def quantile_gev_fun(x, loc, scale, shape):
    quantile = loc + (scale / shape) * ( ( (-math.log(x)) ** (-shape) ) - 1)
    return quantile



Gev_quantile =[]
for  i in np.arange(1, n_years+1):
    val = quantile_gev_fun( i/(n_years+1) ,  loc  = loc , scale = scale , shape = shape)
    Gev_quantile.append(val)
    

plt.plot(Gev_quantile , emp_quantile)
plt.title("Q-Q plot")
plt.xlabel("Generalized extreme value distribution quantiles")
plt.ylabel("Annual maxima")
plt.plot([min(Gev_quantile), max(Gev_quantile)], [min(Gev_quantile), max(Gev_quantile)], color='red')  # x = y line




#### Part 3 : Fitting a Generalized Pareto distribution

threshold = np.quantile(Temp, q= 0.95)

exceedances = []

for value in Temp:
    if value > threshold:
        exceedances.append(value)
        
        
exceedances = np.array(exceedances)     


plt.plot(Temp, color='blue')
plt.scatter(np.where(Temp > threshold), exceedances, color='red')
plt.title("Exceedances of the 0.95 quantile in red")
plt.xlabel("Years")
plt.ylabel("Temperature")
plt.axhline(y=threshold, color='black', linestyle='--')
plt.show()
