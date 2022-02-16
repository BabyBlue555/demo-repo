#!/usr/bin/env python
# coding: utf-8

# # Regression Problem - Predicting the Apparent Temperature

# # מחר להמשיך עם מודלים והשוואה, להתמקד יותר בפרוייקט המרכזי עם דולב

# *data leakage is going to be prevented by using a large amount of data (weather reports) in the train and test sets

# our dummy model - ?

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# read in all three datasets (you'll pick one to use later)
weather =pd.read_csv("weatherHistory[1].csv")


# In[3]:


weather


# In[4]:


weather.describe()


# ### What are the features?
# *Formatted Date
# 
# *Summary
# 
# *Precip Type
# 
# *Temperature (C)
# 
# *Humidity
# 
# *Wind Speed (km/h)	
# 
# *Wind Bearing (degrees)	
# 
# *Visibility (km)	
# 
# *Cloud Cover	(*misprint*)
# 
# *Pressure (millibars)
# 
# *Daily Summary
# 
# 
# What is the response?
# 
# Apparent Temperature : sales of a single product in a given market (in thousands of items)
# 
# What else do we know?
# 
# 1. Because the response variable is continuous, this is a regression problem.
# 2.There are 96453 observations (represented by the rows), and each observation is a weather report from a different date.

# # preprocessing

# In[5]:


weather.dropna()


# #### Precip Type, Temperature, wind speed , Wind Bearing, pressure are features that linked very strongly with the Apparent Temperature , they are crucial for the prediction of the Apparent Temperature. on the other hand, 
# 
# So The date column is really significant column cause the dataset are contined hourly/daily weather coindition predicting. Loud Cover is zero with max , min 25th 50th percentage and other summries,Daily Summary ıncluded a lot of unique variables.Precip Type and Summary could be useful in regression such as predicting Apperent Temparature.
# 
# *The term "wind direction" is defined as the compass heading FROM which the wind is blowing

# In[6]:


del weather['Daily Summary']
del weather['Loud Cover']


# In[7]:


weather


# In[8]:


import datetime


# In[9]:


weather['Formatted Date'] = pd.to_datetime(weather['Formatted Date'],utc=True)


# In[10]:


weather['year'] = weather['Formatted Date'].dt.year
weather['month'] = weather['Formatted Date'].dt.month
weather['day'] = weather['Formatted Date'].dt.day
weather['weekday'] = weather['Formatted Date'].dt.weekday


# In[11]:


del weather['Formatted Date']


# In[12]:


weather


# #### encoding catagorial data

# we need to encode Summary and Precip Type to numeric columns

# In[13]:


weather['Precip Type'].unique()


# If a categorical column has just two categories (it's called a binary category), then we can replace their values with 0 and 1. and because the 'Precip Type' column has only 2 catagoreis:

# In[14]:


precip_types = {'rain':0,'snow':1}
weather['precip_type']= weather['Precip Type'].map(precip_types)


# In[15]:


weather['Summary'].unique()


# In contrast to the 'precip_type' column , the Summary column has a lot of catagories, so we will use labelEncoder in order to transform the non-numerical labels to numerical labels  
# 

# In[16]:


from sklearn import preprocessing
lbl_encoder=preprocessing.LabelEncoder()
weather['summary'] = lbl_encoder.fit_transform(weather['Summary'])
weather['summary'].unique()


# now, after we encoded these values, we need to handle another problem:
# the machine learning model may assume that there is some correlation between these variables, which will produce the wrong output. So to remove this issue, we will use dummy encoding.

# For Dummy Encoding, we will use OneHotEncoder class of preprocessing library.

# In[17]:


# #for Country Variable  
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder  
# label_encoder_x= LabelEncoder()  
# weather['Precip Type']= label_encoder_x.fit_transform(weather['Precip Type'])  
# #Encoding for dummy variables  
# onehot_encoder= OneHotEncoder(categories=weather['Precip Type'])    


# pre= onehot_encoder.fit_transform(pre).toarray()  


# In[18]:


# dummy encoding of categorical features
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)


# In[19]:


ohe.fit_transform(weather[['summary']])


# In[20]:


ohe.categories_


# In[21]:


ohe.fit_transform(weather[['precip_type']])


# In[22]:


(weather[['precip_type']]==1.).count()


# In[23]:


weather


# ## doing onehotencoding at the same time on both of the categorial columns
# 

# In[24]:


X= weather.loc[:, ['summary','precip_type'] ]


# In[25]:


X


# In[26]:


# use when different features need different preprocessing
from sklearn.compose import make_column_transformer


# In[27]:


column_trans = make_column_transformer(
    (OneHotEncoder(), ['summary', 'precip_type']),
    remainder='passthrough')


# In[28]:


#hell= weather.copy()


# In[29]:


#hell=hell.loc[:,:]


# In[30]:


#hell


# In[31]:


X=column_trans.fit_transform(X)


# In[32]:


#hell=column_trans.fit_transform(hell)


# In[33]:


# pd.DataFrame(hell)


# In[ ]:





# In[34]:


X


# In[35]:


weather


# In[36]:


#hell


# now, we will remove Summary and Precip Type columns, cause alredy was made numeric(summary,precip_type)

# In[37]:


del weather['Summary']

del weather['Precip Type']


# In[38]:


weather


# In[39]:


#weather[weather['precip_type']==1.]


# In[40]:


#! pip install plotly==5.6.0


# ### choosing a regression metric

# ##### Mean Squared Error (MSE)

# In[41]:


from matplotlib import pyplot
from sklearn.metrics import mean_squared_error


# In[42]:


# example of calculate the mean squared error
# real value
expected = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# predicted value
predicted = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
# calculate errors
errors = mean_squared_error(expected, predicted)
# report error
print(errors)


# ##### Root Mean Squared Error

# In[ ]:





# #### Mean abs error

# In[43]:


# example of calculate the mean absolute error
from sklearn.metrics import mean_absolute_error
# real value
expected = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# predicted value
predicted = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
# calculate errors
errors = mean_absolute_error(expected, predicted)
# report error
print(errors)


# as we can see from the heatmap above, the correlation between the Apparent Temperature to the pressure is very low

# ### Train Test Split

# In[44]:


X = weather.drop(['Apparent Temperature (C)'],axis=1)
y = weather['Apparent Temperature (C)']


# In[45]:


from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)


# In[46]:


X_test.shape


# In[47]:


X_train


# ## Scaling
# 

# In[48]:


from sklearn.preprocessing import MinMaxScaler


# In[49]:


scaling= MinMaxScaler()


# In[50]:


scaling.fit_transform(X_train)


# In[51]:


arr_y_train=np.array(y_train)


# In[52]:


scaling.fit_transform(arr_y_train.reshape(-1,1))


# In[53]:


arr_x_test=np.array(X_test)


# In[54]:


scaling.transform(arr_x_test.reshape(-1,1))


# In[55]:


arr_y_test=np.array(y_test)


# In[56]:


scaling.transform(arr_y_test.reshape(-1,1))


# In[57]:


# arr.reshape((-1,1))


# In[58]:


# arr2=arr.reshape((-1,1))


# In[59]:


# arr2


# In[60]:


# scaling.transform([arr])


# In[61]:


y_train


# In[62]:


#X_train.astype


# In[63]:


train=X_train


# In[64]:


train['Apparent Temperature (C)']=y_train


# In[65]:


#train.loc[:,13]= y_train


# ## visualization

# In[66]:


import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')

fig , axs = plt.subplots(ncols=7,nrows=2, figsize=(20,10))
index=0
axs= axs.flatten()
for k ,v in X_train.items():
    sns.boxplot(y=k, data=X_train , ax=axs[index])
    index +=1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[ ]:





# In[67]:


fig , axs = plt.subplots(ncols=7,nrows=2, figsize=(20,10))
index=0
axs= axs.flatten()
for k ,v in weather.items():
    sns.boxplot(y=k, data=weather , ax=axs[index])
    index +=1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# correlations

# In[68]:


plt.figure(figsize=(20,10))
sns.heatmap(train.corr(),annot=True);


# In[69]:


# weather.drop(['Pressure (millibars)'],axis=1)


# In[70]:


del train['Pressure (millibars)']


# In[71]:


del train['day']
del train['weekday']


# In[72]:


train


# In[73]:


# visualize the relationship between the features and the response using scatterplots
sns.pairplot(weather, x_vars=['Temperature (C)','Humidity','Wind Speed (km/h)','Wind Bearing (degrees)','Visibility (km)','precip_type','summary','Pressure (millibars)'], y_vars='Apparent Temperature (C)', height=7, aspect=0.7, kind='reg')


# In[74]:


sns.pairplot(weather, x_vars=['year','month'], y_vars='Apparent Temperature (C)', height=7, aspect=0.7, kind='reg')


# ## imbalanced data

# *** in this part we do not touch the test set!

# the data is imbalanced as we can see from the histogram below

# In[75]:


# Histogram 
# from random import sample
# data = sample(range(1, 1000), 100)
plt.hist(train)


# In[76]:


# Histogram 
plt.hist(X_train)


# In[ ]:


#!pip install imblearn
get_ipython().system('pip uninstall -v scikit-learn')


# In[ ]:


get_ipython().system('pip install -v scikit-learn')


# In[ ]:


from collections import Counter
# from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy import where

# summarize class distribution
counter = Counter(y_train)
print(counter)

oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)


counter = Counter(y_train)
print(counter)

# scatter plot of examples by class label
for label, _ in counter.items():
    row_ix = where(y_train == label)[0]
    pyplot.scatter(X_train[row_ix, 0], X_train[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()


# In[ ]:





# ## dummy model

# In[84]:


from sklearn.dummy import DummyRegressor
#X = np.array([1.0, 2.0, 3.0, 4.0])
#y = np.array([2.0, 3.0, 5.0, 10.0])
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train,X_test, y_test)
dummy_regr.predict(X_train, y_train)
dummy_regr.score(X_train, y_train,X_test, y_test)


# In[ ]:




