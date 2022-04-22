#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn import metrics
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import statsmodels.api as sm
from scipy.stats import zscore
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Lasso


# # Importing Data

# In[47]:


data = pd.read_csv(r"C:\Users\serka\Desktop\YTÜ\YTÜ 3-2\proje\notCodeSmell_merged.csv")
data


# # Removing Missing Values and Some Features

# In[48]:


data = data.dropna()
data = data.drop(["JIRA_KEY","HATA_KAYDI_OZET","Unnamed: 0","securityRating","reliabilityRating","Unnamed: 0.1", "KLOC_EFOR_y","ESTIMATED_EFOR_y","faultFixingCommitHash","effort_y","confirmedIssues","nloc","complexity","effort_x","linesAdded","linesRemoved"],axis=1)
data = data.drop_duplicates()
data = data.reset_index(drop=True)


# In[49]:


data


# # Dividing Real Effort and Estimated Effort and Plotting Our Formula

# In[36]:


div_lines = data["KLOC_EFOR_x"]/data["ESTIMATED_EFOR_x"]
data["bölüm"]=div_lines


# In[37]:


data["bölüm"].plot()


# In[39]:


print(data['bölüm'].skew())
data['bölüm'].describe()


# In[40]:


data_plot = pd.DataFrame()
data_plot["KLOC_EFOR"] = data["KLOC_EFOR_x"]
data_plot["ESTIMATED_EFOR"] = data["ESTIMATED_EFOR_x"]


# In[41]:


data_plot


# In[23]:


print(data_plot['ESTIMATED_EFOR'].skew())
data_plot['ESTIMATED_EFOR'].describe()


# In[43]:


data_plot = data_plot[data_plot["ESTIMATED_EFOR"]<4028]
data_plot = data_plot[data_plot["ESTIMATED_EFOR"]>269]


# In[44]:


lines = data_plot.plot.line()


# In[46]:


r2_score(data["KLOC_EFOR_x"], data["ESTIMATED_EFOR_x"])


# # Train Test Split

# In[50]:


x = data.drop(["KLOC_EFOR_x"],axis=1)
y = data["KLOC_EFOR_x"]


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.3)


# In[52]:


y_train


# # Corelation Matrix

# In[53]:


plt.figure(figsize=(20,20))
cor = X_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()


# # Creating Datasets of Different Correlation Levels

# In[152]:


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[153]:


corr_features = correlation(X_train, 0.3)
KLOC_data_clean_0_3_train = X_train.drop(corr_features,axis=1)
KLOC_data_clean_0_3_test = X_test.drop(corr_features,axis=1)
KLOC_data_clean_0_3_train


# In[154]:


corr_features = correlation(X_train, 0.4)
KLOC_data_clean_0_4_train = X_train.drop(corr_features,axis=1)
KLOC_data_clean_0_4_test = X_test.drop(corr_features,axis=1)
KLOC_data_clean_0_4_train


# In[156]:


corr_features = correlation(X_train, 0.55)
KLOC_data_clean_0_5_train = X_train.drop(corr_features,axis=1)
KLOC_data_clean_0_5_test = X_test.drop(corr_features,axis=1)
KLOC_data_clean_0_5_train


# In[159]:


corr_features = correlation(X_train, 0.7)
KLOC_data_clean_0_6_train = X_train.drop(corr_features,axis=1)
KLOC_data_clean_0_6_test = X_test.drop(corr_features,axis=1)
KLOC_data_clean_0_6_train


# In[160]:


corr_features = correlation(X_train, 0.8)
KLOC_data_clean_0_7_train = X_train.drop(corr_features,axis=1)
KLOC_data_clean_0_7_test = X_test.drop(corr_features,axis=1)
KLOC_data_clean_0_7_train


# # Normalization

# In[18]:


scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(KLOC_data_clean_0_3_train)
KLOC_data_clean_0_3_train_scaled = pd.DataFrame(x_train_scaled)
x_test_scaled = scaler.fit_transform(KLOC_data_clean_0_3_test)
KLOC_data_clean_0_3_test_scaled = pd.DataFrame(x_test_scaled)

x_train_scaled = scaler.fit_transform(KLOC_data_clean_0_4_train)
KLOC_data_clean_0_4_train_scaled = pd.DataFrame(x_train_scaled)
x_test_scaled = scaler.fit_transform(KLOC_data_clean_0_4_test)
KLOC_data_clean_0_4_test_scaled = pd.DataFrame(x_test_scaled)

x_train_scaled = scaler.fit_transform(KLOC_data_clean_0_5_train)
KLOC_data_clean_0_5_train_scaled = pd.DataFrame(x_train_scaled)
x_test_scaled = scaler.fit_transform(KLOC_data_clean_0_5_test)
KLOC_data_clean_0_5_test_scaled = pd.DataFrame(x_test_scaled)

x_train_scaled = scaler.fit_transform(KLOC_data_clean_0_6_train)
KLOC_data_clean_0_6_train_scaled = pd.DataFrame(x_train_scaled)
x_test_scaled = scaler.fit_transform(KLOC_data_clean_0_6_test)
KLOC_data_clean_0_6_test_scaled = pd.DataFrame(x_test_scaled)

x_train_scaled = scaler.fit_transform(KLOC_data_clean_0_7_train)
KLOC_data_clean_0_7_train_scaled = pd.DataFrame(x_train_scaled)
x_test_scaled = scaler.fit_transform(KLOC_data_clean_0_7_test)
KLOC_data_clean_0_7_test_scaled = pd.DataFrame(x_test_scaled)
KLOC_data_clean_0_3_train_scaled


# # LinearRegression

# In[161]:


reg=LinearRegression()


# In[162]:


reg.fit(KLOC_data_clean_0_3_train,y_train)
y_pred=reg.predict(KLOC_data_clean_0_3_test)
plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('corr < 0.3')
r2_score(y_test,y_pred)


# In[163]:


reg.fit(KLOC_data_clean_0_4_train,y_train)
y_pred=reg.predict(KLOC_data_clean_0_4_test)
plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('corr < 0.5')
r2_score(y_test,y_pred)


# In[164]:


reg.fit(KLOC_data_clean_0_5_train,y_train)
y_pred=reg.predict(KLOC_data_clean_0_5_test)
plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('corr < 0.7')
r2_score(y_test,y_pred)


# In[165]:


reg.fit(KLOC_data_clean_0_6_train,y_train)
y_pred=reg.predict(KLOC_data_clean_0_6_test)
plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.xlabel('actual')
plt.ylabel('predicted')
plt.title('corr < 0.9')
r2_score(y_test,y_pred)


# In[166]:


reg.fit(KLOC_data_clean_0_7_train,y_train)
y_pred=reg.predict(KLOC_data_clean_0_7_test)
plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('corr < 0.95')
r2_score(y_test,y_pred)


# In[172]:


reg.coef_
pd.DataFrame(reg.coef_, 
             KLOC_data_clean_0_7_train.columns, 
             columns=['coef'])\
            .sort_values(by='coef', ascending=False)

