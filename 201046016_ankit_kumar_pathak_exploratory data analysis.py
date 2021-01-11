#!/usr/bin/env python
# coding: utf-8

# In[87]:


# NAME- ANKIT KUMAR PATHAK
#ROLL NO -201046016


# EXPLORATORY DATA ANALYSIS
# This dataset was originally from the National Institute of Diabetes
# and Digestive and Kidney Diseases. 
# The purpose of the dataset is to diagnostically predict whether a patient
# has diabetes based on the specific diagnostic measures included in the data set. 
# Various restrictions have been imposed on the selection of these 
# samples from a larger database


# In[199]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.graph_objs as go
import plotly.offline as py


# In[200]:


# Reading Data 
diabetes=pd.read_csv('diabetes.csv')

df = diabetes.copy()
df.head()

# In[201]:


df


# In[202]:


df.info()


# In[ ]:





# In[203]:


df.describe().T


# In[204]:


# Handling with Missing Values
# In this dataset missing data are filled with 0. First, we are gonna change zeros with NaN


# In[205]:


df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness',
                                                                      'Insulin','BMI']].replace(0, np.NaN)


# In[206]:


# After filling the 0s with the value of NaN, the missing values 
# will be visualized. We use the missingno library for this.


# In[207]:


msno.bar(df,figsize=(10,6))


# In[208]:


msno.heatmap(df);


# In[209]:


# We will fill in each missing value with its median value.


# In[210]:


def median_target(var):   
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp


# In[211]:


columns = df.columns
columns = columns.drop("Outcome")
for i in columns:
    median_target(i)
    df.loc[(df['Outcome'] == 0 ) & (df[i].isnull()), i] = median_target(i)[i][0]
    df.loc[(df['Outcome'] == 1 ) & (df[i].isnull()), i] = median_target(i)[i][1]


# In[212]:


# After filling if we examine null values in dataset, we will see there are not any missing values.


# In[213]:


df.isnull().sum()


# In[214]:


# Data Visualization


# In[215]:


# Histogram-A histogram is a bar graph representation of a grouped data distribution


# In[216]:


df.hist(bins=20,figsize = (15,15));


# In[217]:


# count plot


# In[218]:


plt.title("Distribution of Outcome")
sns.countplot(df["Outcome"], saturation=1)


# In[219]:


# pie plot


# In[220]:


def PlotPie(df, nameOfFeature):
    labels = [str(df[nameOfFeature].unique()[i]) for i in range(df[nameOfFeature].nunique())]
    values = [df[nameOfFeature].value_counts()[i] for i in range(df[nameOfFeature].nunique())]

    trace=go.Pie(labels=labels,values=values)

    py.iplot([trace])

PlotPie(df, "Outcome")


# In[221]:


# Correlation
def corr_to_target(dataframe, target, title=None, file=None):
    plt.figure(figsize=(4,6))
    sns.heatmap(dataframe.corr()[[target]].sort_values(target,
                                                        ascending=False)[1:],
                                                        annot=True,
                                                        cmap='coolwarm')
    
    plt.title(f'\n{title}\n', fontsize=18)
    
    plt.show();
    
    return

corr_to_target(df, "Outcome", title="Outcome")


# In[222]:


# Correlation matrix of variables with each other.


# In[223]:


corr_matrix = df.corr()
sns.clustermap(corr_matrix, annot=True, fmt=".2f")
plt.title("Correlation Between Features")


# In[198]:


import dtale


# In[169]:


dtale.show(df, ignore_duplicate=True)


# In[170]:


import sweetviz


# In[171]:


diabetes=pd.read_csv('diabetes.csv')
# importing sweetviz
import sweetviz as sv

#analyzing the dataset
advert_report = sv.analyze(df)
#display the report
advert_report.show_html('Advertising.html')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[23]:





# In[ ]:





# In[26]:





# In[ ]:





# In[ ]:




