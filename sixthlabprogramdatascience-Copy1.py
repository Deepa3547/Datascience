#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


# In[19]:


df = pd.read_csv("C:\\Users\\Deepa kumari\\Downloads\\groceries - groceries.csv\\groceries - groceries.csv")


# In[20]:


df.shape



# In[21]:


df.shape


# In[22]:


df.info()


# In[23]:


df.describe()


# In[24]:


df.isnull().sum()


# In[25]:


df.fillna('')


# In[26]:


df_new = pd.DataFrame(df)
df_new = df_new.drop(['Item(s)'],axis=1)


# In[27]:


df_dum = pd.get_dummies(df_new)


# In[28]:


frequent_items = apriori(df_dum,min_support=0.01,use_colnames=True)
frequent_items


# In[29]:


rules = association_rules(frequent_items,metric='confidence',min_threshold=0.01)
rules


# In[30]:


rules = rules.sort_values(['support', 'confidence'], ascending=[False, False])


# In[31]:


rules


# In[ ]:




