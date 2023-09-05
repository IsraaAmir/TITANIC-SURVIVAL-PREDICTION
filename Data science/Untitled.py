#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as seab
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Read Data

# In[2]:


Titanic_df = pd.read_csv('C:/Users/user/Downloads/tested.csv')


#  Data Collection & Processing

# In[3]:


Titanic_df.head()


# In[4]:


Titanic_df.shape


# In[5]:


Titanic_df.info()


# In[6]:


Titanic_df.describe()


# In[7]:


Titanic_df.isnull()


# In[8]:


Titanic_df.isnull().sum()


# In[9]:


Titanic_df['Age'] = Titanic_df['Age'].fillna(Titanic_df['Age'].mean())
Titanic_df['Fare'] = Titanic_df['Fare'].fillna(Titanic_df['Fare'].mean())


# In[10]:


Titanic_df['Embarked'] = Titanic_df['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2])


# In[11]:


Titanic_df['Sex'].unique()
Titanic_df['Sex'] = Titanic_df['Sex'].map({'female': 1, 'male': 0})


# In[12]:


Titanic_df['Age'] = Titanic_df['Age'].astype(int)
Titanic_df['Fare'] = Titanic_df['Fare'].astype(int)


# In[13]:


# Drop Cabin


# In[14]:


Titanic_df = Titanic_df.drop(columns = 'Cabin' , axis = 1)


# In[15]:


Titanic_df.drop(columns = 'Name' , axis = 1, inplace = True)
Titanic_df.drop(columns = 'Ticket' , inplace = True)
Titanic_df.drop(columns = 'PassengerId' ,inplace = True)


# In[16]:


Titanic_df.head()


# Visualization

# In[17]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
seab.histplot(data=Titanic_df, x='Age', hue='Survived', multiple='stack', bins=20, ax=axes[0])
seab.histplot(data=Titanic_df, x='Fare', hue='Survived', multiple='stack', bins=20, ax=axes[1])
axes[0].set_title('Age Histogram with Survival')
axes[1].set_title('Fare Histogram with Survival')


# In[18]:


# Bar Chart
seab.countplot(data=Titanic_df, x='Survived')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.title('Survival Count (0 = No, 1 = Yes)')
plt.show()


# splite data to get col survived

# In[19]:


x = Titanic_df.drop('Survived', axis=1)
y = Titanic_df.iloc[:,1]


# In[20]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


# Model

# In[21]:


LR = LogisticRegression(solver='liblinear', max_iter=200)
LR.fit(x_train, y_train)
y_pred = LR.predict(x_test)
LRAcc = accuracy_score(y_pred,y_test)
print('Logistic regression accuracy: {:.2f}%'.format(LRAcc*100))


# In[22]:


# Initialize and train different classifiers

knn = KNeighborsClassifier()

knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
KnnAcc = accuracy_score(y_pred,y_test)
print('KNeighborsClassifier accuracy: {:.2f}%'.format(KnnAcc*100))


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





# In[ ]:





# In[ ]:





# In[ ]:




