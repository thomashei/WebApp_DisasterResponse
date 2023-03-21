#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[2]:


# import libraries
import pandas as pd
from sqlalchemy import create_engine


# In[3]:


# load data from database
engine = create_engine('sqlite:///disaster_response.db')
df = pd.read_sql_table('messages_categories', engine)
X = df['message']
Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)


# ### 2. Write a tokenization function to process your text data

# In[5]:


# import libraries
import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# In[6]:


def tokenize(text):

    # normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return tokens


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[7]:


# import libraries
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier


# In[8]:


pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[9]:


# import libraries
from sklearn.model_selection import train_test_split

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# train pipeline
pipeline.fit(X_train, y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[10]:


# import libraries
from sklearn.metrics import classification_report

# test model
y_pred = pipeline.predict(X_test)

# print classification report
for i, col in enumerate(Y.columns):
    print(col)
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[ ]:


parameters = 

cv = 


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[ ]:





# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:





# ### 9. Export your model as a pickle file

# In[ ]:





# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




