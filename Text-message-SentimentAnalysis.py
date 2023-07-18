#!/usr/bin/env python
# coding: utf-8

# # SENTIMENT ANALYSIS OF TEXT MESSAGES 

# Sentiment analysis of text messages is the process of determining the `sentiment` or `emotional tone` expressed. It involves analyzing the text to identify and categorize the sentiment as `positive`, `negative`, or `neutral`. The goal is to understand the subjective opinions, attitudes, or emotions conveyed by the text.

# In this project, `TextBlob` is used in sentiment analysis of text messages, offering different functionalities and approaches. 
# `Neattext` is used for data cleaning an pre-processing of data

# - TextBlob - 
# TextBlob is a Python library that provides a simple and intuitive API for natural language processing tasks, including sentiment analysis. It offers built-in sentiment analysis capabilities based on pre-trained models.TextBlob's sentiment analysis feature provides a polarity score and subjectivity score for a given text. The polarity score indicates the sentiment as positive (score > 0), negative (score < 0), or neutral (score = 0), while the subjectivity score represents the degree of subjective or objective nature of the text

# - NeatText - 
# NeatText is a Python library that offers text preprocessing functions for cleaning and normalizing text data. It provides various text cleaning operations to remove noise, normalize text, and handle common text-related issues.NeatText offers functions to remove special characters, URLs, email addresses, numbers, and other unwanted elements from text. It helps to eliminate noise and focus on the meaningful content for sentiment analysis.

# Using `seaborn` library Exploratory Data Analysis is also done to visualize the characteristics of the dataset.

# ## Statistical Analysis, Visualization, Data processing and Sentimental Anaysis

# ### About the dataset

# The dataset initialy contains `2` features
# 1. **Emotion**
# 2. **Text**
# 
# Emotion has 8 particulars namely,
# - Joy
# - Anger
# - Sad
# - Surprise
# - Disgust
# - Shame
# - Neutral
# - Fear
# 
# Text has the messages on which the Analysis is to be performed

# ### Importing necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import neattext.functions as nfx
import seaborn as sns
import matplotlib.pyplot as plt


# ### Importing the data

# In[2]:


df=pd.read_csv("E:\DA_PROJECTS\SENTIMENT-ANALYSIS\dataset.csv")
df.head()


# In[3]:


df.tail()


# ### Statistical Exploration of the data

# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


df.dtypes


# In[8]:


df['Emotion'].value_counts()


# ### Visualizing the emotion feature

# In[9]:


df['Emotion'].value_counts().plot(kind='bar')


# In[10]:


plt.figure(figsize=(8,5))
plt.title("EMOTION PLOT")
sns.countplot(x='Emotion',data=df)
plt.show()


# ### Performing Sentiment Analysis using TextBlob

# In[11]:


from textblob import TextBlob


# In[12]:


def get_sentiment(text):
    blob=TextBlob(text)
    sentiment= blob.sentiment.polarity
    if sentiment >0:
        result='Positive'
    elif sentiment <0:
        result= 'Negative'
    else:
        result= 'Neutral'
    return result


# In[13]:


get_sentiment('I was late to my class')


# Analysed result using TextBlob is appied to every "TEXT" feature of the data to predict the emotional tone. The obtained result is stored as a seperate feature in the dataset and compared with the actual emotion   

# In[14]:


df['Sentiment']=df['Text'].apply(get_sentiment)
df.head()


# A rough comparison is made for actual and predicted emotion given by the text. We can see that different emotions are differently predicted and categorized for different messages. 

# In[15]:


#comparison of actual and predicted
df.groupby(['Emotion','Sentiment']).size()


# We can observe that different emotions are differently categorized. As mentioned, `Joy` is predicted as 
# -  `Negative` in 1682 messages
# -  `Neutral` in 3649 messages
# -  `Positive` in 5714 messages

# ### Plotting predicted emotion using TextBlob 

# In[16]:


sns.catplot(x='Emotion',hue='Sentiment',data=df,kind='count',height=4,aspect=1.5)
plt.title("PREDICTED EMOTION")


# ### Text Cleaning

# This function returns a list of all the attributes and methods available in the `nfx` module of the `neattext` library performing the following functions
# - removes stopwords
# - removes punctuations
# - removes usernames
# - removes urls
# - removes emails
# - removes hashtags
# - removes numbers
# - removes special characters
# - removes multiple spaces

# In[17]:


dir(nfx)


# In[18]:


df['Text']


# Displaying clean text which got rid of `stopwords`, `punctuations`, `user handles` as a new feature in the dataset

# In[19]:


df['Clean_text']=df['Text'].apply(nfx.remove_stopwords)


# In[20]:


df['Clean_text']=df['Clean_text'].apply(nfx.remove_punctuations)


# In[21]:


df['Clean_text']=df['Clean_text'].apply(nfx.remove_userhandles)


# In[22]:


df[['Text','Clean_text']]


# The `Counter` class is a container that provides a convenient way to count the occurrences of elements in a collection or iterable. It is a specialized dictionary subclass where elements in the collection are stored as keys, and their counts are stored as values

# In[23]:


from collections import Counter


# In[24]:


def extract_keywords(text,num=50):
    tokens= [tok for tok in text.split()]
    most_common_tokens=Counter(tokens).most_common(num)
    return dict(most_common_tokens)


# In[25]:


emotion_list=df['Emotion'].unique().tolist()


# Lisitng the different emotions displayed by the text messages

# In[26]:


emotion_list


# Displaying the clean text messages which shows `Joy` emotion

# In[27]:


joy_list=df[df['Emotion']=='joy']['Clean_text'].tolist()
joy_list


# In[28]:


joy_docx=''.join(joy_list)
joy_docx


# ### Keyword Extraction

# In[29]:


keyword_joy=extract_keywords(joy_docx)
keyword_joy


# In[30]:


def plot_common(mydict,name):
    df_1=pd.DataFrame(mydict.items(),columns=['token','count'])
    plt.figure(figsize=(20,8))
    plt.title("Plot of words depicting {}".format(name))
    sns.barplot(x='token',y='count',data=df_1)
    plt.xticks(rotation=90)
    plt.show()


# Visualizing the keywords depicting `Joy` emotion

# In[31]:


plot_common(keyword_joy,"Joy")


# In[32]:


anger_list=df[df['Emotion']=='anger']['Clean_text'].tolist()


# In[33]:


anger_docx=''.join(anger_list)


# In[34]:


keyword_anger=extract_keywords(anger_docx)
keyword_anger


# Visualizng the keywords depicting `Anger` emotion

# In[35]:


plot_common(keyword_anger,"Anger")


# ### Word Cloud

# A word cloud is a visualization technique used in sentiment analysis to represent the most frequent words or terms in a given text corpus. It provides a visual summary of the text data by displaying words in varying sizes, where the size corresponds to their frequency or importance.
# 
# In sentiment analysis, a word cloud can be created to visualize the most commonly occurring words in positive, negative, or neutral sentiment categories. By analyzing the word cloud, one can quickly gain insights into the dominant sentiments expressed in a text corpus and identify the key themes or topics associated with each sentiment.

# ### Importing Word Coud

# In[36]:


from wordcloud import WordCloud


# In[37]:


def plot_cloud(docx):
    myWC= WordCloud().generate(docx)
    plt.figure(figsize=(20,10))
    plt.imshow(myWC,interpolation='bilinear')
    plt.axis('off')
    plt.show()


# In[38]:


plot_cloud(joy_docx)


# In[39]:


plot_cloud(anger_docx)


# ## Model Prediction -  Machine Learning

# ### Importing necessary machine learning libraries

# In[40]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix
from sklearn.model_selection import train_test_split


# Mapping the features to a variable

# In[41]:


x_label=df['Clean_text']
y_label=df['Emotion']


# In[42]:


x_label


# In[43]:


y_label


# ### Model Fitting

# In[44]:


cv=CountVectorizer()
x=cv.fit_transform(x_label)


# In[45]:


cv.get_feature_names_out()


# In[46]:


x_train,x_test,y_train,y_test=train_test_split(x,y_label,test_size=0.2)


# Logistic Regression is used for training the model and predicting the message 

# In[47]:


model=LogisticRegression()
model.fit(x_train,y_train)


# Accuracy Score of the model is calculated

# In[48]:


model.score(x_test,y_test)


# In[49]:


y_pred=model.predict(x_test)
y_pred


# A sample text is given to test the model for emotion prediction

# In[50]:


sample=["get lost idiot"]


# In[51]:


vt=cv.transform(sample).toarray()
model.predict(vt)


# Probabilty of the emotion occurence is calculated

# In[52]:


#probability check
model.predict_proba(vt)


# #### Emotion classes

# In[53]:


model.classes_


# In[54]:


def pred_emotion(sample,model):
    vect=cv.transform(sample).toarray()
    pred=model.predict(vect)
    pred_proba=model.predict_proba(vect)
    pred_percetage=dict(zip(model.classes_,pred_proba[0]))
    print("Prediction : {}, Prediction Score: {}".format(pred[0],np.max(pred_proba)))
    print(pred[0])
    return pred_percetage


# The emotion is predicted and the corresponding prediction score is displayed

# In[55]:


pred_emotion(sample,model)


# In[56]:


pred_emotion(["I love running"],model)


# ### Model Evaluation

# In[57]:


print(classification_report(y_test,y_pred))


# ### Constructing a confusion matrix for the model

# In[58]:


confusion_matrix(y_test,y_pred)


# In[59]:


plot_confusion_matrix(model,x_test,y_test)

