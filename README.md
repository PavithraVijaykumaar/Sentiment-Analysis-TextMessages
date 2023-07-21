# Sentiment Analysis of TextMessages
This project is about the `Sentiment Analysis` of the text messages. Sentiment analysis of text messages is the process of determining the `sentiment` or `emotional` tone expressed. Here, `TextBlob` library is used to determine the emotional tone of the messages in the dataset. It involves analyzing the text to identify and categorize the sentiment as positive, negative, or neutral. For preprocessing the data and cleaning the text, `Neattext` is used. This project also focusses on developing an Machine Learning model to predict the text messages using `LogisticRegressor` and the corresponding confusion matrix is also obtained for the Machine Learning model.

## About the Libraries Used

- Pandas for converting the dataset to DataFrame for performing analysis
- Numpy for numerical functions
- Seaborn and Matplotlib for data visualization
- NeatText for text preprocessing
- TextBlob for performing Sentiment Analysis
- Wordcloud for displaying frequently repeated word showing a particular emotion

## About the dataset
  This dataset contains two features `Emotion` and the `Text`

Emotion has 8 particulars namely,
1. Joy
2. Anger
3. Sad
4. Surprise
5. Disgust
6. Shame
7. Neutral
8. Fear
Text has the messages on which the Analysis is to be performed.

Click to view the [Dataset](dataset.csv)

### Installed Libraries
    import pandas as pd
    import numpy as np
    import neattext.functions as nfx
    import seaborn as sns
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

## About the Libraries
  
### TextBlob 
TextBlob is a Python library that provides a simple and intuitive API for natural language processing tasks, including sentiment analysis. It offers built-in sentiment analysis capabilities based on pre-trained models.TextBlob's sentiment analysis feature provides a polarity score and subjectivity score for a given text. The polarity score indicates the sentiment as positive (score > 0), negative (score < 0), or neutral (score = 0), while the subjectivity score represents the degree of subjective or objective nature of the text.
Code for TextBlob in the analysis

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

### NeatText
NeatText is a Python library that offers text preprocessing functions for cleaning and normalizing text data. It provides various text cleaning operations to remove noise, normalize text, and handle common text-related issues.NeatText offers functions to remove special characters, URLs, email addresses, numbers, and other unwanted elements from text. It helps to eliminate noise and focus on the meaningful content for sentiment analysis.

    dir(nfx)
    df['Clean_text']=df['Text'].apply(nfx.remove_stopwords)
    df['Clean_text']=df['Clean_text'].apply(nfx.remove_punctuations)
    df['Clean_text']=df['Clean_text'].apply(nfx.remove_userhandles)

### WordCloud
Word cloud are visual representations of the most frequently occurring words in a given text.They are helpful for gaining insights into the overall sentiment of a text by visualizing the most common words.Words that appear more frequently will be displayed with a larger font size, making them stand out in the word cloud.
The code for WordCloud in the analysis

    def plot_cloud(docx):
      myWC= WordCloud().generate(docx)
      plt.figure(figsize=(20,10))
      plt.imshow(myWC,interpolation='bilinear')
      plt.axis('off')
      plt.show()

## Machine Learning Model

### Libraries Used

For building a ML model for prediction of the emotion, certain libraries are imported and used for training the existing dataset and build a model predicting the emotion displayed by the text messages.

      from sklearn.linear_model import LogisticRegression
      from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
      from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix
      from sklearn.model_selection import train_test_split
      
- Logistic Regression model is used for sentiment analysis to analyze text messages and determine their sentiment (positive, negative, or neutral). Logistic Regression is suitable for this task as it performs `multiclass classification` (positive, negative, or neutral sentiment) efficiently.
- CountVectorizer in sentimental Analysis is employed to convert a collection of `text messages` into a matrix of `token counts`. It is part of feature extraction process and is particularly useful for `converting raw text data into a numerical format` that machine learning algorithms can understand and process.
- Accuracy score is an evaluation metric to measure the performance of a classification model in sentiment analysis. It represents the `percentage of correctly classified text messages with the correct sentiment label` out of the total number of instances in the dataset
- Test-train splits the dataset into two subsets: a `training set` and `test set`. The training set is used to train the sentiment analysis model, and the test set is used to evaluate the model's performance. 
- Confusion matrix visualizes the performance of a classification model by comparing the `predicted sentiment labels` to the `actual sentiment labels`.

After training the model, the Accuracy score is given as,
         
      model.score(x_test,y_test)
which gives us an accuracy of `0.638597499640753`

A sample text is given for prediction

      sample=["get lost idiot"]
we obtain the prediction as `Prediction : anger, Prediction Score: 0.3194033804521572
anger`

      pred_emotion(["I love running"],model)
this gives the prediction of `Prediction : joy, Prediction Score: 0.44960583627020445
joy`

## Environment
The environment used for this analysis is Jupyter Notebook.

The code for this analysis can be viewed in [Jupyter Notebook](Text-message-SentimentAnalysis.ipynb)

To view the code as raw code click [Python](Text-message-SentimentAnalysis.py)
