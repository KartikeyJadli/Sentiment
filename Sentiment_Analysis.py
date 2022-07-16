#Importing all necessary Libraries
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.stem.porter import *
stemmer = PorterStemmer()
from nltk.corpus import stopwords

# Reading the data
df=pd.read_csv("https://raw.githubusercontent.com/KartikeyJadli/Sentiment/main/Coronavirus_Tweets.csv",encoding='latin1')
print(df.head(15))

# Converting our dataset in DataFrame with two attributes
dataset=pd.DataFrame(df,columns=['OriginalTweet','Sentiment'])
dataset.head(10)
count=dataset["Sentiment"].value_counts()
labels=["Positive","Negative","Neutral","Extremely Postive","Extremly Negative"]
# plt.pie(count,labels=labels)
# plt.legend(["0","1","2","3","4"],labels)
# plt.show()
wp = { 'linewidth' : 1, 'edgecolor' : "Black" }
colors = ['yellow','green','skyblue','orange',"red"]
myexplode = [0.1, 0.1, 0.2, 0.3, 0.2]

# Creating autocpt arguments
def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)
 
# Creating plot
fig, ax = plt.subplots(figsize =(10, 7))
wedges, texts, autotexts = ax.pie(count,autopct = lambda pct: func(pct, count),explode = myexplode,labels = labels,shadow = True,startangle = 90,wedgeprops = wp,colors = colors)

# Adding legend
ax.legend(wedges, labels,title ="TWEETS BY PEOPLE",loc ="upper right",bbox_to_anchor =(1, 0, 0.5, 1))
plt.setp(autotexts, size = 8, weight ="bold")

# Creating a function to remove the pattern(@gmail.com)
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt
# Create new column with removed @user
df['Tweet'] = np.vectorize(remove_pattern)(df['OriginalTweet'], '@[\w]*')
df.head(10)

# Removing the links from the tweets
df['Tweet'] = df['Tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
df['Tweet'].head(15)

# Replacing/Removing the hashtags from the tweets
df['Tweet'] = df['Tweet'].str.replace('[^a-zA-Z#]+',' ')
df['Tweet'].head(20)

# Removing all words which have length less than 3
df['Tweet'] = df['Tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
df.head(10)

# Converting the tweets comments in array by splitting them
df['Tweet'] = df['Tweet'].apply(lambda x: x.split())
df['Tweet'].head(20)

# Converting all the words to their root words Eg:runner,running-->run
df['Tweet'] = df['Tweet'].apply(lambda x: [stemmer.stem(i) for i in x])
df.head(10)

# Again Joining the Tweets
for i in range(len(df['Tweet'])):
    df['Tweet'][i] = ' '.join(df['Tweet'][i])
df.head(20)

import nltk
nltk.download('stopwords')

# from nltk.corpus import stopwords(which does not add meaning to sentance like and,is the)
stop = stopwords.words('english')

# Removing all the stop words from tweets and store it in stopword_x
stopword_x=df['Tweet'].apply(lambda x: [item for item in x if item not in stop])
stopword_x.head(30)

# Considered only two atrributes preprocessed Tweets and sentiment and save it in cdf
cdf=df[['Tweet','Sentiment']]
cdf.head(10)

# Being sure that we do not have null values
cdf.isnull().sum()

from sklearn.model_selection import train_test_split
# Dividing our dataset in training and testing subset which have same labels
train,test= train_test_split(cdf,test_size = 0.4,random_state=0,stratify = cdf.Sentiment.values) 
print("Train shape : ", train.shape)
print("Test shape : ", test.shape)

# Countvectorizer count the each word and make them like an attribute
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
stop = list(stopwords.words('english'))
vectorizer = CountVectorizer(decode_error = 'replace',stop_words = stop)
print(vectorizer)

# X_train,X_test are the features(independent variable) on which our sentiment depends
X_train = vectorizer.fit_transform(train.Tweet.values)
X_test = vectorizer.transform(test.Tweet.values)

# y_train,y_test are the sentiments(dependent variable) which our model will predict
y_train = train.Sentiment.values
y_test = test.Sentiment.values

print("X_train.shape : ", X_train.shape)
print("X_test.shape : ", X_test.shape)
print("y_train.shape : ", y_train.shape)
print("y_test.shape : ", y_test.shape)

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay,confusion_matrix

# Using Stochastic Gradient Descent(SGD) to train our model
sgd_clf = SGDClassifier(loss = 'hinge', penalty = 'l2', random_state=0)

sgd_clf.fit(X_train,y_train)

# Predicting the sentiment for test dataset
sgd_prediction = sgd_clf.predict(X_test)

# Finding the Accuracy Score that how our model is?
sgd_accuracy = accuracy_score(y_test,sgd_prediction)
print("Training accuracy Score    : ",sgd_clf.score(X_train,y_train))
print("Test accuracy Score : ",sgd_accuracy )
print(classification_report(sgd_prediction,y_test))
fig, ax = plt.subplots(figsize=(10,10)) 
cm = confusion_matrix(y_test, sgd_prediction, labels=sgd_clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sgd_clf.classes_)
disp.plot(ax=ax)
plt.show()