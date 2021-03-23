import pandas as pd
import numpy as np
import re
from nltk.tokenize import sent_tokenize
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import operator
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
##Apply model on Dutch News
data=pd.read_csv("reliable_all.csv", encoding = "utf-8",header=0)
df=pd.DataFrame(data)
#Index(['news_title', 'Lables'], dtype='object')
my_tags=['Misleading','FALSE','TRUE']
###########################################################
####Stemming

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
###############
#Convert treebank tags to Wordnet tag
import nltk
from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
###############

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    stem_sentence=[]
    for word in token_words:
        pos=nltk.pos_tag(word)[0][1][0].upper()
        wntag = get_wordnet_pos(pos)
        if len(word)>2:
            if wntag is None:
                stem_sentence.append(wordnet_lemmatizer.lemmatize(word)) 
            else:
                stem_sentence.append(wordnet_lemmatizer.lemmatize(word, pos=wntag))
        else:
            continue    
    return stem_sentence

stem_desc=[]
df['stem_title']=''
for i in range(0,len(df['news_title'])):
    df['stem_title'][i]=stemSentence(df['news_title'][i])
##########################################################
from gensim.models import Word2Vec
import gensim
import logging

wv = gensim.models.KeyedVectors.load_word2vec_format(
    "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz", binary=True, limit=300000)
wv.init_sims(replace=True)

def word_averaging(wv, words):
    all_words, mean = set(), []
    
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)
    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        return np.zeros(wv.vector_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def  word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, post) for post in text_list ])


##########################################################
##Cross Validation
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['stem_title'],df['new_label1'],test_size=0.2)

X_train_word_average = word_averaging_list(wv,Train_X)
X_test_word_average = word_averaging_list(wv,Test_X)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
#print(Encoder.inverse_transform(Train_Y))
# Classifier - Algorithm - SVM
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg = logreg.fit(X_train_word_average, Train_Y)
y_pred = logreg.predict(X_test_word_average)
#print('accuracy %s' % accuracy_score(y_pred, Test_Y))
#print(classification_report(Test_Y, y_pred,target_names=my_tags))
'''
new_label = ['Misleading','FALSE','Unproven','TRUE']
accuracy 0.8264984227129337
              precision    recall  f1-score   support

  Misleading       0.86      0.91      0.88       219
       FALSE       0.39      0.30      0.34        30
    Unproven       0.85      0.83      0.84        64
        TRUE       1.00      0.25      0.40         4
#####################################################
new_label1 = ['Misleading','FALSE','TRUE']
accuracy 0.861198738170347
              precision    recall  f1-score   support

  Misleading       0.91      0.91      0.91       230
       FALSE       0.41      0.50      0.45        22
        TRUE       0.87      0.82      0.84        65

    accuracy                           0.84       317
   macro avg       0.68      0.70      0.69       317
weighted avg       0.85      0.84      0.84       317
'''
#####################################################
##Apply model on Dutch News
data=pd.read_csv("Dutch_news_translated_.csv", encoding = "utf-8",header=0)
dutch_news=pd.DataFrame(data)
#Index(['headlines', 'content', 'date', 'headlines_en', 'content_en'], dtype='object')
dutch_news['stem_headline_en']=''
for i in range(0,len(dutch_news['headlines_en'])):
    dutch_news['stem_headline_en'][i]=stemSentence(dutch_news['headlines_en'][i])

X_dutch_news_average = word_averaging_list(wv,dutch_news['stem_headline_en'])
y_ducth_pred = logreg.predict(X_dutch_news_average)
dutch_news['reliable']=Encoder.inverse_transform(y_ducth_pred)
dutch_news.to_csv("Dutch_news_translated_reliable1.csv")

