# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:40:31 2020

@author: 20194066
"""
def translate_and_polarity(text):
    translation = Translator().translate(text, dest='en')
    print(translation.origin, ' -> ', translation.text)    
    vs = SentimentIntensityAnalyzer().polarity_scores(translation.text)
    print(vs)
    results = pd.DataFrame([translation.origin,translation.text, vs['neg'],  vs['neu'],  vs['pos'],  vs['compound']]).T
    return results

from newsapi.newsapi_client import NewsApiClient
import pandas as pd
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
api_key='a5c184912f3c454d98143a47ea6109e9'
newsapi = NewsApiClient(api_key)

municipalities = pd.read_excel("municipalities.xlsx")
municipality = municipalities['name'][0]
#municipality = 'brabant'
keyword = 'corona+ AND '
query = keyword + municipality
all_articles = newsapi.get_everything(q=query,
                                      language='nl')


# translate and apply vader
final_results = pd.DataFrame()
for i in range(len(all_articles['articles'])):

    title = all_articles['articles'][i]['title']
    description = all_articles['articles'][i]['description']
    content = all_articles['articles'][i]['content']
    
    results_title = translate_and_polarity(title)
    #results_description= translate_and_polarity(description)
    results_content= translate_and_polarity(content)
    
    results = pd.concat([results_title,results_content], axis=1)
   
    final_results = pd.concat([final_results,results])
#    
final_results.columns = ['title', 'title_en','negative', 'neutral', 'positive', 'compound', 
                         'content', 'content_en','negative', 'neutral', 'positive', 'compound']
final_results.to_csv("polarity_analysis_results.csv", index=False, encoding='utf-8-sig')

