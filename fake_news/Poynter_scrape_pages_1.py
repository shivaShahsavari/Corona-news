# -*- coding: utf-8 -*-
"""
Created on Thu May 14 09:17:43 2020

@author: 20195474
"""
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:54:13 2020

@author: 20195474
"""

import requests
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re

#load url list
url_datf = pd.read_csv('Poynter_page_list.csv',header=0)

headlines = []
dates = []
country=[]
factCheckedBy = []
labels=[]
url_list = []
soup = []

#go through each url and get the parsed version
for i in range(len(url_datf)):
    url = url_datf['link'].iloc[i]
    print(url)
    headers = {"Accept-Language": "en-US, en;q=0.5", 'User-agent': 'Super Bot 9000'}
    url_list.append(url)
    results = requests.get(url, headers=headers)  
    soup = BeautifulSoup(results.text, "html.parser")

    #body = soup.find_all('div', class_='post-wrapper container')

    for simay in soup.find_all('div', class_="post-container"):
        a=simay.h2.a.text
        chunks = re.split('[:]',a)
        headlines.append(chunks[1])
        labels.append(chunks[0])
        factCheckedBy.append(re.sub(r'Fact-Checked by:', '',simay.p.text, flags=re.MULTILINE))
        b=simay.strong.text
        chunks2= re.split('[|]',b)
        dates.append(chunks2[0])
        country.append(chunks2[1])

news = pd.DataFrame({'news_title':headlines,'Lables':labels,'news_date':dates,'country':country,'fact_checking_source':factCheckedBy})
news.to_csv('scrape_Poynter_merged3.csv')
    

    

















