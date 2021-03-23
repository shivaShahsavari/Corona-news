import requests
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re

headlines = []
dates = []
labels=[]
links=['https://www.government.nl/topics/coronavirus-covid-19/news','https://www.government.nl/topics/coronavirus-covid-19/news?page=2'
,'https://www.government.nl/topics/coronavirus-covid-19/news?page=3','https://www.government.nl/topics/coronavirus-covid-19/news?page=4',
'https://www.government.nl/topics/coronavirus-covid-19/news?page=5','https://www.government.nl/topics/coronavirus-covid-19/news?page=6',
'https://www.government.nl/topics/coronavirus-covid-19/news?page=7','https://www.government.nl/topics/coronavirus-covid-19/news?page=8',
'https://www.government.nl/topics/coronavirus-covid-19/news?page=9','https://www.government.nl/topics/coronavirus-covid-19/news?page=10'
,'https://www.government.nl/topics/coronavirus-covid-19/news?page=11']
#go through each url and get the parsed version
for i in range(1,len(links)):
    headers = {"Accept-Language": "en-US, en;q=0.5", 'User-agent': 'Super Bot 9000'}
    results = requests.get(links[i], headers=headers)  
    soup = BeautifulSoup(results.text, "html.parser")

    for sec in soup.find_all('a', class_="news"):
        a_part=[item.text.strip() for item in sec.find_all('h3')]
        headlines.append(a_part[0])
        labels.append("TRUE")


news = pd.DataFrame({'headlines':headlines,'Lables':labels})
news.to_csv('scrape_gov.csv')
  

