import requests
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re

headlines = []
dates = []
labels=[]

#go through each url and get the parsed version
headers = {"Accept-Language": "en-US, en;q=0.5", 'User-agent': 'Super Bot 9000'}
results = requests.get("https://www.who.int/emergencies/diseases/novel-coronavirus-2019/media-resources/news", headers=headers)  
soup = BeautifulSoup(results.text, "html.parser")

for sec in soup.find_all('div', class_="list-view horizontal-title-only highlight-widget bg-white"):
    #print(str(sec))
    a_part=re.findall('<p class="heading text-underline".+?</p>',str(sec))
    b_part=re.sub(r'<p class="heading text-underline">', '', str(a_part), flags=re.MULTILINE)
    bb_part=re.sub(r'</p>', '', str(b_part), flags=re.MULTILINE)
    headlines.append(bb_part.replace("'","").replace("[","").replace("]",""))
    labels.append("TRUE")
    c_part=re.findall('<p class="sub-title".+?</p>',str(sec))
    d_part=re.sub(r'<p class="sub-title">', '', str(c_part), flags=re.MULTILINE)
    b=re.sub(r'</p>', '', str(d_part.replace("'","").replace("[","").replace("]","")), flags=re.MULTILINE)
    chunks2= re.split('[,|I]',b)
    dates.append(chunks2[0])

news = pd.DataFrame({'headlnes':headlines,'Lables':labels,'news_date':dates})
news.to_csv('scrape_who.csv')
  

