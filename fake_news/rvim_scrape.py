import requests
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re

headlines = []
dates = []
labels=[]
links=['https://www.rivm.nl/en/news','https://www.rivm.nl/en/news?search=&onderwerp=&page=0%2C1',
'https://www.rivm.nl/en/news?search=&onderwerp=&page=0%2C2','https://www.rivm.nl/en/news?search=&onderwerp=&page=0%2C3']
#go through each url and get the parsed version
for i in range(1,len(links)):
    headers = {"Accept-Language": "en-US, en;q=0.5", 'User-agent': 'Super Bot 9000'}
    results = requests.get(links[i], headers=headers)  
    soup = BeautifulSoup(results.text, "html.parser")

    for sec in soup.find_all('div', class_="card-list news col-12 p-0"):
        a_part=re.findall('<h2 class="card-title">.+?</h2>',str(sec))
        b_part=re.sub(r'<h2 class="card-title">', '', str(a_part), flags=re.MULTILINE)
        bb_part=re.sub(r'</h2>', '', str(b_part), flags=re.MULTILINE)
        headlines.append(bb_part.replace("'","").replace("[","").replace("]",""))
        labels.append("TRUE")

    for sec in soup.find_all('div', class_="card-list news col-12 p-0"):
        aa_part=re.findall('<h3 class="card-title ">.+?</h3>',str(sec))
        ab_part=re.sub(r'<h3 class="card-title ">', '', str(aa_part), flags=re.MULTILINE)
        abb_part=re.sub(r'</h3>', '', str(ab_part), flags=re.MULTILINE)
        if abb_part is not None:
            headlines.append(abb_part.replace("'","").replace("[","").replace("]",""))
            labels.append("TRUE")
        else:
            continue

news = pd.DataFrame({'headlnes':headlines,'Lables':labels})
news.to_csv('scrape_rvim.csv')
  

