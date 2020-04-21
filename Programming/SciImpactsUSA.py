import requests
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from datetime import date


headers = {"Accept-Language": "en-US, en;q=0.5"}
##only for last 24 hours
url = "https://news.google.com/topics/CAAqBwgKMPW5lwsw6uKuAw?hl=en-US&gl=US&ceid=US%3Aen"
results = requests.get(url, headers=headers)
soup = BeautifulSoup(results.text, "html.parser")

SciHeadlines=[]
a_part = soup.find_all('div', jslog='93789')
for container in a_part:
    b_part=container.find_all('article',jslog='85008')
    SciHeadlines.append(b_part) 

aa_part = soup.find_all('div', jslog='88374')
for container in aa_part:
    b_part=container.find_all('article',jslog='85008')
    SciHeadlines.append(b_part)

SciTitles=[]
for wrd in SciHeadlines:
    c_part=re.findall('<a class="DY5T1d" .+?</a>',str(wrd))
    d_part=re.sub(r'</a>', '', str(c_part), flags=re.MULTILINE)
    SciTitles.append(re.sub(r'<a class="DY5T1d" .+?>', '', str(d_part), flags=re.MULTILINE))

pd.DataFrame(SciTitles).to_csv('SciUsaNews_'+str(date.today())+'.csv')
