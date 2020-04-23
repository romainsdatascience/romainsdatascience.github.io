---
title: "Web Scraping Trump Speeches on Coronavirus"
date: 2020-04-10
tags: [Web Scraping]
excerpt: "Fast Web Scraping of Trump's speeches"
classes: wide
---

## Import modules


```python
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import pickle
```

## Web Scraping the data


```python
# Function that scrapes transcript data from rev.com
def urls_to_dataframe(urls):

    titles = []
    dates = []
    transcripts = []

    for url in urls:
        page_html = requests.get(url).text
        soup = BeautifulSoup(page_html, "html.parser")
        title = soup.h1.findAll("span", {'class': 'fl-heading-text'})[0].text
        date = soup.findAll("div", {'class': 'fl-rich-text'})[0].text
        text = [p.text for p in soup.find("div", {'class': 'fl-callout-text'}).find_all('p')]

        titles.append(title)
        dates.append(date)
        transcripts.append(text)

    return pd.DataFrame({'titles': titles, 'dates': dates, 'transcripts': transcripts})
```


```python
# URLs of transcripts in scope
links = ['https://www.rev.com/blog/transcripts/donald-trump-coronavirus-press-conference-transcript-april-20',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-press-conference-transcript-march-26-plans-to-classify-counties-by-covid-19-risk',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-press-conference-transcript-april-19',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-press-conference-transcript-april-18',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-press-conference-transcript-april-17',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-press-briefing-transcript-april-16',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-press-briefing-transcript-april-15',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-press-briefing-transcript-april-14-trump-halts-who-funding',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-press-conference-transcript-april-13',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-press-conference-transcript-april-21',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-update-transcript-feb-29-warns-not-to-travel-to-italy-south-korea',
         'https://www.rev.com/blog/transcripts/donald-trump-mike-pence-and-coronavirus-update-transcript-march-9',
         'https://www.rev.com/blog/transcripts/donald-trump-speech-transcript-on-coronavirus-ban-on-europe-travel',
         'https://www.rev.com/blog/transcripts/donald-trump-speech-transcript-declares-coronavirus-national-emergency',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-press-conference-transcript-march-14',
         'https://www.rev.com/blog/transcripts/donald-trump-and-coronavirus-task-force-news-conference-transcript-march-15',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-update-march-16',
         'https://www.rev.com/blog/transcripts/coronavirus-task-force-briefing-transcript-march-17-trump-pence-mnuchin-speak-about-covid-19',
         'https://www.rev.com/blog/transcripts/donald-trump-and-coronavirus-task-force-news-briefing-march-18-invokes-defense-production-act-calls-covid-19-chinese-virus',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-briefing-transcript-march-19-trump-takes-shots-at-the-media',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-march-20-press-conference-transcript-trump-spars-with-reporters-in-fiery-briefing',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-update-transcript-march-21',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-briefing-transcript-march-22-national-guard-activated-in-new-york-california-and-washington-state',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-briefing-transcript-march-23',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-briefing-transcript-march-24',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-press-conference-transcript-march-25',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-press-conference-transcript-march-26-plans-to-classify-counties-by-covid-19-risk',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-briefing-transcript-march-27-trump-says-michigan-and-washington-governors-not-appreciative',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-briefing-transcript-march-29-trump-extends-task-force-guidelines-to-april-30',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-press-conference-transcript-march-30',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-briefing-transcript-march-31-painful-weeks-ahead',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-briefing-transcripts-april-1',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-briefing-april-2',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-briefing-transcript-april-3-new-cdc-face-mask-recommendations',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-transcript-april-4',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-briefing-transcript-april-5',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-briefing-transcript-april-6',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-briefing-april-7',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-briefing-transcript-april-8',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-briefing-april-9',
         'https://www.rev.com/blog/transcripts/donald-trump-coronavirus-task-force-press-conference-transcript-april-10'
        ]
```


```python
data = urls_to_dataframe(links)
```


```python
pickle.dump(test, open('dataframetrump', 'wb'))
OL = pickle.load(open('dataframetrump', 'rb'))
```

### Cleaning the transcripts column


```python
data.transcripts[1][10]
```




    'Johnny: (12:14) I have concerns, a very important vessel.'



For the moment, the transcripts column contains for each press conference a list of paragraphs. Indeed, other intervenants speak during these conferences. The analysis focuses only on Trump. Therefore, a first step in the cleaning of the dataframe is to keep only paragraphs where Trump speaks and convert this list of lists in a full text.


```python
def clean_transcripts(self):
    text = []
    for x in self:
        text.append(re.findall(r'(^Donald Trump.*|^President Trump.*)', x))

    return text
```


```python
clean = data.transcripts.apply(clean_transcripts)
```


```python
def flat(x):
    flat_list = []
    for sublist in x:
        for item in sublist:
            flat_list.append(item)

    return ' '.join(flat_list)
```


```python
data.transcripts = clean.apply(flat)
```


```python
data.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>titles</th>
      <th>dates</th>
      <th>transcripts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Donald Trump Coronavirus Press Conference Tran...</td>
      <td>Apr 20, 2020</td>
      <td>Donald Trump: (01:12) Thank you very much ever...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Donald Trump &amp; Coronavirus Task Force Press Co...</td>
      <td>Mar 26, 2020</td>
      <td>Donald Trump: (00:15) Thank you very much. Tha...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Donald Trump Coronavirus Press Conference Tran...</td>
      <td>Apr 19, 2020</td>
      <td>Donald Trump: (00:01) Thank you very much. I’d...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Donald Trump Coronavirus Press Conference Tran...</td>
      <td>Apr 18, 2020</td>
      <td>Donald Trump: (02:13) Thank you very much. Goo...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Donald Trump Coronavirus Press Conference Tran...</td>
      <td>Apr 17, 2020</td>
      <td>Donald Trump: (02:30) Thank you. Thank you ver...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Donald Trump Coronavirus Press Conference Tran...</td>
      <td>Apr 16, 2020</td>
      <td>Donald Trump: (03:19) Thank you very much. Our...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Donald Trump Coronavirus Press Briefing Transc...</td>
      <td>Apr 15, 2020</td>
      <td>Donald Trump: (00:09) Okay. Thank you very muc...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Donald Trump Coronavirus Press Briefing Transc...</td>
      <td>Apr 14, 2020</td>
      <td>Donald Trump: (00:03) Very importantly, I’d li...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Donald Trump Coronavirus Press Conference Tran...</td>
      <td>Apr 13, 2020</td>
      <td>Donald Trump: (11:23) Thank you very much, eve...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Donald Trump Coronavirus Press Conference Tran...</td>
      <td>Apr 21, 2020</td>
      <td>Donald Trump: (03:04) Thank you very much ever...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Donald Trump Coronavirus Update Transcript Feb...</td>
      <td>Feb 29, 2020</td>
      <td>Donald Trump: (00:00) Thank you very much, eve...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Donald Trump, Mike Pence, and Coronavirus Task...</td>
      <td>Mar 9, 2020</td>
      <td>Donald Trump: (03:01) Thank you very much. We ...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Donald Trump Speech Transcript on Coronavirus,...</td>
      <td>Mar 11, 2020</td>
      <td>President Trump: (00:00) My fellow Americans, ...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Donald Trump Speech Transcript: Declares Coron...</td>
      <td>Mar 13, 2020</td>
      <td>Donald Trump: (00:00) To unleash the full powe...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Donald Trump Coronavirus Press Conference Tran...</td>
      <td>Mar 14, 2020</td>
      <td>Donald Trump: (00:19) Thank you very much. Tha...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Donald Trump and Coronavirus Task Force News C...</td>
      <td>Mar 15, 2020</td>
      <td>Donald Trump: (00:06) Beautiful day outside. A...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Donald Trump Coronavirus Task Force Update: Ma...</td>
      <td>Mar 16, 2020</td>
      <td>Donald Trump: (09:24) I’m glad to see that you...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Coronavirus Task Force Briefing Transcript Mar...</td>
      <td>Mar 17, 2020</td>
      <td>Donald Trump: (18:28) Thank you very much ever...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Donald Trump and Coronavirus Task Force News B...</td>
      <td>Mar 18, 2020</td>
      <td>Donald Trump: (19:18) Thank you very much. I w...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Donald Trump Coronavirus Task Force Briefing T...</td>
      <td>Mar 19, 2020</td>
      <td>Donald Trump: (01:51) Thank you very much. I t...</td>
    </tr>
  </tbody>
</table>
</div>
