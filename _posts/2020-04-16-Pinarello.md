---
title: "Pinarello Android App Opportunity"
date: 2020-04-10
tags: [EDA]
header:
  image: "images/Pinarello_files/Pinarello5.jpeg"
excerpt: "Exploration Data Analysis on Google Play Store dataset"
classes: wide
---

In this project, we will put ourself in the place of a data scientist working for Pinarello.  Of course, this project is fictive but it allows me to work for the glamorous brand Pinarello for a moment...

Pinarello is an Italian bicycle manufacturer founded in 1952. Its product range is mainly composed of road, track and cyclo-cross bikes. Pinarello's bikes are expensive, optimized and incredibly good looking. Although it is easy, we can compare Pinarello to Ferrari in Formula 1.

Pinarello is looking for new ways to improve its marketing strategy and communication. A small team has been assembled to achieve this mission. My mission is to help the team to make the right decision and to work in collaboration with them to discover new ways to improve the marketing strategy of Pinarello.

Quickly the team decided to explore the possibility to develop **an application on the Android Store**. Indeed, this would enable Pinarello to:

1. Acquire information on the clients and prospects.
2. Enable Push notifications.
3. Increase customer loyalty.
4. Create a sense of community.
5. Increase interactions between brand lovers.

### My mission

My mission is to investigate if the development of a gaming mobile application is an appropriate marketing move for Pinarello and to get a global picture of the applications in Google Play Store.  


# Origin of the data

The dataset used for the analysis of the Android mobile app market originates from kaggle. The dataset contain details on applications that can be found on the Google Play Store. The dataset was created on the 2019-04-05. I chose to work with the minimal version consisting of 32.000 applications.

<a href="https://www.kaggle.com/gauthamp10/google-playstore-apps">Source for data and data description.</a>

# Data Cleaning

## Data profiling


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

all_data = pd.read_csv('Desktop/datasets_portfolio/pinarello/Google-Playstore-32K.csv')
```


```python
all_data.shape
```




    (32000, 11)




```python
all_data.columns
```




    Index(['App Name', 'Category', 'Rating', 'Reviews', 'Installs', 'Size',
           'Price', 'Content Rating', 'Last Updated', 'Minimum Version',
           'Latest Version'],
          dtype='object')




```python
all_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 32000 entries, 0 to 31999
    Data columns (total 11 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   App Name         32000 non-null  object
     1   Category         32000 non-null  object
     2   Rating           32000 non-null  object
     3   Reviews          31999 non-null  float64
     4   Installs         32000 non-null  object
     5   Size             32000 non-null  object
     6   Price            32000 non-null  object
     7   Content Rating   32000 non-null  object
     8   Last Updated     32000 non-null  object
     9   Minimum Version  32000 non-null  object
     10  Latest Version   31999 non-null  object
    dtypes: float64(1), object(10)
    memory usage: 2.7+ MB


Looking at the type of the columns, we can already see that we will have to work on the columns: **Rating, Installs, Size and Price**. Indeed, these columns have an object type while their type should preferably be numerical.

Glimpse at what the data look like:

```python
all_data.sample(10)
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
      <th>App Name</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Reviews</th>
      <th>Installs</th>
      <th>Size</th>
      <th>Price</th>
      <th>Content Rating</th>
      <th>Last Updated</th>
      <th>Minimum Version</th>
      <th>Latest Version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19579</th>
      <td>Real Car Simulator</td>
      <td>GAME_SIMULATION</td>
      <td>3.609745502</td>
      <td>4638.0</td>
      <td>1,000,000+</td>
      <td>49M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>February 5, 2019</td>
      <td>4.1 and up</td>
      <td>20</td>
    </tr>
    <tr>
      <th>17949</th>
      <td>BBVA Wallet Spain. Mobile Payment</td>
      <td>FINANCE</td>
      <td>4.433828831</td>
      <td>21792.0</td>
      <td>1,000,000+</td>
      <td>18M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>March 6, 2019</td>
      <td>4.1 and up</td>
      <td>4.9.190222</td>
    </tr>
    <tr>
      <th>23846</th>
      <td>StandUp Alaoula TV</td>
      <td>ENTERTAINMENT</td>
      <td>4.008111477</td>
      <td>2589.0</td>
      <td>100,000+</td>
      <td>14M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>February 12, 2019</td>
      <td>4.4 and up</td>
      <td>5.0.5</td>
    </tr>
    <tr>
      <th>18431</th>
      <td>Expedia Group Partner Central</td>
      <td>BUSINESS</td>
      <td>3.165771723</td>
      <td>1490.0</td>
      <td>100,000+</td>
      <td>9.9M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>April 3, 2019</td>
      <td>5.0 and up</td>
      <td>1.7.0-751</td>
    </tr>
    <tr>
      <th>1596</th>
      <td>Ocean Block Puzzle</td>
      <td>GAME_PUZZLE</td>
      <td>4.40559721</td>
      <td>4645.0</td>
      <td>500,000+</td>
      <td>12M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>June 12, 2018</td>
      <td>4.0.3 and up</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15811</th>
      <td>iRealtor MY</td>
      <td>LIFESTYLE</td>
      <td>3.882352829</td>
      <td>85.0</td>
      <td>10,000+</td>
      <td>8.7M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>June 5, 2018</td>
      <td>2.3 and up</td>
      <td>1.8.0</td>
    </tr>
    <tr>
      <th>28377</th>
      <td>Glofox</td>
      <td>HEALTH_AND_FITNESS</td>
      <td>3.714640141</td>
      <td>403.0</td>
      <td>50,000+</td>
      <td>11M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>February 18, 2019</td>
      <td>4.1 and up</td>
      <td>8.1.6</td>
    </tr>
    <tr>
      <th>30886</th>
      <td>Drop The Wheel</td>
      <td>GAME_RACING</td>
      <td>3.611111164</td>
      <td>18.0</td>
      <td>5,000+</td>
      <td>34M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>February 21, 2019</td>
      <td>4.1 and up</td>
      <td>1.0.1</td>
    </tr>
    <tr>
      <th>11525</th>
      <td>Sesli Masallar- ?nternetsiz</td>
      <td>ENTERTAINMENT</td>
      <td>4.511210918</td>
      <td>446.0</td>
      <td>100,000+</td>
      <td>76M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>March 13, 2019</td>
      <td>4.1 and up</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>14604</th>
      <td>Tabela Tuss</td>
      <td>HEALTH_AND_FITNESS</td>
      <td>4.255814075</td>
      <td>86.0</td>
      <td>5,000+</td>
      <td>3.2M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>October 31, 2016</td>
      <td>4.0.3 and up</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


## Is there any missing values ?


```python
all_data.isnull().sum().sort_values(ascending=False)
```




    Latest Version     1
    Reviews            1
    Minimum Version    0
    Last Updated       0
    Content Rating     0
    Price              0
    Size               0
    Installs           0
    Rating             0
    Category           0
    App Name           0
    dtype: int64




```python
all_data.loc[all_data.isna().any(axis=1),:]
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
      <th>App Name</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Reviews</th>
      <th>Installs</th>
      <th>Size</th>
      <th>Price</th>
      <th>Content Rating</th>
      <th>Last Updated</th>
      <th>Minimum Version</th>
      <th>Latest Version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6941</th>
      <td>ELer Japanese - NHK News</td>
      <td>Podcasts</td>
      <td>Lessons</td>
      <td>NaN</td>
      <td>EDUCATION</td>
      <td>4.705075264</td>
      <td>1458</td>
      <td>100,000+</td>
      <td>9.5M</td>
      <td>0</td>
      <td>Everyone</td>
    </tr>
    <tr>
      <th>30935</th>
      <td>Baby Bella Caring</td>
      <td>ENTERTAINMENT</td>
      <td>4.18956852</td>
      <td>5560.0</td>
      <td>1,000,000+</td>
      <td>39M</td>
      <td>0</td>
      <td>Everyone 10+</td>
      <td>October 9, 2018</td>
      <td>2.3 and up</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_all_data = all_data.dropna(how='any', axis=0)
```

Dropping the two rows with missing values


```python
new_all_data.isna().sum()
```




    App Name           0
    Category           0
    Rating             0
    Reviews            0
    Installs           0
    Size               0
    Price              0
    Content Rating     0
    Last Updated       0
    Minimum Version    0
    Latest Version     0
    dtype: int64



Nice ! All the observations with missing values have been dropped.

## Is there any duplicates ?


```python
new_all_data.loc[new_all_data.duplicated(keep=False), :]
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
      <th>App Name</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Reviews</th>
      <th>Installs</th>
      <th>Size</th>
      <th>Price</th>
      <th>Content Rating</th>
      <th>Last Updated</th>
      <th>Minimum Version</th>
      <th>Latest Version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23769</th>
      <td>????? Yabila</td>
      <td>LIFESTYLE</td>
      <td>4.3125</td>
      <td>6464.0</td>
      <td>100,000+</td>
      <td>8.3M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>March 18, 2019</td>
      <td>4.1 and up</td>
      <td>3.4</td>
    </tr>
    <tr>
      <th>23770</th>
      <td>????? Yabila</td>
      <td>LIFESTYLE</td>
      <td>4.3125</td>
      <td>6464.0</td>
      <td>100,000+</td>
      <td>8.3M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>March 18, 2019</td>
      <td>4.1 and up</td>
      <td>3.4</td>
    </tr>
  </tbody>
</table>
</div>



There is only one duplicate. Surprisingly, this application has several '?' in its name. Let's dig deeper and see if there are other applications with the same issue.


```python
index_special_char = new_all_data.loc[new_all_data['App Name'].str.contains(pat = r'\?\?+', regex = True),:].index
new_all_data.loc[new_all_data['App Name'].str.contains(pat = r'\?\?+', regex = True),:]
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
      <th>App Name</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Reviews</th>
      <th>Installs</th>
      <th>Size</th>
      <th>Price</th>
      <th>Content Rating</th>
      <th>Last Updated</th>
      <th>Minimum Version</th>
      <th>Latest Version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>???????????????????</td>
      <td>FOOD_AND_DRINK</td>
      <td>4.511622429</td>
      <td>28996.0</td>
      <td>1,000,000+</td>
      <td>Varies with device</td>
      <td>0</td>
      <td>Everyone</td>
      <td>April 2, 2019</td>
      <td>4.1 and up</td>
      <td>6.6.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>DELISH KITCHEN - ??????????????????</td>
      <td>FOOD_AND_DRINK</td>
      <td>4.568304539</td>
      <td>39280.0</td>
      <td>1,000,000+</td>
      <td>7.8M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>March 31, 2019</td>
      <td>4.1 and up</td>
      <td>2.5.7</td>
    </tr>
    <tr>
      <th>33</th>
      <td>?????? - McDonald's Japan</td>
      <td>FOOD_AND_DRINK</td>
      <td>3.439350128</td>
      <td>131608.0</td>
      <td>10,000,000+</td>
      <td>46M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>April 1, 2019</td>
      <td>4.0.3 and up</td>
      <td>4.0.41</td>
    </tr>
    <tr>
      <th>34</th>
      <td>?????? - ????????????????????</td>
      <td>FOOD_AND_DRINK</td>
      <td>4.177659035</td>
      <td>71930.0</td>
      <td>10,000,000+</td>
      <td>17M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>April 3, 2019</td>
      <td>5.0 and up</td>
      <td>19.14.0.7</td>
    </tr>
    <tr>
      <th>58</th>
      <td>?? Super Boy Adventure&amp;Jungle Adventure</td>
      <td>GAME_CASUAL</td>
      <td>4.524364948</td>
      <td>1457.0</td>
      <td>100,000+</td>
      <td>20M</td>
      <td>0</td>
      <td>Teen</td>
      <td>March 9, 2019</td>
      <td>4.1 and up</td>
      <td>1.0.3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>31949</th>
      <td>?????????????APP</td>
      <td>FINANCE</td>
      <td>4.114130497</td>
      <td>8648.0</td>
      <td>1,000,000+</td>
      <td>34M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>April 2, 2019</td>
      <td>5.0 and up</td>
      <td>5.4.0</td>
    </tr>
    <tr>
      <th>31950</th>
      <td>????-??VIP</td>
      <td>FINANCE</td>
      <td>3.576423645</td>
      <td>1001.0</td>
      <td>100,000+</td>
      <td>22M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>February 14, 2019</td>
      <td>4.2 and up</td>
      <td>7.2.1150.2.719</td>
    </tr>
    <tr>
      <th>31951</th>
      <td>???? ????</td>
      <td>FINANCE</td>
      <td>3.182618856</td>
      <td>8630.0</td>
      <td>1,000,000+</td>
      <td>45M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>March 21, 2019</td>
      <td>5.0 and up</td>
      <td>6.6.0.0319</td>
    </tr>
    <tr>
      <th>31952</th>
      <td>????-Phone??</td>
      <td>FINANCE</td>
      <td>3.511627913</td>
      <td>86.0</td>
      <td>10,000+</td>
      <td>20M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>October 22, 2018</td>
      <td>4.0.3 and up</td>
      <td>7.2.1097.TouchFO.2.690.TouchFO4</td>
    </tr>
    <tr>
      <th>31953</th>
      <td>????e???24H????????????</td>
      <td>FINANCE</td>
      <td>3.721518993</td>
      <td>79.0</td>
      <td>10,000+</td>
      <td>22M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>March 19, 2019</td>
      <td>4.4 and up</td>
      <td>2.4.3</td>
    </tr>
  </tbody>
</table>
<p>2539 rows × 11 columns</p>
</div>



The origin of the '?' signs comes mainly from the non-latin alphabet names. Applications are developed everywhere around the globe. For example, the name of the McDonalds App is composed of its name in english and also in the Japanese alphabet causing the appearance of multiple '?' in its App Name.

I decided to drop these apps. **Why ?**

Although they contain valuable information, the fact that I can not trace back the name of the application to an application is an issue. Imagine that a group of interest has multiple of these apps, I would not be able to look for more information on them. I will only be able to say: 'Well... This application uses non-latin alphabet in its name...'</br>

I also choose this option because this project is about data visualization and I do not wish to spend too much time on data cleaning. Alternatively, I could have keep them all or try to disentangle the applications with full non-latin alphabet name and the ones with both (e.g. '?????? - McDonald's Japan')


```python
clean_data = new_all_data.drop(index_special_char, axis = 0)
clean_data.duplicated().sum()
```




    0



The applications with '?' in their name have been removed. There is no duplicate anymore in the dataset.


```python
clean_data.sample(5)
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
      <th>App Name</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Reviews</th>
      <th>Installs</th>
      <th>Size</th>
      <th>Price</th>
      <th>Content Rating</th>
      <th>Last Updated</th>
      <th>Minimum Version</th>
      <th>Latest Version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10316</th>
      <td>Smart Home Surveillance Picket - reuse old phones</td>
      <td>VIDEO_PLAYERS</td>
      <td>4.350564957</td>
      <td>4604.0</td>
      <td>100,000+</td>
      <td>Varies with device</td>
      <td>0</td>
      <td>Everyone</td>
      <td>March 26, 2019</td>
      <td>Varies with device</td>
      <td>Varies with device</td>
    </tr>
    <tr>
      <th>19198</th>
      <td>Daily Words English to Punjabi</td>
      <td>BOOKS_AND_REFERENCE</td>
      <td>4.141304493</td>
      <td>92.0</td>
      <td>50,000+</td>
      <td>2.7M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>September 24, 2018</td>
      <td>4.4 and up</td>
      <td>1.4</td>
    </tr>
    <tr>
      <th>10585</th>
      <td>Magic KWGT</td>
      <td>PERSONALIZATION</td>
      <td>4.444444656</td>
      <td>288.0</td>
      <td>10,000+</td>
      <td>66M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>March 22, 2019</td>
      <td>5.0 and up</td>
      <td>v2019.Mar.13.07</td>
    </tr>
    <tr>
      <th>4685</th>
      <td>Orthopedics by Dr. Apurv Mehra</td>
      <td>EDUCATION</td>
      <td>4.43537426</td>
      <td>147.0</td>
      <td>10,000+</td>
      <td>10M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>February 25, 2019</td>
      <td>4.0.3 and up</td>
      <td>1.0.7</td>
    </tr>
    <tr>
      <th>31033</th>
      <td>ESL Play</td>
      <td>ENTERTAINMENT</td>
      <td>4.274233818</td>
      <td>1207.0</td>
      <td>100,000+</td>
      <td>12M</td>
      <td>0</td>
      <td>Everyone</td>
      <td>March 13, 2019</td>
      <td>4.4 and up</td>
      <td>3.6.0</td>
    </tr>
  </tbody>
</table>
</div>



## Transformation of the columns

### Rating column


```python
# Spotting erroneous observations

clean_data.loc[clean_data['Rating'].str.contains(pat = r'[a-zA-Z]', regex = True),:]
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
      <th>App Name</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Reviews</th>
      <th>Installs</th>
      <th>Size</th>
      <th>Price</th>
      <th>Content Rating</th>
      <th>Last Updated</th>
      <th>Minimum Version</th>
      <th>Latest Version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13504</th>
      <td>Never have I ever 18+</td>
      <td>)</td>
      <td>GAME_STRATEGY</td>
      <td>4.000000</td>
      <td>6</td>
      <td>100+</td>
      <td>2.4M</td>
      <td>$0.99</td>
      <td>Mature 17+</td>
      <td>December 30, 2018</td>
      <td>4.0.3 and up</td>
    </tr>
    <tr>
      <th>23457</th>
      <td>Israel News</td>
      <td>Channel 2 News</td>
      <td>NEWS_AND_MAGAZINES</td>
      <td>3.857799</td>
      <td>11976</td>
      <td>1,000,000+</td>
      <td>Varies with device</td>
      <td>0</td>
      <td>Everyone 10+</td>
      <td>March 16, 2019</td>
      <td>Varies with device</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Dropping erroneous observations

indx_drop_rating = clean_data.loc[clean_data['Rating'].str.contains(pat = r'[A-Z]', regex = True),:].index
clean_data.drop(indx_drop_rating, axis = 0, inplace = True)
```


```python
# Changing the type of Rating from object to float64

clean_data['Rating'] = clean_data['Rating'].astype('float64')
```


```python
clean_data['Rating'].dtype
```




    dtype('float64')



### Price column


```python
# Converting Price to float64

clean_data['Price'] = clean_data['Price'].str.replace('$','')
clean_data['Price'] = clean_data['Price'].astype('float64')
clean_data['Price'].dtype
```




    dtype('float64')



### Install column


```python
# Converting Installs to float64

clean_data['Installs'] = clean_data['Installs'].apply(lambda x: x.replace(',',''))
clean_data['Installs'] = clean_data['Installs'].apply(lambda x: x.replace('+',''))
clean_data['Installs'] = clean_data['Installs'].astype('float64')
```

### Size column


```python
clean_data['Size'].replace('Varies with device', np.nan, inplace = True )
```


```python
clean_data['Size'] = clean_data['Size'].str.replace(',','.')
```


```python
clean_data.Size = ((clean_data.Size.replace(r'[kM]+$', '', regex=True).astype(float)) * clean_data.Size.str.extract(r'[\d\.]+([kM]+)', expand=False).fillna(1).replace(['k','M'], [10**3, 10**6]).astype(int))
```


```python
### The 'Varies with device' is replaced by the mean size of the category of the application.
clean_data['Size'].fillna(clean_data.groupby('Category')['Size'].transform('mean'),inplace = True)
```


```python
clean_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 29457 entries, 0 to 31999
    Data columns (total 11 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   App Name         29457 non-null  object
     1   Category         29457 non-null  object
     2   Rating           29457 non-null  float64
     3   Reviews          29457 non-null  float64
     4   Installs         29457 non-null  float64
     5   Size             29457 non-null  float64
     6   Price            29457 non-null  float64
     7   Content Rating   29457 non-null  object
     8   Last Updated     29457 non-null  object
     9   Minimum Version  29457 non-null  object
     10  Latest Version   29457 non-null  object
    dtypes: float64(5), object(6)
    memory usage: 2.7+ MB


The **Rating, Installs, Size and Price** columns have been converted successfully to numerical type.

### Last updated column


```python
# Converting 'Last Updated' column to datetime

clean_data['Last Updated'] = pd.to_datetime(clean_data['Last Updated'])
```

### Strange observation


```python
# Dropping the Google Camera application from the dataset because erroneous value in the 'Installs' column

index_to_drop = clean_data.loc[clean_data['App Name'] == 'Google Camera', :].index
clean_data.loc[clean_data['App Name'] == 'Google Camera', :]
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
      <th>App Name</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Reviews</th>
      <th>Installs</th>
      <th>Size</th>
      <th>Price</th>
      <th>Content Rating</th>
      <th>Last Updated</th>
      <th>Minimum Version</th>
      <th>Latest Version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3035</th>
      <td>Google Camera</td>
      <td>PHOTOGRAPHY</td>
      <td>3.997701</td>
      <td>401547.0</td>
      <td>1.0</td>
      <td>2.114847e+07</td>
      <td>0.0</td>
      <td>Everyone</td>
      <td>2019-03-26</td>
      <td>Varies with device</td>
      <td>Varies with device</td>
    </tr>
  </tbody>
</table>
</div>




```python
clean_data.drop(axis = 0, index = index_to_drop, inplace = True)
```

## Feature creation

### New column: Free vs Paid


```python
def price_vs_paid(x):
    if x == 0:
        return 'Free'
    else:
        return 'Non-free'
```


```python
clean_data['type'] = clean_data['Price'].apply(lambda x: price_vs_paid(x))
```

# Handling extreme values


```python
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,6))
fig.suptitle('Extreme values in Reviews & Installs', fontsize=18, fontweight = 'regular')
sns.distplot(clean_data['Reviews'], ax = ax1, color = 'brown')
ax1.set_xlabel('Reviews', fontsize = 14)
sns.distplot(clean_data['Installs'], ax = ax2, color = 'brown')
ax2.set_xlabel('Installs', fontsize = 14)
plt.show()
```


![png](/images/Pinarello_files/Pinarello_63_0.png)


The extreme values in the Reviews and Installs column make the vizualisation somewhat complicated for certain relations. Therefore I have decided to build two dataframe which exclude extreme values in the Reviews and Installs column.

**Why not stantardize the data ?**

I did not standardize the data to keep all the interpretability and power of visualization.


```python
installs_desc = clean_data['Installs'].describe()
reviews_desc = clean_data['Reviews'].describe()
extreme = pd.concat([installs_desc, reviews_desc], axis=1, keys=['Installs', 'Reviews'])
extreme
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
      <th>Installs</th>
      <th>Reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.945600e+04</td>
      <td>2.945600e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.408489e+06</td>
      <td>1.053417e+05</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.117709e+07</td>
      <td>1.222478e+06</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000e+04</td>
      <td>1.440000e+02</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000e+05</td>
      <td>1.600500e+03</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000e+06</td>
      <td>1.619225e+04</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000e+09</td>
      <td>8.621429e+07</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_no_extreme_installs = clean_data.loc[(clean_data.Installs < clean_data.Installs.quantile(0.95)),:]
data_no_extreme_reviews = clean_data.loc[(clean_data.Reviews < clean_data.Reviews.quantile(0.95)),:]
```

I decided to keep the observations that are in the 0.95 percentile. The motivation behind it is to vizualise more easily the data. A separate analysis could try to understand what drive the success of the most well-known applications (which will typically be in this 5% upper percentile).  

Remark: as these observations are simply applications that have (extremely) well succeed it would not have been wise to withdraw them in the case of machine learning.

# Exploration Data Analysis

## Category


```python
clean_data.Category.unique()
```




    array(['FOOD_AND_DRINK', 'TRAVEL_AND_LOCAL', 'SHOPPING', 'LIFESTYLE',
           'GAME_ACTION', 'GAME_CASUAL', 'GAME_ROLE_PLAYING', 'GAME_PUZZLE',
           'GAME_RACING', 'GAME_ADVENTURE', 'GAME_ARCADE', 'GAME_STRATEGY',
           'GAME_SPORTS', 'GAME_SIMULATION', 'GAME_MUSIC', 'MUSIC_AND_AUDIO',
           'FINANCE', 'EVENTS', 'ENTERTAINMENT', 'EDUCATION',
           'GAME_EDUCATIONAL', 'BOOKS_AND_REFERENCE', 'NEWS_AND_MAGAZINES',
           'PHOTOGRAPHY', 'VIDEO_PLAYERS', 'GAME_WORD', 'ART_AND_DESIGN',
           'GAME_TRIVIA', 'GAME_BOARD', 'BUSINESS', 'PRODUCTIVITY',
           'COMMUNICATION', 'HEALTH_AND_FITNESS', 'HOUSE_AND_HOME', 'SOCIAL',
           'BEAUTY', 'GAME_CASINO', 'MAPS_AND_NAVIGATION', 'PERSONALIZATION',
           'GAME_CARD', 'TOOLS', 'SPORTS', 'AUTO_AND_VEHICLES',
           'LIBRARIES_AND_DEMO', 'COMICS', 'PARENTING', 'DATING', 'WEATHER',
           'MEDICAL'], dtype=object)



Accross the category in the Google Store, what are the most probable categories for an application developed for Pinarello?

Although in real life, this should be determined in collaboration with the business team of Pinarello. We decided after a thorough meeting with ourself that the following categories were the most promising (**group of interest**):

* Shopping: an application where clients could easily buy there pinarello equipments.
* Game casual: Simple user interface, easy to understand and targeting wide audience. (e.g. puzzle-like games, Angry Birds)
* Game racing: Cycling game racing.
* Sports game: The gamer embodies the manager of a cycling team.
* News and magazines: Application with all the news on sponsored cycling team, new equipments...


```python
category_ascending = clean_data.Category.value_counts(ascending = True).index
```


```python
# set color to category

dict_color = {}

keys = clean_data.groupby(by = 'Category').Rating.mean().sort_values().index

for i in keys:
        dict_color[i] = 'gray'

list_color = ['SHOPPING', 'GAME_CASUAL', 'GAME_RACING', 'GAME_SPORTS', 'NEWS_AND_MAGAZINES']

for i in list_color:
    dict_color[i] = 'brown'

plt.figure(figsize= (16,8))
sns.countplot(x = 'Category', data = clean_data, palette = dict_color, order = category_ascending)
plt.xticks(rotation=90)
plt.show()
```


![png](/images/Pinarello_files/Pinarello_73_0.png)


This countplot can be seen from two perspective. The number of application in a category is representative of the competition in this category but might also be indicative of the demand there is for this type of game.

**Information from graph:**

1. Top 3 of most represented category: Education, Tools and Entertainment.
2. Amongst the group of interest, game casual & news_and_magazines are the two most represented.
3. Few libraries and demo, comics, beauty, events and dating application in the Google Store.


```python
order_boxplot = clean_data.groupby(by = 'Category').Rating.median().sort_values().index
keys = clean_data.groupby(by = 'Category').Rating.mean().index
for i in keys:
        dict_color[i] = 'gray'

list_color = ['SHOPPING', 'GAME_CASUAL', 'GAME_RACING', 'GAME_SPORTS', 'NEWS_AND_MAGAZINES']

for i in list_color:
    dict_color[i] = 'brown'

plt.subplots(figsize = (16,8))
sns.boxplot(x="Category", y="Rating",palette = dict_color, order = order_boxplot,data=clean_data)
plt.ylim((0,5.5))
plt.xticks(rotation=90)
plt.show()
```


![png](/images/Pinarello_files/Pinarello_75_0.png)


**Information from graph:**

1. Ratings accross categories is more are less the same and roughly around 4.5 on 5.
2. Amongst the group of interest, the ranking is (from higher median to lower): Game casual, Shopping, Game Sports, News and Magazines and Game Racing


```python
## Grouping by category using dataset without extreme values

grouped_installs_no_outliers = data_no_extreme_installs.groupby(by = 'Category').Installs.mean().sort_values()
df_grouped_installs_no_outliers = pd.DataFrame(grouped_installs_no_outliers)


## Grouping by category using dataset with extreme values

grouped_installs = clean_data.groupby(by = 'Category').Installs.mean().sort_values()
df_grouped_installs = pd.DataFrame(grouped_installs)
```


```python
## set color to category

dict_color = {}
keys = grouped_installs_no_outliers.index
for i in keys:
        dict_color[i] = 'gray'

interest_color = ['SHOPPING', 'GAME_CASUAL', 'GAME_RACING', 'GAME_SPORTS', 'NEWS_AND_MAGAZINES']

for i in interest_color:
    dict_color[i] = 'brown'

dict_color['COMMUNICATION'] = 'lightsteelblue'

## Set order ascending for both graphs

order_no_extreme = df_grouped_installs_no_outliers
order_all_installs = clean_data.groupby(by = 'Category').Installs.mean().sort_values().index

plt.figure(figsize= (16,16))
fig, (ax1,ax2) = plt.subplots(2,1, figsize = (16,16))
fig.tight_layout(pad = 14)

sns.barplot(x = df_grouped_installs.index, y = 'Installs', data = df_grouped_installs, order = order_all_installs, palette = dict_color, ax = ax1)
ax1.tick_params(labelrotation = 90)
ax1.set_xlabel('')
ax1.set_title('Installs per category, with extreme value')

sns.barplot(x = grouped_installs_no_outliers.index, y = 'Installs', data = order_no_extreme, palette = dict_color, ax = ax2)
plt.title('Installs per category, no extreme value')
plt.xticks(rotation=90)
plt.show()
```


    <Figure size 1152x1152 with 0 Axes>



![png](/images/Pinarello_files/Pinarello_78_1.png)


**Information from graph:**

1. Extreme value (i.e. very successful applications) can have a considerable impact. Look at the ranking change for the 'communication' category.
2. Whether including the extreme values or not, the categories of the group of interest are ranked pretty high.
3. Game casual, Game sports and game racing are among the categories that have the higher mean of installs.


```python
#Set ascending order in Free applications
df_free = data_no_extreme_installs.loc[clean_data['type'] == 'Free', :]
free_cat_installs = df_free.groupby(by = 'Category')['Installs','type'].mean().sort_values(by = 'Installs')
order = free_cat_installs.index
```


```python
plt.figure(figsize= (16,8))
sns.pointplot(x = 'Category', y = 'Installs', hue = 'type', order = order, data = data_no_extreme_installs, palette = ['maroon','gray'], n_boot = 10)
sns.despine()
plt.xticks(rotation=90)
plt.show()
```


![png](/images/Pinarello_files/Pinarello_81_0.png)


1. Unsurprisingly, the average number of installs in each category is higher in free applications than in non-free applications.



```python
free_cat_reviews = df_free.groupby(by = 'Category')['Reviews','type'].mean().sort_values(by = 'Reviews')
order = free_cat_reviews.index
plt.figure(figsize= (16,8))
sns.pointplot(x = 'Category', y = 'Reviews', hue = 'type', order = order, data = data_no_extreme_installs, palette = ['maroon','gray'], n_boot = 10)
sns.despine()
plt.xticks(rotation=90)
plt.show()
```

![png](/images/Pinarello_files/Pinarello_83_1.png)


The difference in the mean reviews between free and non-free applications within category is less straightforward. One hypothesis is that people tend to review more applications for which they have paid than the ones they acquire for free.


```python
df_price_no_extreme = clean_data.loc[clean_data['Price'] < 100, :]
```


```python
clean_data.loc[clean_data['Price'] > 100, :]
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
      <th>App Name</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Reviews</th>
      <th>Installs</th>
      <th>Size</th>
      <th>Price</th>
      <th>Content Rating</th>
      <th>Last Updated</th>
      <th>Minimum Version</th>
      <th>Latest Version</th>
      <th>days_since_update</th>
      <th>len_name</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1154</th>
      <td>Plasma duct - Premium Game</td>
      <td>GAME_PUZZLE</td>
      <td>3.378378</td>
      <td>74.0</td>
      <td>1000.0</td>
      <td>18000000.0</td>
      <td>399.99</td>
      <td>Everyone</td>
      <td>2019-03-04</td>
      <td>4.0.3 and up</td>
      <td>7.7.7</td>
      <td>31.0</td>
      <td>26</td>
      <td>Non-free</td>
    </tr>
    <tr>
      <th>13935</th>
      <td>I AM RICH</td>
      <td>FINANCE</td>
      <td>5.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1600000.0</td>
      <td>299.99</td>
      <td>Everyone</td>
      <td>2019-02-09</td>
      <td>4.0.3 and up</td>
      <td>1</td>
      <td>54.0</td>
      <td>9</td>
      <td>Non-free</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize = (16,8))
sns.stripplot(x="Category", y="Price",color = 'brown', data=df_price_no_extreme, jitter=True)
plt.xticks(rotation=90)
plt.show()
```


![png](/images/Pinarello_files/Pinarello_87_0.png)


Medical applications seems to have several expensive applications.


```python
plt.figure(figsize = (16,8))
sns.barplot(x="Category", y="Price",color = 'brown', data= df_price_no_extreme.loc[df_price_no_extreme['Price'] > 0, :])
plt.xticks(rotation=90)
plt.show()
```


![png](/images/Pinarello_files/Pinarello_89_0.png)



```python
df_price_no_extreme_non_free = df_price_no_extreme.loc[df_price_no_extreme['Price'] > 0, :]
df_price_no_extreme_non_free.Price.mean()
```


    4.1436118059027995


Among the non-free applications, the mean price across category is 4.14$. The categories with the higher average price are medical and libraries and demo followed by finance, business and sports applications

## Installs


```python
plt.figure(figsize = (16,8))
sns.countplot(clean_data['Installs'], color = 'brown')
plt.xticks(rotation = 90)
plt.show()
```


![png](/images/Pinarello_files/Pinarello_93_0.png)



```python
clean_data.Installs.describe()
```




    count    2.945600e+04
    mean     4.408489e+06
    std      6.117709e+07
    min      0.000000e+00
    25%      1.000000e+04
    50%      1.000000e+05
    75%      1.000000e+06
    max      5.000000e+09
    Name: Installs, dtype: float64



The application with the highest number of installs has more than 5 billions. However most applications have fewer number of installs. In fact, 50% of the application have between 10.000 and 100.000 installs.


```python
df_copy = clean_data.copy()

df_copy = df_copy.loc[df_copy.Reviews > 10, :]
df_copy = df_copy.loc[df_copy.Installs > 0, :]

df_copy['Installs'] = np.log10(df_copy['Installs'])
df_copy['Reviews'] = np.log10(df_copy['Reviews'])

sns.lmplot("Reviews", "Installs", data=df_copy)
ax = plt.gca()
_ = ax.set_title('Number of Reviews Vs Number of Downloads (Log scaled)')
```


![png](/images/Pinarello_files/Pinarello_96_0.png)


There is a linear relationship between the number of installs and the number of reviews. This relationship is more easily seen on a log scale for the installs since the number of installs is a lot bigger than the number of reviews. One hypothesis that people install the application before reviewing it but the opposite is not true.

## Price


```python
percentage = clean_data['type'].value_counts()/clean_data['type'].count()
#plt.pie(percentage, explode= (0,0.1), labels=['Free', 'Paid'],
        #autopct='%1.1f%%', shadow=True, startangle=250)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))
plt.title(" Proportion of Free vs Paid applications")
sns.countplot(clean_data['type'], palette = ['brown','gray'], ax = ax1)
#sns.set_style("whitegrid")
sns.despine(left=True)
ax2 =plt.pie(percentage, explode= (0,0.1), labels=['Free', 'Non-free'], autopct='%1.1f%%', colors = ['gray', 'brown'], textprops = {'fontsize': 12}, startangle=250)
plt.xticks(rotation = 90)

plt.show();
```


![png](/images/Pinarello_files/Pinarello_99_0.png)



```python
clean_data['type'].value_counts()
```




    Free        27455
    Non-free     2001
    Name: type, dtype: int64



The Google Play Store is mainly composed of free applications: in the dataset 27455 applications are free compared to 2001 that are non-free. In relative terms, only 6.8% of the applications in the dataset are non-free.


```python
df_price = clean_data.loc[(clean_data.type == 'Non-free'), ['Price','Reviews', 'Installs']]
df_price_no_extreme_reviews = data_no_extreme_reviews.loc[(clean_data.type == 'Non-free'), ['Price','Reviews', 'Installs']]
df_price_no_extreme_installs = data_no_extreme_installs.loc[(clean_data.type == 'Non-free'), ['Price','Reviews', 'Installs']]
```


```python
bins = [0,1,2,3,4,5,10,20,50,100,1000]
price_bins = pd.cut(df_price.Price, bins = bins, labels = ['0+','1-2','2-3','3-4','4-5','5-10','10-20','20-50','50-100','100+'])
plt.figure(figsize = (12,6))
sns.countplot(price_bins, palette = ['brown'])
plt.xticks(rotation = 90)
plt.show()
```


![png](/images/Pinarello_files/Pinarello_103_0.png)


Amongst the non-free applications, more than 50% have a price ranging from 0 to 3 dollar and the price of an application rarely exceeds 20$.

## Content rating


```python
plt.figure(figsize = (12,8))
sns.countplot(clean_data['Content Rating'], palette = 'deep')
plt.xticks(rotation = 90)
plt.show()
```


![png](/images/Pinarello_files/Pinarello_106_0.png)


The majority of applications are in the 'Everyone' content rating. 'Adults only 18+' and 'Unrated' have only 3 observations each.


```python
# Creating a DataFrame without observations from 'Unrated' and 'Adults only 18+' categories

index_to_drop = data_no_extreme_installs.loc[(data_no_extreme_installs['Content Rating'] == 'Unrated') | (data_no_extreme_installs['Content Rating'] == 'Adults only 18+'),:].index
df_no_extreme_installs = data_no_extreme_installs.copy()
df_no_extreme_installs.drop(index_to_drop, inplace = True)
```


```python
plt.figure(figsize = (12,8))
sns.barplot(y = 'Installs', x = 'Content Rating', data = df_no_extreme_installs, palette = 'deep')
plt.xticks(rotation = 90)
plt.show()
```


![png](/images/Pinarello_files/Pinarello_109_0.png)


When the extreme value are not taken into account, the category 'Everyone 10+' has the higher average number of installs.


```python
plt.figure(figsize = (16,8))
sns.boxplot(y = clean_data['Rating'], x = clean_data['Content Rating'], palette = 'deep')
plt.xticks(rotation = 90)
plt.show()
```


![png](/images/Pinarello_files/Pinarello_111_0.png)


There is no differences in the distribution of ratings for the content ratings.

# Conclusion

Through this exploration data analysis, we strive to summarize the main characteristics of the Google Play Store.  
