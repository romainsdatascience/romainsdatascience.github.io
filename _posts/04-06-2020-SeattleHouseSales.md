---
title: "Seattle House Sales Predictions"
date: 2020-04-10
tags: [Classification]
header:
excerpt: "Regression on the House Sales in King County dataset"
classes: wide
---

# Prediction King County houses prices

This project aims at predicting the prices of the houses located in King County, Washington.

# Importing modules


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
```


```python
full_dataset = pd.read_csv('kc_house_data.csv')
```

# Data description

Few information is given on the dataset and its features. All we know is that it contains house sale prices for King County and that it includes houses sold between May 2014 and May 2015. However, by looking in the discussions of the dataset we can find additional variable explanations:

1. "id": Unique ID for each home sold.
2. "date": Date of the sale.
3. "price": Price of each house sold.
4. "bedrooms": Number of bedrooms.
5. "bathrooms": Number of bathrooms, some decimals appears because of the way the number of bathrooms is computed. For example, a room with a toilet but no shower will account for 0.5.
6. "sqft_living": Square footage of the living space.
7. "sqft_lot": Square footage of the land space.
8. "floors": Number of floors.
9. "waterfront": A dummy variable taking 1 if the house is near a body of water, else equals 0.
10. "view": An index from 0 to 4 of how good the view of the property is.
11. "condition": An index from 1 to 5 on the condition of the house.(See appendices)
12. "grade": An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design. (See appendices)
13. "sqft_above": The square footage of the interior housing space that is above ground level.
14. "sqft_basement": The square footage of the interior housing space that is below ground level.
15. "yr_built": The year the house was initially built.
16. "yr_renovated": The year of the house’s last renovation.
17. "zipcode": zipcode of the house.
18. "lat": Lattitude.
19. "long": Longitude.
20. "sqft_living15": The square footage of interior housing living space for the nearest 15 neighbors.
21. "sqft_lot15": The square footage of the land lots of the nearest 15 neighbors.

Remark: Although we can not trust these variables at a hundred percent, it is the best information I could gather so we will go on with these.

<a href="https://www.kaggle.com/harlfoxem/housesalesprediction"></a>
<a href="https://rstudio-pubs-static.s3.amazonaws.com/155304_cc51f448116744069664b35e7762999f.html"></a>
<a href="https://info.kingcounty.gov/assessor/esales/Glossary.aspx?type=r"></a>

# Data profiling


```python
full_dataset.shape
```




    (21613, 21)



The dataset has 21613 instances and 21 features.


```python
full_dataset.sample(5)
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
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>...</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>541</th>
      <td>2270000070</td>
      <td>20141030T000000</td>
      <td>280000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>1560</td>
      <td>4350</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1560</td>
      <td>0</td>
      <td>2003</td>
      <td>0</td>
      <td>98056</td>
      <td>47.5025</td>
      <td>-122.186</td>
      <td>1560</td>
      <td>4350</td>
    </tr>
    <tr>
      <th>15398</th>
      <td>8946750170</td>
      <td>20150421T000000</td>
      <td>281000.0</td>
      <td>4</td>
      <td>2.25</td>
      <td>1677</td>
      <td>3600</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1677</td>
      <td>0</td>
      <td>2012</td>
      <td>0</td>
      <td>98092</td>
      <td>47.3200</td>
      <td>-122.178</td>
      <td>1677</td>
      <td>3600</td>
    </tr>
    <tr>
      <th>15023</th>
      <td>1253200170</td>
      <td>20140520T000000</td>
      <td>250000.0</td>
      <td>4</td>
      <td>1.50</td>
      <td>2500</td>
      <td>6300</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1500</td>
      <td>1000</td>
      <td>1961</td>
      <td>0</td>
      <td>98032</td>
      <td>47.3781</td>
      <td>-122.284</td>
      <td>1720</td>
      <td>8925</td>
    </tr>
    <tr>
      <th>6435</th>
      <td>8832900780</td>
      <td>20150408T000000</td>
      <td>647500.0</td>
      <td>5</td>
      <td>2.00</td>
      <td>1760</td>
      <td>21562</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>8</td>
      <td>1560</td>
      <td>200</td>
      <td>1959</td>
      <td>0</td>
      <td>98028</td>
      <td>47.7597</td>
      <td>-122.263</td>
      <td>2150</td>
      <td>12676</td>
    </tr>
    <tr>
      <th>5130</th>
      <td>3327020290</td>
      <td>20150220T000000</td>
      <td>300000.0</td>
      <td>4</td>
      <td>1.75</td>
      <td>2200</td>
      <td>7600</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>2200</td>
      <td>0</td>
      <td>1978</td>
      <td>0</td>
      <td>98092</td>
      <td>47.3131</td>
      <td>-122.191</td>
      <td>1910</td>
      <td>7600</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



The "date", "yr_built" and "yr_renovated" features are not in an appropriate format. It is impotant to notice that "yr_renovated" has a value of zero when the house has never been renovated.


```python
full_dataset.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21613 entries, 0 to 21612
    Data columns (total 21 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   id             21613 non-null  int64  
     1   date           21613 non-null  object
     2   price          21613 non-null  float64
     3   bedrooms       21613 non-null  int64  
     4   bathrooms      21613 non-null  float64
     5   sqft_living    21613 non-null  int64  
     6   sqft_lot       21613 non-null  int64  
     7   floors         21613 non-null  float64
     8   waterfront     21613 non-null  int64  
     9   view           21613 non-null  int64  
     10  condition      21613 non-null  int64  
     11  grade          21613 non-null  int64  
     12  sqft_above     21613 non-null  int64  
     13  sqft_basement  21613 non-null  int64  
     14  yr_built       21613 non-null  int64  
     15  yr_renovated   21613 non-null  int64  
     16  zipcode        21613 non-null  int64  
     17  lat            21613 non-null  float64
     18  long           21613 non-null  float64
     19  sqft_living15  21613 non-null  int64  
     20  sqft_lot15     21613 non-null  int64  
    dtypes: float64(5), int64(15), object(1)
    memory usage: 3.5+ MB


There is no missing value in the dataset. All features are numerical except for the "date" column.


```python
full_dataset.describe()
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
      <th>id</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.161300e+04</td>
      <td>2.161300e+04</td>
      <td>21613.000000</td>
      <td>21613.000000</td>
      <td>21613.000000</td>
      <td>2.161300e+04</td>
      <td>21613.000000</td>
      <td>21613.000000</td>
      <td>21613.000000</td>
      <td>21613.000000</td>
      <td>21613.000000</td>
      <td>21613.000000</td>
      <td>21613.000000</td>
      <td>21613.000000</td>
      <td>21613.000000</td>
      <td>21613.000000</td>
      <td>21613.000000</td>
      <td>21613.000000</td>
      <td>21613.000000</td>
      <td>21613.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.580302e+09</td>
      <td>5.400881e+05</td>
      <td>3.370842</td>
      <td>2.114757</td>
      <td>2079.899736</td>
      <td>1.510697e+04</td>
      <td>1.494309</td>
      <td>0.007542</td>
      <td>0.234303</td>
      <td>3.409430</td>
      <td>7.656873</td>
      <td>1788.390691</td>
      <td>291.509045</td>
      <td>1971.005136</td>
      <td>84.402258</td>
      <td>98077.939805</td>
      <td>47.560053</td>
      <td>-122.213896</td>
      <td>1986.552492</td>
      <td>12768.455652</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.876566e+09</td>
      <td>3.671272e+05</td>
      <td>0.930062</td>
      <td>0.770163</td>
      <td>918.440897</td>
      <td>4.142051e+04</td>
      <td>0.539989</td>
      <td>0.086517</td>
      <td>0.766318</td>
      <td>0.650743</td>
      <td>1.175459</td>
      <td>828.090978</td>
      <td>442.575043</td>
      <td>29.373411</td>
      <td>401.679240</td>
      <td>53.505026</td>
      <td>0.138564</td>
      <td>0.140828</td>
      <td>685.391304</td>
      <td>27304.179631</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000102e+06</td>
      <td>7.500000e+04</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>290.000000</td>
      <td>5.200000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>290.000000</td>
      <td>0.000000</td>
      <td>1900.000000</td>
      <td>0.000000</td>
      <td>98001.000000</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>399.000000</td>
      <td>651.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.123049e+09</td>
      <td>3.219500e+05</td>
      <td>3.000000</td>
      <td>1.750000</td>
      <td>1427.000000</td>
      <td>5.040000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1190.000000</td>
      <td>0.000000</td>
      <td>1951.000000</td>
      <td>0.000000</td>
      <td>98033.000000</td>
      <td>47.471000</td>
      <td>-122.328000</td>
      <td>1490.000000</td>
      <td>5100.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.904930e+09</td>
      <td>4.500000e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1910.000000</td>
      <td>7.618000e+03</td>
      <td>1.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1560.000000</td>
      <td>0.000000</td>
      <td>1975.000000</td>
      <td>0.000000</td>
      <td>98065.000000</td>
      <td>47.571800</td>
      <td>-122.230000</td>
      <td>1840.000000</td>
      <td>7620.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.308900e+09</td>
      <td>6.450000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2550.000000</td>
      <td>1.068800e+04</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>2210.000000</td>
      <td>560.000000</td>
      <td>1997.000000</td>
      <td>0.000000</td>
      <td>98118.000000</td>
      <td>47.678000</td>
      <td>-122.125000</td>
      <td>2360.000000</td>
      <td>10083.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.900000e+09</td>
      <td>7.700000e+06</td>
      <td>33.000000</td>
      <td>8.000000</td>
      <td>13540.000000</td>
      <td>1.651359e+06</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>13.000000</td>
      <td>9410.000000</td>
      <td>4820.000000</td>
      <td>2015.000000</td>
      <td>2015.000000</td>
      <td>98199.000000</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>6210.000000</td>
      <td>871200.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
histo_data = full_dataset.drop(labels = ['id', 'lat', 'long', 'zipcode'], axis = 1)
histo_data.hist(bins = 50,figsize=(20,20))
plt.show()
```


![png](/images/SeattleHouseSalesPredictions_files/SeattleHouseSalesPredictions_15_0.png)


Here are some noticeable elements that caught my eye during this quick overview:

1. The feature 'id' does not convey any information. We can remove it.

2. The feature 'date' and 'yr_built' should be transform to the datetime format.

3. 'yr_renovated' contains valuable information but can be transformed to be enhance interpretability.

4. I suspect some of the square feet features to be correlated together. If correlation is high, it might be judicious to drop some of them.

## Erroneous instance

While searching for additional information over the dataset, I stumble upon discussions over a particular instance. One house has 33 bedrooms for a square footage living of 1620. Even if we consider that the house is only composed of bedrooms, it gives us an average square footage of 49 sqft per bedroom. The equivalent in square meter is roughly 4.5 m^2 per bedroom. This number is highly unlikely and probably constitutes an error. Therefore, I decide to delete this instance.


```python
full_dataset.loc[full_dataset["bedrooms"] == 33, :]
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
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>...</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15870</th>
      <td>2402100895</td>
      <td>20140625T000000</td>
      <td>640000.0</td>
      <td>33</td>
      <td>1.75</td>
      <td>1620</td>
      <td>6000</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1040</td>
      <td>580</td>
      <td>1947</td>
      <td>0</td>
      <td>98103</td>
      <td>47.6878</td>
      <td>-122.331</td>
      <td>1330</td>
      <td>4700</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>




```python
copy_data = full_dataset.copy()
copy_data.drop(index = 15870, inplace = True)
```

## Geographical exploration

In the section, we investigate if the localisation of a house influence its price. Fortunately, the dataset contains the latitude and the longitude for each house sold. All the houses are represented on the map of the King County here below. Additionally, the houses have been "colored" according to the median price of the houses in their zipcode area.


```python
# Copy the original dataset
geo_data = copy_data.copy()

# Create new column containing the median price per zipcode for each instance
zipcode_grouped = geo_data.groupby("zipcode")["price"].median().sort_values(ascending = False)
mapping = zipcode_grouped.to_dict()
geo_data["median_price_zipcode"] = geo_data["zipcode"].map(mapping)
```


```python
# Plot all the houses
ax = geo_data.plot(kind="scatter", x="long", y="lat", figsize=(15,15),
                       c="median_price_zipcode", cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.1)

# King County map in the background
king_county_img= mpimg.imread('kingcountyseattle.png') 
plt.imshow(king_county_img, alpha=0.5, extent=[-122.5717,-121.058509,47.045649,47.802443],
           cmap=plt.get_cmap("jet"))

# Add title, labels, colorbar
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
plt.title("Median House Price per zipcode", fontsize = 18)

prices = geo_data["median_price_zipcode"]
tick_values = np.linspace(prices.min(), prices.max(), 11)

cbar = plt.colorbar(fraction=0.025, pad=0.04)
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=12)
cbar.set_label('Median House Value per zipcode', fontsize=12)

plt.show()
```


![png](/images/SeattleHouseSalesPredictions_files/SeattleHouseSalesPredictions_24_0.png)


We discover that the median price per zipcode varies greatly following its localisation:  

- In the south of the county, median price is around 200.000 dollars.
- The median prices tend to increase as we go north.
- The median prices skyrocket around the city of Bellevue. The median price for Medina is 1.892.500 dollars (the area represented in red).

The patterns highlighted in this map does not necessarily means that localisation influences the price of the houses of the dataset. Who knows ? Maybe houses in Medina are particularly big, with saunas and swimming pools while in the north of the county people are living happily in small cottages... Initially, it was meant to be a joke but after spending a bit of time on Google Street View, I can say that it has some truth in it.

All this being said, all the variations in price can not be explained by the house in itself and based on what we know of real estate, we can infer that the localisation influences the price of a house in the King County.

## Splitting the dataset

I took a quick glance at the full dataset but now it's time to split the dataset into train and test sets and to not look anymore at the test set until I test the final model. This way I am sure to not unconsciously overfit the test set.


```python
copy_data_shuffled = shuffle(copy_data)
train_set, test_set = train_test_split(copy_data_shuffled, test_size = 0.2)
```


```python
print(train_set.shape)
print(test_set.shape)
```

    (17289, 21)
    (4323, 21)


## Exploration data analysis

Now that we have splitted the dataset, we are free to perform an EDA on the train set. 1... 2... 3... LET'S GO !


```python
sns.set(style="whitegrid", font_scale=1, palette = "muted")
```


```python
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(16,12))
sns.boxplot(x=train_set['bedrooms'],y=train_set['price'], ax=ax1)
sns.boxplot(x=train_set['floors'],y=train_set['price'], ax=ax2)
sns.boxplot(x=train_set['waterfront'],y=train_set['price'], width = 0.4, ax=ax3)
sns.boxplot(x=train_set['view'],y=train_set['price'], ax=ax4)
sns.despine(left=True, bottom=True)

ax1.set(xlabel='Number of bedrooms', ylabel='Price')
ax1.yaxis.tick_left()
ax1.set_title("Bedrooms", fontsize = 14)

ax2.set_title("Floors", fontsize = 14)
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set(xlabel='Number of floors', ylabel='Price')

ax3.set(xlabel='Waterfront', ylabel='Price')
ax3.yaxis.tick_left()
ax3.set_title("Waterfront", fontsize = 14)

ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()
ax4.set(xlabel='View rating', ylabel='Price')
ax4.set_title("View", fontsize = 14)

f.suptitle('Relations with the target variable', fontsize = 20, fontweight = "semibold")
plt.tight_layout(pad = 4)
plt.show()
```


![png](/images/SeattleHouseSalesPredictions_files/SeattleHouseSalesPredictions_33_0.png)


1. The number of bedrooms increases with the price of a house.
2. The price of a house goes up with the number of floors until 2.5 floors. Houses with 3 floors have a lower price on average than the ones with 2 floors.
3. Houses with waterfront have a higher price than those without.
4. The better the view rating, the higher the price.


```python
f, ((ax1), (ax2)) = plt.subplots(2, 1,figsize=(14,12))
sns.boxplot(x=train_set['grade'],y=train_set['price'], ax=ax1)
sns.boxplot(x=train_set['condition'],y=train_set['price'], ax=ax2, width = 0.6)
sns.despine(left=True, bottom=True)
ax1.set(xlabel='Grade', ylabel='Price')
ax1.set_title("Building grade", fontsize = 14)
ax2.set(xlabel='Grade', ylabel='Price')
ax2.set_title("Condition", fontsize = 14)

f.suptitle('Relations with the target variable', fontsize = 20, fontweight = "semibold")
plt.tight_layout(pad = 4)
plt.show()
```


![png](/images/SeattleHouseSalesPredictions_files/SeattleHouseSalesPredictions_35_0.png)


The features "price" and "grade" are positively correlated. The effect does not seem to be linear. Between 3 and 8 the effect of an additional grade point is minor while the average price jumps from the 11 to 12 grade (or from 12 to 13 grade). On the contrary, the average price of houses does not increase with the condition grade. This apparent effect is misleading. Indeed, the condition grades are "relative to age and grade" (see appendices) and thus, their effects might be hidden when displayed as we did above.


```python
f, ((ax1, ax2)) = plt.subplots(1, 2,figsize=(15,6))
grades_7 = train_set.loc[train_set['grade'] == 7, :]
sns.boxplot(x=grades_7['condition'],y=grades_7['price'], width = 0.6, ax=ax1)

ax1.set(xlabel='Condition grade', ylabel='Price')
ax1.yaxis.tick_left()
ax1.set_title("Building grade equals 7", fontsize = 14)


grades_11 = train_set.loc[train_set['grade'] == 11, :]
sns.boxplot(x=grades_11['condition'],y=grades_11['price'], width = 0.6, ax=ax2)

ax2.set_title("Building grade equals 11", fontsize = 14)
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set(xlabel='Condition grade', ylabel='Price')
plt.show()
```


![png](/images/SeattleHouseSalesPredictions_files/SeattleHouseSalesPredictions_37_0.png)


If we look at one building grade at a time, a positive effect appears. To enhance performance it might be judicious to include interaction terms in our models.


```python
df_pairplot = train_set.loc[:,['sqft_living', 'sqft_lot','sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']]
sns.pairplot(df_pairplot)
plt.show()
```


![png](/images/SeattleHouseSalesPredictions_files/SeattleHouseSalesPredictions_39_0.png)


In the graphs above, one graph particularly draws my attention. The scatterplot between 'sqft_living' and 'sqft_above' where the points draw a diagonal line in the graph.


```python
sns.jointplot(x = train_set['sqft_living'], y = train_set['sqft_above'])
plt.ylim((0,14000))
plt.show()
```


![png](/images/SeattleHouseSalesPredictions_files/SeattleHouseSalesPredictions_41_0.png)


This line underlines the fact that 'sqft_living' and 'sqft_above' are linked 'by definition'. The diagonal line represent the fact that 'sqft_above' can not be superior than 'sqft_living'. We will take care of this in the Data cleaning section.

# Data cleaning

In this section, we do the necessary cleaning for the various issues we have highlighted in 'Data profiling', 'geographical exploration' and 'Exploration data analysis'.

## Remove the 'id' feature


```python
def remove_id(dataset):
    clean = dataset.drop(labels = 'id', axis = 1)
    return clean

train_set = remove_id(train_set)
```

## Transform 'date' and 'yr_built' to datetime format


```python
def to_date(dataset):
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset['yr_built'] = pd.to_datetime(dataset['yr_built'], format = '%Y')

to_date(train_set)
```

## Feature engineering: new feature based on 'yr_renovated'


```python
def days_since_renovation(row):

    if row['yr_renovated'] == 0:
        result = row.date - row['yr_built']

    if row['yr_renovated'] != 0:
        result = row.date - pd.to_datetime(row['yr_renovated'], format = "%Y")

    return result.days
```


```python
train_set['days_since_renovation'] = train_set.apply(days_since_renovation, axis = 1)
```

Some of the instances have a negative value for the newly created feature 'days_since_renovation'. Let's take a closer look:


```python
train_set.loc[train_set['days_since_renovation'] < 0, ['date', 'days_since_renovation', 'yr_built', 'yr_renovated']]
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
      <th>date</th>
      <th>days_since_renovation</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7097</th>
      <td>2014-10-28</td>
      <td>-65</td>
      <td>1940-01-01</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>14859</th>
      <td>2014-06-06</td>
      <td>-209</td>
      <td>1956-01-01</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>2687</th>
      <td>2014-10-29</td>
      <td>-64</td>
      <td>2015-01-01</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11599</th>
      <td>2014-05-22</td>
      <td>-224</td>
      <td>1923-01-01</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>20770</th>
      <td>2014-08-28</td>
      <td>-126</td>
      <td>2015-01-01</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8039</th>
      <td>2014-06-24</td>
      <td>-191</td>
      <td>2015-01-01</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7526</th>
      <td>2014-12-31</td>
      <td>-1</td>
      <td>2015-01-01</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21262</th>
      <td>2014-11-25</td>
      <td>-37</td>
      <td>2015-01-01</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17098</th>
      <td>2014-06-17</td>
      <td>-198</td>
      <td>2015-01-01</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14489</th>
      <td>2014-08-26</td>
      <td>-128</td>
      <td>2015-01-01</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20852</th>
      <td>2014-07-09</td>
      <td>-176</td>
      <td>2015-01-01</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2295</th>
      <td>2014-07-28</td>
      <td>-157</td>
      <td>1922-01-01</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>1763</th>
      <td>2014-06-25</td>
      <td>-190</td>
      <td>2015-01-01</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15687</th>
      <td>2014-10-06</td>
      <td>-87</td>
      <td>1955-01-01</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>21372</th>
      <td>2014-05-20</td>
      <td>-226</td>
      <td>2015-01-01</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18575</th>
      <td>2014-07-01</td>
      <td>-184</td>
      <td>1945-01-01</td>
      <td>2015</td>
    </tr>
  </tbody>
</table>
</div>



The negative values are due to the fact that the houses are sold before the year they are built or the year of their renovation. When we think of it... It is not that strange. It is not unusual to buy a house while it is still under construction. Since it does not consist of an error, I determine 0 as the lower threshold. A value of zero in the "days_since_renovation" means that the house was sold before the construction or the renovation was finished. In other words, 0 days elapse between the renovation/construction of the house and the sale.


```python
def clip_and_drop(dataset):
    dataset['days_since_renovation'].clip(lower = 0, inplace = True)
    dataset.drop('yr_renovated', inplace = True, axis = 1)

clip_and_drop(train_set)
```


```python
plt.figure(figsize = (14,8))
bins = pd.cut(train_set['days_since_renovation'], [-1,100, 500, 1000, 5000, 10000, 20000, np.inf])
sns.boxplot(x=bins,y=train_set['price'])
sns.despine(left=True, bottom=True)
plt.show()
```


![png](/images/SeattleHouseSalesPredictions_files/SeattleHouseSalesPredictions_56_0.png)


## Inspecting correlation between the features


```python
# for more clarity 'date', 'lat', 'long' and 'zipcode' are not included in the features.

features = ['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront',
            'view','condition','grade','sqft_above','sqft_basement','yr_built',
            'sqft_living15','sqft_lot15', 'days_since_renovation']

mask = np.zeros_like(train_set[features].corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(train_set[features].corr(),linewidths=0.25,vmax=0.9,square=True,cmap="PuBuGn",
            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});
```


![png](/images/SeattleHouseSalesPredictions_files/SeattleHouseSalesPredictions_58_0.png)


The most correlated features are 'sqft_above' and 'sqft_living'. For this and the reasons I mentionned earlier, I decided to remove the 'sqft_above' feature.


```python
def remove_corr(dataset):
    clean = dataset.drop(axis = 1, labels = 'sqft_above')
    return clean

train_set = remove_corr(train_set)
```

## Feature engineering : geolocalisation

The goal here is to create a feature that captures the information about the localisation of the houses. Three different ways to implement it come to my mind:

1. Encode the zipcode feature as a one-hot numeric array with OneHotEncoder(). Problem: there are 70 zipcodes resulting in the creation of 70 additional features.

2. In collaboration with someone who knows well the geography of King County (e.g. an expert), divide the territory into different relevant areas. These areas could reflect economic, geographic, social differences between areas. Then, we can map each zipcode to its correspondant area.

3. Use a clustering learning techniques to divide the territory of the King County.

I decided to go on with the third option with the K-means algorithm. K-means is not ideal for geographical data because of the way the distances are computed. It does not take into account the spherical component of the earth. Moreover, the 15 clusters will minimize the squared distance between the sample and their closest cluster but it does not mean that the houses close to the same cluster are indeed similar in terms of geography, neighborhood, proximity to schools, city centers, etc. So why use K-means then ??? Because I want to see what clustering using k-means algorithm can bring in these settings. However, I personally think that the second option is the most promising. First of all, by studying the geography of King County, we could discover other drivers of price. Secondly, a segmentation done by humans is generally more interpretable than one resulting from clustering. For example, if the expert divide the territory into urban, center business district, periphery, rural areas etc. one quickly understands their characteristics since it based on some common knowledge.


```python
geo = train_set.loc[:, ['long', 'lat']]
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(geo)
                for k in range(6, 26)]
inertias = [model.inertia_ for model in kmeans_per_k]
cluster_centers = [model.cluster_centers_ for model in kmeans_per_k]
```


```python
plt.figure(figsize=(12, 6))
plt.plot(range(6, 26), inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.show()
```


![png](/images/SeattleHouseSalesPredictions_files/SeattleHouseSalesPredictions_64_0.png)


To determine the "best" number of clusters, one generally searches fo an elbow on the above graph. As the number of clusters increase, the sum of squared distances of instances to their closest cluster center naturally decreases. Thus, the inertia also decreases. The science here is to locate the number of clusters where the inertia decreases not only due to the increased number of clusters but because of attributes specific to the data. This point can be found by looking where the inertia decreases sharply. In our graph, it is not clear if such a point exists. At the risk of being castigated, I decide (quite arbitrarely) to go with 15 clusters. The 15 cluster centers are represented on the map below.


```python
def localisation_15(df):
    dataset = df.reset_index(drop = True).copy()
    model = KMeans(n_clusters=15).fit(dataset.loc[:,['long', 'lat']])
    labels = model.labels_
    labels = labels.reshape(-1,1)
    encoder = OneHotEncoder()
    onehot = encoder.fit_transform(labels).toarray()
    labels_ohe = ["area_{}".format(x) for x in range(1,16)]
    df_onehot = pd.DataFrame(onehot, columns=labels_ohe)

    return pd.concat([dataset,df_onehot], axis = 1), labels_ohe, model.cluster_centers_

train_set, labels_list, centers_list = localisation_15(train_set)
```


```python
latitude = [x[0] for x in centers_list]
longitude = [x[1] for x in centers_list]
```


```python
king_county_img= mpimg.imread('kingcountyseattle.png')
plt.figure(figsize = (15,15))
plt.scatter(x= latitude, y = longitude,  marker = "x")
plt.imshow(king_county_img, alpha=0.5, extent=[-122.5717,-121.058509,47.045649,47.802443],
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
plt.show()
```


![png](/images/SeattleHouseSalesPredictions_files/SeattleHouseSalesPredictions_68_0.png)


# Predict houses prices

The second part of the project is dedicated to the prediction of houses price. For this project, I want to investigate if an ensemble method can help me achieve better results. In a first step, we quickly try multiple algorithms and then narrow down the number of algorithms to the most promising ones. Afterwards, we tune their hyperparameters to improve performance.

## Import modules


```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.spatial import distance
```

## Preparing the training data


```python
# Create the train set and the test set
features_to_drop = ['price', 'date', 'yr_built', 'zipcode', 'lat', 'long']
X_train_set, y_train_set = train_set.drop(axis = 1, labels = features_to_drop), train_set["price"]
```


```python
def scale_num(X_data):
    data = X_data.copy()
    scaler = StandardScaler()
    col_names = ['sqft_living', 'sqft_lot', 'sqft_basement', 'sqft_living15', 'sqft_lot15', 'days_since_renovation']
    features = data[col_names]
    features_scaled = scaler.fit_transform(features.values)
    X_data.loc[: , col_names] = features_scaled
    return X_data

X_train_scaled = scale_num(X_train_set)
```

## Models

### Basic linear regression


```python
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train_set)
scores = cross_validate(lin_reg, X_train_scaled, y_train_set, scoring = 'r2', cv=5, return_train_score =True)
scores
```




    {'fit_time': array([0.01083684, 0.01349401, 0.01051283, 0.010221  , 0.01029611]),
     'score_time': array([0.00189018, 0.00177479, 0.00171304, 0.00153995, 0.00154495]),
     'test_score': array([0.72240129, 0.72675668, 0.73382313, 0.72695897, 0.72430756]),
     'train_score': array([0.73218155, 0.73153311, 0.7301965 , 0.731401  , 0.73212742])}



### Ridge


```python
ridge_reg = Ridge()
scores = cross_validate(ridge_reg, X_train_scaled, y_train_set, scoring = 'r2', cv=3, return_train_score =True)
scores
```




    {'fit_time': array([0.01782179, 0.00735879, 0.00735497]),
     'score_time': array([0.0038259 , 0.00173807, 0.00173593]),
     'test_score': array([0.73187571, 0.71531714, 0.73394579]),
     'train_score': array([0.72959374, 0.73762554, 0.7291468 ])}



### Elastic Net


```python
elastic_reg = ElasticNet()
scores = cross_validate(elastic_reg, X_train_scaled, y_train_set, scoring = 'r2', cv=3, return_train_score =True)
```


```python
scores
```




    {'fit_time': array([0.03390503, 0.01691079, 0.01995683]),
     'score_time': array([0.00270176, 0.00211096, 0.00177002]),
     'test_score': array([0.62208609, 0.58743309, 0.61866903]),
     'train_score': array([0.60870649, 0.61565025, 0.60789825])}



### Lasso


```python
lasso_reg = Lasso()
scores = cross_validate(lasso_reg, X_train_scaled, y_train_set, cv=3, return_train_score =True)
scores
```


    {'fit_time': array([0.58281016, 0.66304898, 0.61118007]),
     'score_time': array([0.00257683, 0.00631189, 0.00393987]),
     'test_score': array([0.73185496, 0.71531818, 0.7339361 ]),
     'train_score': array([0.72959715, 0.73762934, 0.7291506 ])}



### Polynomial


```python
poly = PolynomialFeatures(degree=2)
cols_to_trans = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_basement', 'sqft_living15', 'sqft_lot15']
X_ = poly.fit_transform(X_train_scaled[cols_to_trans])
poly_df = pd.DataFrame(X_)
X_train_poly = pd.concat([X_train_scaled,poly_df], axis = 1)

poly_reg = LinearRegression()
scores = cross_validate(poly_reg,X_train_poly, y_train_set, scoring = 'r2',cv=3, return_train_score =True)
scores
```




    {'fit_time': array([0.08575606, 0.05306506, 0.076087  ]),
     'score_time': array([0.00436282, 0.004287  , 0.00723577]),
     'test_score': array([0.77724503, 0.78180868, 0.77556245]),
     'train_score': array([0.80694129, 0.79526656, 0.80548136])}



### Random Forest


```python
randomforest_reg = RandomForestRegressor()
scores = cross_validate(randomforest_reg, X_train_scaled, y_train_set, cv=3, return_train_score =True)
scores
```




    {'fit_time': array([8.07128501, 7.71711493, 7.75728583]),
     'score_time': array([0.14071512, 0.14184213, 0.13763094]),
     'test_score': array([0.8156137 , 0.78697385, 0.79969389]),
     'train_score': array([0.97378638, 0.97385534, 0.9748951 ])}



### Support Vector Regression


```python
poly_SVM = SVR(kernel="poly", degree=3, C=100, epsilon=0.1)
scores = cross_validate(poly_SVM, X_train_scaled, y_train_set, cv=3, return_train_score =True)
scores
```




    {'fit_time': array([6.25891995, 6.13603878, 5.9624989 ]),
     'score_time': array([1.77045918, 1.75651312, 1.72181392]),
     'test_score': array([0.4225272 , 0.42774886, 0.42849825]),
     'train_score': array([0.43137231, 0.42628376, 0.42265582])}




```python
lin_SVR = LinearSVR()
scores = cross_validate(lin_SVR, X_train_scaled, y_train_set, cv=3, return_train_score =True)
scores
```




    {'fit_time': array([0.0237236 , 0.01543999, 0.01580691]),
     'score_time': array([0.00204539, 0.00193691, 0.0019331 ]),
     'test_score': array([0.01821899, 0.02699499, 0.03113129]),
     'train_score': array([0.02859899, 0.02322034, 0.02520966])}




```python
poly_SVM = SVR(kernel="rbf", C=30000, epsilon=0.1)
scores = cross_validate(poly_SVM, X_train_scaled, y_train_set, cv=3, return_train_score =True)
scores
```




    {'fit_time': array([8.12917686, 7.94500995, 7.70576501]),
     'score_time': array([2.67584991, 2.57315922, 2.54428291]),
     'test_score': array([0.71090087, 0.63174621, 0.69734663]),
     'train_score': array([0.66610235, 0.70209793, 0.67600254])}



## The most promising models

Apart from their performances, the "diversity" of the models was important goal for chosing the models. The more different they are, the more likely they are to complete each other (Damn... That's deep !). The averaging ensemble method will work best if their errors are not correlated. Eventually, I have chosen four models to use:

1. Random Forest regression.
2. Ridge regression: having a reguralized model can help decorrelating the errors. It is performing better than Elastic Net. I chose Ridge over Lasso (although they have comparable performance) because, in view of the dataset, I did not value the possibility that some features weights are set to zero.
3. Polynomial regression.
4. SVM with polynomial kernel.


### Random Forest


```python
param_grid = {'min_samples_split': [2, 3, 4, 5],
              'min_samples_leaf': [2,3,4,5,6,7]}
randomforest_search = RandomForestRegressor()
best_forest = GridSearchCV(randomforest_search, cv = 3, param_grid = param_grid)
best_forest.fit(X_train_scaled, y_train_set)
```




    GridSearchCV(cv=3, error_score=nan,
                 estimator=RandomForestRegressor(bootstrap=True, ccp_alpha=0.0,
                                                 criterion='mse', max_depth=None,
                                                 max_features='auto',
                                                 max_leaf_nodes=None,
                                                 max_samples=None,
                                                 min_impurity_decrease=0.0,
                                                 min_impurity_split=None,
                                                 min_samples_leaf=1,
                                                 min_samples_split=2,
                                                 min_weight_fraction_leaf=0.0,
                                                 n_estimators=100, n_jobs=None,
                                                 oob_score=False, random_state=None,
                                                 verbose=0, warm_start=False),
                 iid='deprecated', n_jobs=None,
                 param_grid={'min_samples_leaf': [2, 3, 4, 5, 6, 7],
                             'min_samples_split': [2, 3, 4, 5]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=0)




```python
best_forest.best_params_
```




    {'min_samples_leaf': 2, 'min_samples_split': 4}




```python
best_forest.best_score_
```




    0.7976122822060585



### Ridge regression


```python
params = {'alpha': [0.1,0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
ridge_search = Ridge()
gridsearch = GridSearchCV(ridge_search, param_grid = params)
gridsearch.fit(X_train_scaled, y_train_set)
```




    GridSearchCV(cv=None, error_score=nan,
                 estimator=Ridge(alpha=1.0, copy_X=True, fit_intercept=True,
                                 max_iter=None, normalize=False, random_state=None,
                                 solver='auto', tol=0.001),
                 iid='deprecated', n_jobs=None,
                 param_grid={'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 7,
                                       8, 9, 10]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=0)




```python
best_params_ridge = gridsearch.best_params_
```


```python
gridsearch.best_score_
```




    0.7268641884102542




```python
best_params_ridge
```




    {'alpha': 3}



### Support Vector Regression


```python
params = {'C': [90000,100000, 150000, 200000, 300000, 400000, 500000]}
SVM_search = SVR(kernel="rbf", degree = 2)
gridsearch = GridSearchCV(SVM_search, param_grid = params, scoring = "r2")
gridsearch.fit(X_train_scaled, y_train_set)
```




    GridSearchCV(cv=None, error_score=nan,
                 estimator=SVR(C=1.0, cache_size=200, coef0=0.0, degree=2,
                               epsilon=0.1, gamma='scale', kernel='rbf',
                               max_iter=-1, shrinking=True, tol=0.001,
                               verbose=False),
                 iid='deprecated', n_jobs=None,
                 param_grid={'C': [90000, 100000, 150000, 200000, 300000, 400000,
                                   500000]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='r2', verbose=0)




```python
best_params_SVM = gridsearch.best_params_
```


```python
best_params_SVM
```




    {'C': 500000}




```python
gridsearch.best_score_
```




    0.7850828732093288



### Polynomial regression


```python
def polynomial_space(data):
    poly = PolynomialFeatures(degree=2)
    cols_to_trans = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_basement', 'sqft_living15', 'sqft_lot15']
    X_ = poly.fit_transform(data[cols_to_trans])
    poly_df = pd.DataFrame(X_)
    full_poly_df = pd.concat([data,poly_df], axis = 1)
    return full_poly_df

X_train_poly = polynomial_space(X_train_scaled)
```


```python
params = {'alpha': [1, 2, 3, 4, 5, 6, 7, 8, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 35]}
poly_ridge = Ridge()
gridsearch = GridSearchCV(poly_ridge, param_grid = params, scoring = "r2")
gridsearch.fit(X_train_poly, y_train_set)
```




    GridSearchCV(cv=None, error_score=nan,
                 estimator=Ridge(alpha=1.0, copy_X=True, fit_intercept=True,
                                 max_iter=None, normalize=False, random_state=None,
                                 solver='auto', tol=0.001),
                 iid='deprecated', n_jobs=None,
                 param_grid={'alpha': [1, 2, 3, 4, 5, 6, 7, 8, 15, 20, 21, 22, 23,
                                       24, 25, 26, 27, 28, 35]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='r2', verbose=0)




```python
best_params_poly_ridge = gridsearch.best_params_
```


```python
best_params_poly_ridge
```




    {'alpha': 27}




```python
gridsearch.best_score_
```




    0.7828014039355116



## Preparing the test data

To execute the models on the test set, we have to clean it the same way we did with the train set. Therefore we apply on the test set the same data cleaning functions than on the train set. However, we need to keep the same cluster centers for the geographical partitionning. For this reason, we create a new function that assigns to each instance of the test set its closest cluster center (computed on the train set).



```python
def assign_cluster(row):

    dict_1 = {}
    for x, y in zip(centers_list, labels_list):
        dict_1[str(x)] = y

    long = row['long']
    lat = row['lat']
    p1 = [long, lat]
    closer = 100000

    for x in centers_list:
        d = distance.euclidean(p1, x)
        if d < closer:
            closer = d
            cluster = x

        else: pass

    return dict_1[str(cluster)]
```


```python
dict_1
```




    {'[-122.00696078   47.70592003]': 'area_1',
     '[-122.17120574   47.47929881]': 'area_2',
     '[-122.17501152   47.71541216]': 'area_3',
     '[-122.18206028   47.34908399]': 'area_4',
     '[-122.31751995   47.46324061]': 'area_5',
     '[-122.30795467   47.7238496 ]': 'area_6',
     '[-121.99714595   47.21637297]': 'area_7',
     '[-122.32120789   47.32257509]': 'area_8',
     '[-121.81557114   47.51939198]': 'area_9',
     '[-122.36375522   47.66966809]': 'area_10',
     '[-122.14955635   47.59658705]': 'area_11',
     '[-122.37977251   47.54031246]': 'area_12',
     '[-122.02704811   47.5798756 ]': 'area_13',
     '[-122.04483945   47.3703297 ]': 'area_14',
     '[-122.28131565   47.57847878]': 'area_15'}




```python
# removing the 'id' column
test_set = remove_id(test_set)

# formating dates columns into datetime format
to_date(test_set)

# creating 'days_since_renovation' feature
test_set['days_since_renovation'] = test_set.apply(days_since_renovation, axis = 1)
clip_and_drop(test_set)

# removing 'sqft_above'
test_set = remove_corr(test_set)

#Create the 15 geographical areas
ohe_geo = pd.get_dummies(test_set.apply(assign_cluster, axis = 1))

new = []
for x in range(1,16):
    new.append("area_{}".format(x))

pipo = ohe_geo[new]
test_set = pd.concat([test_set,pipo], axis = 1)

# Splitting the test set
features_to_drop = ['price', 'date', 'yr_built', 'zipcode', 'lat', 'long']
X_test_set, y_test_set = test_set.drop(axis = 1, labels = features_to_drop), test_set["price"]

#Scaling the test set
X_test_scaled = scale_num(X_test_set)
X_test_scaled = X_test_scaled.reset_index(drop =True)
```

## Ensemble method: averaging

The very basic ensemble method here above averages the four predictions of the most promising algorithms.


```python
def average_predictions(X_train, y_train, x_test, y_test):

    model1 = RandomForestRegressor(min_samples_leaf = 2, min_samples_split = 4)

    X_test_poly = polynomial_space(x_test)
    model2 = Ridge(alpha = 27)

    model3 = Ridge(alpha = 3)
    model4 = SVR(kernel="rbf", degree = 2, C = 500000)

    model1.fit(X_train,y_train)
    model2.fit(X_train_poly,y_train)
    model3.fit(X_train,y_train)
    model4.fit(X_train, y_train)

    pred1 = model1.predict(x_test)    
    pred2 = model2.predict(X_test_poly)
    pred3 = model3.predict(x_test)
    pred4 = model4.predict(x_test)

    finalpred= (pred1 + pred2 + pred3)/3

    r2 = r2_score(finalpred, y_test)
    mse = mean_squared_error(finalpred, y_test)
    mae = mean_absolute_error(finalpred, y_test)

    return r2, mse, mae, pred1
```


```python
r2_final, mse_final, mae_final, pred1 = average_predictions(X_train_scaled, y_train_set, X_test_scaled, y_test_set)
```


```python
print("The coefficient of determination for our ensemble method is {:.2f}.".format(r2_final))
print("The mean squarred error for the ensemble method is {:.2f}.".format(mse_final))
print("The mean absolute error for the ensemble method is {:.2f}.".format(mae_final))
```

    The coefficient of determination for our ensemble method is 0.66.
    The mean squarred error for the ensemble method is 32720468944.21.
    The mean absolute error for the ensemble method is 95052.55.



```python
randomforestr2 = r2_score(pred1, y_test_set)
randomforestr2
```




    0.73129591070599



The Random Forest algorithm is doing a better job on its own than the ensemble method.

# Conclusion

Alright mate, there are several points we need to talk ! Let's be honest... This project is a bit messy, the results are not really convinving and I most definitely have made a lot of mistakes. Nevertheless, this project is also the one where I learned the most while doing it. Indeed, I tried things that I never did before and accept that the final result would be imperfect. In the few paragraphs below, I think back on several important elements of the project:

 - Dividing the territory of King County using K-means algorithm is not optimal. For the reasons already mentionned: the spherical dimension of the earth is not taken into account and minimizing the total squared distance between samples and cluster centers does not guarantees us that similar "zipcode area" are grouped together. In fact, I achieved better results when I use the columns 'long' and 'lat' instead of the 15 areas.

 - Many blogs, notebooks, books that I have read praise the performances of ensemble methods so I wanted to try one (even very a basic one). From my experience, ensemble methods are more often used on classification but I implement it on a regression exercise. The least I can say is that it did not performed well... If all the models have similar performance, average their predictions might reduce the variance. Nevertheless, if one of the model outperforms the others, averaging makes little sense. Moreover, the less effective model should be removed to not jeopardize the averaged predictions.

 - More time could have been allocated to the fine tuning of the models.

 - GridSearchCV has a kind of addictive effect on me: I spend a lot of time tweaking the hyperparameters manually (with GridSearchCV). I always have the feeling that I could improve their performance by changing the hyperparameter (even of a decimal). Of course it makes no sense... Using GridSearchRandomized would have been more judicious.


## Bibliography:

1. Géron, A., n.d. Hands-On Machine Learning With Scikit-Learn, Keras, And Tensorflow. 2nd ed. O'Reilly, pp.279-442.

2. Kaggle notebook, *Predicting House Prices*, https://www.kaggle.com/burhanykiyakoglu/predicting-house-prices

## Appendices:

#### Condition:

Relative to age and grade. Coded 1-5.

1 = Poor- Worn out. Repair and overhaul needed on painted surfaces, roofing, plumbing, heating and numerous functional inadequacies. Excessive deferred maintenance and abuse, limited value-in-use, approaching abandonment or major reconstruction; reuse or change in occupancy is imminent. Effective age is near the end of the scale regardless of the actual chronological age.

2 = Fair- Badly worn. Much repair needed. Many items need refinishing or overhauling, deferred maintenance obvious, inadequate building utility and systems all shortening the life expectancy and increasing the effective age.

3 = Average- Some evidence of deferred maintenance and normal obsolescence with age in that a few minor repairs are needed, along with some refinishing. All major components still functional and contributing toward an extended life expectancy. Effective age and utility is standard for like properties of its class and usage.

4 = Good- No obvious maintenance required but neither is everything new. Appearance and utility are above the standard and the overall effective age will be lower than the typical property.

5= Very Good- All items well maintained, many having been overhauled and repaired as they have shown signs of wear, increasing the life expectancy and lowering the effective age with little deterioration or obsolescence evident with a high degree of utility.

#### Building grade:

Represents the construction quality of improvements. Grades run from grade 1 to 13. Generally defined as:

1-3 Falls short of minimum building standards. Normally cabin or inferior structure.

4 Generally older, low quality construction. Does not meet code.

5 Low construction costs and workmanship. Small, simple design.

6 Lowest grade currently meeting building code. Low quality materials and simple designs.

7 Average grade of construction and design. Commonly seen in plats and older sub-divisions.

8 Just above average in construction and design. Usually better materials in both the exterior and interior finish work.

9 Better architectural design with extra interior and exterior design and quality.

10 Homes of this quality generally have high quality features. Finish work is better and more design quality is seen in the floor plans. Generally have a larger square footage.

11 Custom design and higher quality finish work with added amenities of solid woods, bathroom fixtures and more luxurious options.

12 Custom design and excellent builders. All materials are of the highest quality and all conveniences are present.

13 Generally custom designed and built. Mansion level. Large amount of highest quality cabinet work, wood trim, marble, entry ways etc.


```python

```
