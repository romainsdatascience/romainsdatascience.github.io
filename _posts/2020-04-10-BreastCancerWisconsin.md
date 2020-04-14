---
title: "Breast Cancer Wisconsin"
date: 2020-04-10
tags: [Classification]
header:
  image: "images/datahorizontal.jpg"
excerpt: "Classification of cancerous tumors on the Breast Cancer Wisconsin dataset"
classes: wide
---

# <center> Breast Cancer Wisconsin </center>

In this project, I will apply basic classification models on the Breast Cancer Wisconsin dataset. The main subject of discussion is centered around the different basic classification models and the best way to preprocess the data according to the model we use. Nevertheless, this project also include brief data profiling, exploration data analysis and feature selection sections.

**Description of the data:**

This dataset regroups data on cells located in the breast mass. Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

**Attribute Information:**

1. ID number
2. Diagnosis (M = malignant, B = benign)

**From columns 3 to 32:**

Ten real-valued features are computed for each cell nucleus:

1. radius (mean of distances from center to points on the perimeter)
2. texture (standard deviation of gray-scale values)
3. perimeter
4. area
5. smoothness (local variation in radius lengths)
6. compactness (perimeter^2 / area - 1.0)
7. concavity (severity of concave portions of the contour)
8. concave points (number of concave portions of the contour)
9. symmetry
10. fractal dimension ("coastline approximation" - 1)

*Source for data and data description: http://mlr.cs.umass.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29*


```python
# Importing modules

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
```


```python
data = pd.read_csv("Desktop/datasets_portfolio/wbc.csv")
```

## Data profiling

Let's take a first look at the data.


```python
data.sample(10)
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
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>...</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
      <th>Unnamed: 32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>443</th>
      <td>909777</td>
      <td>B</td>
      <td>10.57</td>
      <td>18.32</td>
      <td>66.82</td>
      <td>340.9</td>
      <td>0.08142</td>
      <td>0.04462</td>
      <td>0.01993</td>
      <td>0.011110</td>
      <td>...</td>
      <td>23.31</td>
      <td>69.35</td>
      <td>366.3</td>
      <td>0.09794</td>
      <td>0.06542</td>
      <td>0.03986</td>
      <td>0.02222</td>
      <td>0.2699</td>
      <td>0.06736</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>54</th>
      <td>857438</td>
      <td>M</td>
      <td>15.10</td>
      <td>22.02</td>
      <td>97.26</td>
      <td>712.8</td>
      <td>0.09056</td>
      <td>0.07081</td>
      <td>0.05253</td>
      <td>0.033340</td>
      <td>...</td>
      <td>31.69</td>
      <td>117.70</td>
      <td>1030.0</td>
      <td>0.13890</td>
      <td>0.20570</td>
      <td>0.27120</td>
      <td>0.15300</td>
      <td>0.2675</td>
      <td>0.07873</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>177</th>
      <td>87281702</td>
      <td>M</td>
      <td>16.46</td>
      <td>20.11</td>
      <td>109.30</td>
      <td>832.9</td>
      <td>0.09831</td>
      <td>0.15560</td>
      <td>0.17930</td>
      <td>0.088660</td>
      <td>...</td>
      <td>28.45</td>
      <td>123.50</td>
      <td>981.2</td>
      <td>0.14150</td>
      <td>0.46670</td>
      <td>0.58620</td>
      <td>0.20350</td>
      <td>0.3054</td>
      <td>0.09519</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>122</th>
      <td>865423</td>
      <td>M</td>
      <td>24.25</td>
      <td>20.20</td>
      <td>166.20</td>
      <td>1761.0</td>
      <td>0.14470</td>
      <td>0.28670</td>
      <td>0.42680</td>
      <td>0.201200</td>
      <td>...</td>
      <td>23.99</td>
      <td>180.90</td>
      <td>2073.0</td>
      <td>0.16960</td>
      <td>0.42440</td>
      <td>0.58030</td>
      <td>0.22480</td>
      <td>0.3222</td>
      <td>0.08009</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>394</th>
      <td>903554</td>
      <td>B</td>
      <td>12.10</td>
      <td>17.72</td>
      <td>78.07</td>
      <td>446.2</td>
      <td>0.10290</td>
      <td>0.09758</td>
      <td>0.04783</td>
      <td>0.033260</td>
      <td>...</td>
      <td>25.80</td>
      <td>88.33</td>
      <td>559.5</td>
      <td>0.14320</td>
      <td>0.17730</td>
      <td>0.16030</td>
      <td>0.06266</td>
      <td>0.3049</td>
      <td>0.07081</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>340</th>
      <td>89813</td>
      <td>B</td>
      <td>14.42</td>
      <td>16.54</td>
      <td>94.15</td>
      <td>641.2</td>
      <td>0.09751</td>
      <td>0.11390</td>
      <td>0.08007</td>
      <td>0.042230</td>
      <td>...</td>
      <td>21.51</td>
      <td>111.40</td>
      <td>862.1</td>
      <td>0.12940</td>
      <td>0.33710</td>
      <td>0.37550</td>
      <td>0.14140</td>
      <td>0.3053</td>
      <td>0.08764</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>313</th>
      <td>893988</td>
      <td>B</td>
      <td>11.54</td>
      <td>10.72</td>
      <td>73.73</td>
      <td>409.1</td>
      <td>0.08597</td>
      <td>0.05969</td>
      <td>0.01367</td>
      <td>0.008907</td>
      <td>...</td>
      <td>12.87</td>
      <td>81.23</td>
      <td>467.8</td>
      <td>0.10920</td>
      <td>0.16260</td>
      <td>0.08324</td>
      <td>0.04715</td>
      <td>0.3390</td>
      <td>0.07434</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>323</th>
      <td>895100</td>
      <td>M</td>
      <td>20.34</td>
      <td>21.51</td>
      <td>135.90</td>
      <td>1264.0</td>
      <td>0.11700</td>
      <td>0.18750</td>
      <td>0.25650</td>
      <td>0.150400</td>
      <td>...</td>
      <td>31.86</td>
      <td>171.10</td>
      <td>1938.0</td>
      <td>0.15920</td>
      <td>0.44920</td>
      <td>0.53440</td>
      <td>0.26850</td>
      <td>0.5558</td>
      <td>0.10240</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>373</th>
      <td>901288</td>
      <td>M</td>
      <td>20.64</td>
      <td>17.35</td>
      <td>134.80</td>
      <td>1335.0</td>
      <td>0.09446</td>
      <td>0.10760</td>
      <td>0.15270</td>
      <td>0.089410</td>
      <td>...</td>
      <td>23.17</td>
      <td>166.80</td>
      <td>1946.0</td>
      <td>0.15620</td>
      <td>0.30550</td>
      <td>0.41590</td>
      <td>0.21120</td>
      <td>0.2689</td>
      <td>0.07055</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>383</th>
      <td>90251</td>
      <td>B</td>
      <td>12.39</td>
      <td>17.48</td>
      <td>80.64</td>
      <td>462.9</td>
      <td>0.10420</td>
      <td>0.12970</td>
      <td>0.05892</td>
      <td>0.028800</td>
      <td>...</td>
      <td>23.13</td>
      <td>95.23</td>
      <td>600.5</td>
      <td>0.14270</td>
      <td>0.35930</td>
      <td>0.32060</td>
      <td>0.09804</td>
      <td>0.2819</td>
      <td>0.11180</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 33 columns</p>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 569 entries, 0 to 568
    Data columns (total 33 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   id                       569 non-null    int64  
     1   diagnosis                569 non-null    object
     2   radius_mean              569 non-null    float64
     3   texture_mean             569 non-null    float64
     4   perimeter_mean           569 non-null    float64
     5   area_mean                569 non-null    float64
     6   smoothness_mean          569 non-null    float64
     7   compactness_mean         569 non-null    float64
     8   concavity_mean           569 non-null    float64
     9   concave points_mean      569 non-null    float64
     10  symmetry_mean            569 non-null    float64
     11  fractal_dimension_mean   569 non-null    float64
     12  radius_se                569 non-null    float64
     13  texture_se               569 non-null    float64
     14  perimeter_se             569 non-null    float64
     15  area_se                  569 non-null    float64
     16  smoothness_se            569 non-null    float64
     17  compactness_se           569 non-null    float64
     18  concavity_se             569 non-null    float64
     19  concave points_se        569 non-null    float64
     20  symmetry_se              569 non-null    float64
     21  fractal_dimension_se     569 non-null    float64
     22  radius_worst             569 non-null    float64
     23  texture_worst            569 non-null    float64
     24  perimeter_worst          569 non-null    float64
     25  area_worst               569 non-null    float64
     26  smoothness_worst         569 non-null    float64
     27  compactness_worst        569 non-null    float64
     28  concavity_worst          569 non-null    float64
     29  concave points_worst     569 non-null    float64
     30  symmetry_worst           569 non-null    float64
     31  fractal_dimension_worst  569 non-null    float64
     32  Unnamed: 32              0 non-null      float64
    dtypes: float64(31), int64(1), object(1)
    memory usage: 146.8+ KB


In the entire dataset, there is no null values. All the independant variables are in a numerical type (float64). Only the target variable, 'diagnosis', is in a object type (probably string type).  


```python
data.describe()
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
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>symmetry_mean</th>
      <th>...</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
      <th>Unnamed: 32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.690000e+02</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>...</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.037183e+07</td>
      <td>14.127292</td>
      <td>19.289649</td>
      <td>91.969033</td>
      <td>654.889104</td>
      <td>0.096360</td>
      <td>0.104341</td>
      <td>0.088799</td>
      <td>0.048919</td>
      <td>0.181162</td>
      <td>...</td>
      <td>25.677223</td>
      <td>107.261213</td>
      <td>880.583128</td>
      <td>0.132369</td>
      <td>0.254265</td>
      <td>0.272188</td>
      <td>0.114606</td>
      <td>0.290076</td>
      <td>0.083946</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.250206e+08</td>
      <td>3.524049</td>
      <td>4.301036</td>
      <td>24.298981</td>
      <td>351.914129</td>
      <td>0.014064</td>
      <td>0.052813</td>
      <td>0.079720</td>
      <td>0.038803</td>
      <td>0.027414</td>
      <td>...</td>
      <td>6.146258</td>
      <td>33.602542</td>
      <td>569.356993</td>
      <td>0.022832</td>
      <td>0.157336</td>
      <td>0.208624</td>
      <td>0.065732</td>
      <td>0.061867</td>
      <td>0.018061</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>8.670000e+03</td>
      <td>6.981000</td>
      <td>9.710000</td>
      <td>43.790000</td>
      <td>143.500000</td>
      <td>0.052630</td>
      <td>0.019380</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.106000</td>
      <td>...</td>
      <td>12.020000</td>
      <td>50.410000</td>
      <td>185.200000</td>
      <td>0.071170</td>
      <td>0.027290</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.156500</td>
      <td>0.055040</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.692180e+05</td>
      <td>11.700000</td>
      <td>16.170000</td>
      <td>75.170000</td>
      <td>420.300000</td>
      <td>0.086370</td>
      <td>0.064920</td>
      <td>0.029560</td>
      <td>0.020310</td>
      <td>0.161900</td>
      <td>...</td>
      <td>21.080000</td>
      <td>84.110000</td>
      <td>515.300000</td>
      <td>0.116600</td>
      <td>0.147200</td>
      <td>0.114500</td>
      <td>0.064930</td>
      <td>0.250400</td>
      <td>0.071460</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9.060240e+05</td>
      <td>13.370000</td>
      <td>18.840000</td>
      <td>86.240000</td>
      <td>551.100000</td>
      <td>0.095870</td>
      <td>0.092630</td>
      <td>0.061540</td>
      <td>0.033500</td>
      <td>0.179200</td>
      <td>...</td>
      <td>25.410000</td>
      <td>97.660000</td>
      <td>686.500000</td>
      <td>0.131300</td>
      <td>0.211900</td>
      <td>0.226700</td>
      <td>0.099930</td>
      <td>0.282200</td>
      <td>0.080040</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.813129e+06</td>
      <td>15.780000</td>
      <td>21.800000</td>
      <td>104.100000</td>
      <td>782.700000</td>
      <td>0.105300</td>
      <td>0.130400</td>
      <td>0.130700</td>
      <td>0.074000</td>
      <td>0.195700</td>
      <td>...</td>
      <td>29.720000</td>
      <td>125.400000</td>
      <td>1084.000000</td>
      <td>0.146000</td>
      <td>0.339100</td>
      <td>0.382900</td>
      <td>0.161400</td>
      <td>0.317900</td>
      <td>0.092080</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.113205e+08</td>
      <td>28.110000</td>
      <td>39.280000</td>
      <td>188.500000</td>
      <td>2501.000000</td>
      <td>0.163400</td>
      <td>0.345400</td>
      <td>0.426800</td>
      <td>0.201200</td>
      <td>0.304000</td>
      <td>...</td>
      <td>49.540000</td>
      <td>251.200000</td>
      <td>4254.000000</td>
      <td>0.222600</td>
      <td>1.058000</td>
      <td>1.252000</td>
      <td>0.291000</td>
      <td>0.663800</td>
      <td>0.207500</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 32 columns</p>
</div>



The range of the observation differs widely accross variables. 'Area_worst' ranges from 185.2 to 4254 while 'concavity_mean' ranges from 0.00 to 0.4268. To visualize the data properly, we will have to standardize the data.


```python
data.shape
```




    (569, 33)



33 columns ? Strange, we should only have 32 columns.


```python
data.columns
```




    Index(['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
           'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
           'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
           'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
           'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
           'fractal_dimension_se', 'radius_worst', 'texture_worst',
           'perimeter_worst', 'area_worst', 'smoothness_worst',
           'compactness_worst', 'concavity_worst', 'concave points_worst',
           'symmetry_worst', 'fractal_dimension_worst', 'Unnamed: 32'],
          dtype='object')



Let's look more into the 'Unnamed: 32' column.


```python
data['Unnamed: 32'].isna().sum()
```




    569



The column 'Unnamed: 32' has only NaN value in it. It contains no information. Similarly, the column 'id' does not provide any information. Therefore, we will remove these columns.


```python
data.drop(axis = 1, labels = ['id','Unnamed: 32'], inplace = True)
```


```python
data.isna().sum()
```




    diagnosis                  0
    radius_mean                0
    texture_mean               0
    perimeter_mean             0
    area_mean                  0
    smoothness_mean            0
    compactness_mean           0
    concavity_mean             0
    concave points_mean        0
    symmetry_mean              0
    fractal_dimension_mean     0
    radius_se                  0
    texture_se                 0
    perimeter_se               0
    area_se                    0
    smoothness_se              0
    compactness_se             0
    concavity_se               0
    concave points_se          0
    symmetry_se                0
    fractal_dimension_se       0
    radius_worst               0
    texture_worst              0
    perimeter_worst            0
    area_worst                 0
    smoothness_worst           0
    compactness_worst          0
    concavity_worst            0
    concave points_worst       0
    symmetry_worst             0
    fractal_dimension_worst    0
    dtype: int64



There is no NaN value anymore and all the columns are in the appropriate type. The dataset is very 'clean' in the sense that nearly none data cleaning is needed. (it the reason why i have chosen this data)

## Exploration Data Analysis


```python
predictors = data.drop(axis = 1, labels = 'diagnosis')
target = data.loc[:,'diagnosis']
```


```python
print(target.value_counts())

sns.countplot(target, palette = ['indianred', 'khaki'])
plt.show()
```

    B    357
    M    212
    Name: diagnosis, dtype: int64



<center> ![png](/images/BreastCancerWisconsinFinal_files/BreastCancerWisconsinFinal_23_1.png) </center>

The target data is not so much imbalanced: 357 patients have a benign tumor.


```python
data_standardized = (predictors - predictors.mean()) / (predictors.std())           
data_melted = pd.concat([target,data_standardized.iloc[:,0:10]],axis=1)
data_melted= pd.melt(data_melted,id_vars="diagnosis",
                    var_name="predictors",
                    value_name='value')
plt.figure(figsize=(14,10))
sns.boxplot(x="predictors", y="value", hue="diagnosis", data=data_melted, palette = ['indianred', 'khaki'])
plt.xticks(rotation=90)
plt.show()
```


![png](/images/BreastCancerWisconsinFinal_files/BreastCancerWisconsinFinal_25_0.png)


The mean of the observations in most independant variable differs following the classes. In some of the independant variables, the interquartile in each class does not even overlap which can be use to distinguish the two classes by a classification model.


```python
data_standardized = (predictors - predictors.mean()) / (predictors.std())           
data_melted = pd.concat([target,data_standardized.iloc[:,10:20]],axis=1)
data_melted= pd.melt(data_melted,id_vars="diagnosis",
                    var_name="predictors",
                    value_name='value')
plt.figure(figsize=(14,10))
sns.boxplot(x="predictors", y="value", hue="diagnosis", data=data_melted, palette = ['indianred', 'khaki'])
plt.xticks(rotation=90)
plt.show()
```


![png](/images/BreastCancerWisconsinFinal_files/BreastCancerWisconsinFinal_27_0.png)


The distinction between the two classes is less evident for the 10 first variables. However, some independant variables might be helpful to ditinguish the two classes e.g. there is a clear difference between 'area_se' in the benign class and milagnant class.


```python
data_standardized = (predictors - predictors.mean()) / (predictors.std())           
data_melted = pd.concat([target,data_standardized.iloc[:,20:30]],axis=1)
data_melted= pd.melt(data_melted,id_vars="diagnosis",
                    var_name="predictors",
                    value_name='value')
plt.figure(figsize=(14,10))
sns.boxplot(x="predictors", y="value", hue="diagnosis", data=data_melted, palette = ['indianred', 'khaki'])
plt.xticks(rotation=90)
plt.show()
```


![png](/images/BreastCancerWisconsinFinal_files/BreastCancerWisconsinFinal_29_0.png)


## Feature selection


```python
#correlation matrix
corrmat = data.corr()
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corrmat, vmax=.9,annot = True, square=True);
```


![png](/images/BreastCancerWisconsinFinal_files/BreastCancerWisconsinFinal_31_0.png)


Some of the independant variables are highly correlated with others. The inclusion of a correlated variable in a linear model brings few additional information and add more noise in the model. Therefore, when two variables are correlated, a common practice is to remove the one of the correlated variables and keep only one. By doing so we keep most of the information while removing noise.

To ease the process of feature selection, I have define two functions:

1. A function that will retrieve the correlated variables. The parameter 'level_correlation' enables us to choose the threshold for which the variables are considered correlated. For example, setting the parameter 'level_correlation' to 0.9 will create a dictionary of list containing all the variables that have a correlation of 0.9 or higher with other variables.

2. A function that return a list of non-highly correlated variables that can be used for modelling. How does it work?

    1. Begin with a list containing all the independant variables.

    2. Remove the variables that are correlated and only keep the one that is the more correlated with the target variable 'diagnosis'.

    3. Remark: Here the parameter 'level_correlation' defines the threshold at which variables are considered to be correlated. Thus, the lower the threshold, the smaller the list of uncorrelated variables gets.


```python
def highly_corrolated_variable(df, level_correlation):

    from collections import defaultdict

    corr = df.corr()
    dict_list = defaultdict(list)
    list_index = corr.index.to_list()
    columns = corr.columns.to_list()

    for index in list_index:

        for col in columns:
            if (corr.loc[index, col] >= level_correlation) & (col != index):
                dict_list[index].append(col)

    return dict_list


def feature_selection_correlation(df, level_correlation):

    from collections import defaultdict

    corr = df.corr()
    dict_list = defaultdict(list)
    list_index = corr.index.to_list()
    columns = corr.columns.to_list()

    for index in list_index:

        for col in columns:
            if (corr.loc[index, col] >= level_correlation) & (col != index):
                dict_list[index].append(col)


    target_bin = pd.get_dummies(df.diagnosis).M
    data_with_binary = pd.concat([target_bin, df], axis = 1)
    target_corr_sorted = data_with_binary.corr().M.sort_values(ascending = False)

    list_all = target_corr_sorted.index.to_list()
    list_all.remove('M')

    for x in list_all:
        if x in dict_list:

            for y in dict_list[x]:
                if y in list_all:
                    list_all.remove(y)

                else:pass


        else: pass

    return list_all
```


```python
highly_corrolated_variable(data, 0.7)
```




    defaultdict(list,
                {'radius_mean': ['perimeter_mean',
                  'area_mean',
                  'concave points_mean',
                  'area_se',
                  'radius_worst',
                  'perimeter_worst',
                  'area_worst',
                  'concave points_worst'],
                 'texture_mean': ['texture_worst'],
                 'perimeter_mean': ['radius_mean',
                  'area_mean',
                  'concavity_mean',
                  'concave points_mean',
                  'area_se',
                  'radius_worst',
                  'perimeter_worst',
                  'area_worst',
                  'concave points_worst'],
                 'area_mean': ['radius_mean',
                  'perimeter_mean',
                  'concave points_mean',
                  'radius_se',
                  'perimeter_se',
                  'area_se',
                  'radius_worst',
                  'perimeter_worst',
                  'area_worst',
                  'concave points_worst'],
                 'smoothness_mean': ['smoothness_worst'],
                 'compactness_mean': ['concavity_mean',
                  'concave points_mean',
                  'compactness_se',
                  'compactness_worst',
                  'concavity_worst',
                  'concave points_worst'],
                 'concavity_mean': ['perimeter_mean',
                  'compactness_mean',
                  'concave points_mean',
                  'perimeter_worst',
                  'compactness_worst',
                  'concavity_worst',
                  'concave points_worst'],
                 'concave points_mean': ['radius_mean',
                  'perimeter_mean',
                  'area_mean',
                  'compactness_mean',
                  'concavity_mean',
                  'perimeter_se',
                  'radius_worst',
                  'perimeter_worst',
                  'area_worst',
                  'concavity_worst',
                  'concave points_worst'],
                 'fractal_dimension_mean': ['fractal_dimension_worst'],
                 'radius_se': ['area_mean',
                  'perimeter_se',
                  'area_se',
                  'radius_worst',
                  'perimeter_worst',
                  'area_worst'],
                 'perimeter_se': ['area_mean',
                  'concave points_mean',
                  'radius_se',
                  'area_se',
                  'perimeter_worst',
                  'area_worst'],
                 'area_se': ['radius_mean',
                  'perimeter_mean',
                  'area_mean',
                  'radius_se',
                  'perimeter_se',
                  'radius_worst',
                  'perimeter_worst',
                  'area_worst'],
                 'compactness_se': ['compactness_mean',
                  'concavity_se',
                  'concave points_se',
                  'fractal_dimension_se'],
                 'concavity_se': ['compactness_se',
                  'concave points_se',
                  'fractal_dimension_se'],
                 'concave points_se': ['compactness_se', 'concavity_se'],
                 'fractal_dimension_se': ['compactness_se', 'concavity_se'],
                 'radius_worst': ['radius_mean',
                  'perimeter_mean',
                  'area_mean',
                  'concave points_mean',
                  'radius_se',
                  'area_se',
                  'perimeter_worst',
                  'area_worst',
                  'concave points_worst'],
                 'texture_worst': ['texture_mean'],
                 'perimeter_worst': ['radius_mean',
                  'perimeter_mean',
                  'area_mean',
                  'concavity_mean',
                  'concave points_mean',
                  'radius_se',
                  'perimeter_se',
                  'area_se',
                  'radius_worst',
                  'area_worst',
                  'concave points_worst'],
                 'area_worst': ['radius_mean',
                  'perimeter_mean',
                  'area_mean',
                  'concave points_mean',
                  'radius_se',
                  'perimeter_se',
                  'area_se',
                  'radius_worst',
                  'perimeter_worst',
                  'concave points_worst'],
                 'smoothness_worst': ['smoothness_mean'],
                 'compactness_worst': ['compactness_mean',
                  'concavity_mean',
                  'concavity_worst',
                  'concave points_worst',
                  'fractal_dimension_worst'],
                 'concavity_worst': ['compactness_mean',
                  'concavity_mean',
                  'concave points_mean',
                  'compactness_worst',
                  'concave points_worst'],
                 'concave points_worst': ['radius_mean',
                  'perimeter_mean',
                  'area_mean',
                  'compactness_mean',
                  'concavity_mean',
                  'concave points_mean',
                  'radius_worst',
                  'perimeter_worst',
                  'area_worst',
                  'compactness_worst',
                  'concavity_worst'],
                 'fractal_dimension_worst': ['fractal_dimension_mean',
                  'compactness_worst']})



The function returns a dictionnary of lists that contains all the variables 'linked' with the variables with which they have a correlation of 0.7 or higher.


```python
feature_selection_correlation(data, 0.7)
```




    ['concave points_worst',
     'radius_se',
     'texture_worst',
     'smoothness_worst',
     'symmetry_worst',
     'concave points_se',
     'symmetry_mean',
     'fractal_dimension_worst',
     'fractal_dimension_se',
     'symmetry_se',
     'texture_se',
     'smoothness_se']



The twelve independant variables here above are the twelve highest corrolated variables with the target variable under the condition that they have a correlation lower than 0.7 between them.

## Linear classification models

### Linear Discriminant Analysis

With LDA:
* A gaussian distribution for all the input variables is assumed.
* A common covariance matrix in each class is assumed.
* Multicollinearity can decrease the predictive power of the model.
* The variables that have a close to zero correlation with the target variable will add noise while not bringing information.

To answer these issues, we will standardize all the variables, choose the most relevant variables and remove the variables that have a close to zero correlation with the target variable. We could have applied some transformation to the variable to approximate a normal distribution.


```python
# Choose the relevant variables

list_var = feature_selection_correlation(data, 0.8)

# Remove variable with no correlation with target

no_corr = ['fractal_dimension_se', 'symmetry_se', 'texture_se', 'fractal_dimension_mean', 'smoothness_se']
new_list = [x for x in list_var if x not in no_corr]

# Standardize the variables
new_predictors = predictors.loc[:,new_list]
x_all = (new_predictors - new_predictors.mean()) / (new_predictors.std())

# Converting the target in a binary variable (Malignant = 1, Benign = 0)

dummies = pd.get_dummies(target)
y_all = dummies.M

# Splitting the dataset in two: train and test dataset.

num_test = 0.3

X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test)
```


```python
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


results_cv = cross_validate(clf, X_test, y_test, scoring = ['recall', 'accuracy'], cv=25)
print("The cross-validated recall is {:.3f} ".format(results_cv['test_recall'].mean()))
print("The cross-validated accuracy is {:.3f} ".format(results_cv['test_accuracy'].mean()))
```

    The cross-validated recall is 0.887
    The cross-validated accuracy is 0.953


The recall is the metric I will use to assess the performance of the different models. Indeed, I want to reduce as much as possible the number of false negative i.e. tell a patient she/he has no cancer although he/she has one. In other words, I want to predict accurately the patients that have cancer and, equivalently, have a recall of 1. I will also keep track of the accuracy of the models

Linear Discriminant Analysis correctly predict 95% of the observations. Nevertheless, the recall is only at 0.887. This is not surprising since LDA tries to approximate the Bayes classifier and strive to yield the smallest possible total number of misclassified observations, irrespective of which class the erros come from.

### Logistic Regression

To prepare the data for a logistic regression we have to:
1. Remove the maximum of noise.
2. Remove correlated inputs.

For the ease of interpretability, I did not scale the input variables.



```python
# Choose the relevant variables

list_var = feature_selection_correlation(data, 0.8)

# Remove variable with no correlation with target

no_corr = ['fractal_dimension_se', 'symmetry_se', 'texture_se', 'fractal_dimension_mean', 'smoothness_se']
new_list = [x for x in list_var if x not in no_corr]
x_all = predictors.loc[:,new_list]

# Converting the target in a binary variable (Malignant = 1, Benign = 0)

dummies = pd.get_dummies(target)
y_all = dummies.M

# Splitting the dataset in two: train and test dataset.

num_test = 0.3

X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test)
```


```python
clf = LogisticRegression()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

results_cv = cross_validate(clf, X_test, y_test, scoring = ['recall', 'accuracy'], cv=25)
print("The cross-validated recall is {:.3f} ".format(results_cv['test_recall'].mean()))
print("The cross-validated accuracy is {:.3f} ".format(results_cv['test_accuracy'].mean()))

coefficients = pd.concat([pd.DataFrame(X_train.columns, columns = ['variable']),pd.DataFrame(np.transpose(clf.coef_), columns = ['coefficient'])], axis = 1)
coefficients.sort_values(by = 'coefficient', ascending = False)
```

    The cross-validated recall is 0.927
    The cross-validated accuracy is 0.954





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
      <th>variable</th>
      <th>coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>symmetry_worst</td>
      <td>1.456574</td>
    </tr>
    <tr>
      <th>2</th>
      <td>radius_se</td>
      <td>1.420185</td>
    </tr>
    <tr>
      <th>0</th>
      <td>concave points_worst</td>
      <td>1.286041</td>
    </tr>
    <tr>
      <th>1</th>
      <td>radius_worst</td>
      <td>1.106606</td>
    </tr>
    <tr>
      <th>4</th>
      <td>smoothness_worst</td>
      <td>0.644871</td>
    </tr>
    <tr>
      <th>7</th>
      <td>symmetry_mean</td>
      <td>0.507094</td>
    </tr>
    <tr>
      <th>8</th>
      <td>fractal_dimension_worst</td>
      <td>0.497820</td>
    </tr>
    <tr>
      <th>9</th>
      <td>compactness_se</td>
      <td>0.179009</td>
    </tr>
    <tr>
      <th>3</th>
      <td>texture_worst</td>
      <td>0.176882</td>
    </tr>
    <tr>
      <th>6</th>
      <td>concave points_se</td>
      <td>0.073865</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.exp(0.073865)
```




    1.0766614463894384



Wooow! The recall has jumped to 0.927 while the accuracy remains around 0.95.
**Coefficients interpretation**

* An increase of one unit in 'symmetry_worst' accounts for an increase in the odds of having cancer disease of np.exp(1.456574) = 4.29 or 429%.
* Similarly, an increase of one unit in 'concave points_se' accounts for an increase in the odds of having cancer disease of np.exp(0.073865) = 1.077 or 7.67%.


## Random Forest

Random forest does not require much data preprocessing to perform relatively well. It deals well with outliers, does not assume any distribution and select the most relevant feature by itself. Nevertheless, what we gain in prediction accuracy, we lose in interpretability.


```python
x_all = predictors
y_all = dummies.M

# Splitting the dataset in two: train and test dataset.

num_test = 0.3

X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test)


clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


results_cv = cross_validate(clf, X_test, y_test, scoring = ['recall', 'accuracy'], cv=25)
print("The cross-validated recall is {:.3f} ".format(results_cv['test_recall'].mean()))
print("The cross-validated accuracy is {:.3f} ".format(results_cv['test_accuracy'].mean()))
```

    The cross-validated recall is 0.900
    The cross-validated accuracy is 0.936


The Random Forest performs kind of poorly compared to logistic regression. It seems to have trouble identifying cancerous tumors (recall of 0.9). However, it is important to indicate that no preprocessing was done before feeding the random forest with our inputs.

## Support Vector Machines  

Before applying the Support Vector Machines classification method to the data. I normalize the input variable so that each variable range from 0 to 1. I decide to keep all the variables since SVM performs well in high dimensional space and that feature selection based on correlation is less relevant when using a non linear kernel (radial basis function kernel).


```python
new_predictors = predictors
y_all = dummies.M

# Normalize the variables
scaler = MinMaxScaler()
x_all = scaler.fit_transform(new_predictors)

# Splitting the dataset in two: train and test dataset.

num_test = 0.3
X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test)

clf = SVC()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


results_cv = cross_validate(clf, x_all, y_all, scoring = ['recall', 'accuracy'], cv=25)
print("The cross-validated recall is {:.3f} ".format(results_cv['test_recall'].mean()))
print("The cross-validated accuracy is {:.3f} ".format(results_cv['test_accuracy'].mean()))
```

    The cross-validated recall is 0.958
    The cross-validated accuracy is 0.977


The Support Vector Machines yield good results with a recall of 0.958 and a accuracy of 0.977

## K-Nearest-Neighbors

For the KNN, I normalize the input variables. I did not remove any variable. However, the performance of KNN may improve with a reduction in dimensionality. Therefore, in a second time, I applied a Principal Component Analysis on the input variables before feeding it to KNN.


```python
new_predictors = predictors
y_all = dummies.M

# Standardize the variables

scaler = MinMaxScaler()
x_all = scaler.fit_transform(new_predictors)

# Splitting the dataset in two: train and test dataset.

num_test = 0.3
X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test)

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


results_cv = cross_validate(clf, x_all, y_all, scoring = ['recall', 'accuracy'], cv=25)
print("The cross-validated recall is {:.3f} ".format(results_cv['test_recall'].mean()))
print("The cross-validated accuracy is {:.3f} ".format(results_cv['test_accuracy'].mean()))
```

    The cross-validated recall is 0.934
    The cross-validated accuracy is 0.967



```python
new_predictors = predictors
y_all = dummies.M


pca = PCA()
pca_data = pca.fit_transform(predictors)

# Standardize the variables

scaler = MinMaxScaler()
x_all = scaler.fit_transform(new_predictors)

pca = PCA()
pca_data = pd.DataFrame(pca.fit_transform(x_all))
pca_final = pca_data.loc[:, 0:7]

# Splitting the dataset in two: train and test dataset.

num_test = 0.3
X_train, X_test, y_train, y_test = train_test_split(pca_data, y_all, test_size=num_test)

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


results_cv = cross_validate(clf, pca_final, y_all, scoring = ['recall', 'accuracy'], cv=25)
print("The cross-validated recall is {:.3f} ".format(results_cv['test_recall'].mean()))
print("The cross-validated accuracy is {:.3f} ".format(results_cv['test_accuracy'].mean()))
```

    The cross-validated recall is 0.948
    The cross-validated accuracy is 0.975


With the PCA, the performances of KNN have slightly improve. The appropriate number of principal components should be determined through cross validation.

## Conclusion

The performance varies greatly following the model we use. For example, Linear Discriminant analysis yields poor result while Support Vector Machines achieve a 95% recall. Moreover, they are numerous hyper-parameters on which we can 'play' to increase the performance. The one thing all these models have in common is that their performances vary greatly with the data you feed them with.

# Resources:


1. Machine Learning Mastery: https://machinelearningmastery.com/logistic-regression-for-machine-learning/
2. Machine Learning Mastery: https://machinelearningmastery.com/linear-discriminant-analysis-for-machine-learning/
3. James, G., Witten, D., Hastie, T. and Tibshirani, R., *An Introduction To Statistical Learning*.
