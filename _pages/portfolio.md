---
layout: archive
permalink: /portfolio/
title: "Data Science Projects"
author_profile: true
header:
  image: "/images/nature.jpg"
---
# <center> Breast Cancer Wisconsin </center>

This dataset regroups data on cells located in the breast mass.

**Description of the data:**
<br></br>
<br>Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. </br>
<br></br>
<br>**Attribute Information:**</br>

1. ID number
2. Diagnosis (M = malignant, B = benign)

<br>**From columns 3 to 32:**</br>

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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
```


```python
data = pd.read_csv("Desktop/datasets_portfolio/wbc.csv")
```

## Data profiling


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
      <th>257</th>
      <td>886776</td>
      <td>M</td>
      <td>15.320</td>
      <td>17.27</td>
      <td>103.20</td>
      <td>713.3</td>
      <td>0.13350</td>
      <td>0.22840</td>
      <td>0.24480</td>
      <td>0.124200</td>
      <td>...</td>
      <td>22.66</td>
      <td>119.80</td>
      <td>928.8</td>
      <td>0.1765</td>
      <td>0.45030</td>
      <td>0.4429</td>
      <td>0.22290</td>
      <td>0.3258</td>
      <td>0.11910</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>537</th>
      <td>919812</td>
      <td>B</td>
      <td>11.690</td>
      <td>24.44</td>
      <td>76.37</td>
      <td>406.4</td>
      <td>0.12360</td>
      <td>0.15520</td>
      <td>0.04515</td>
      <td>0.045310</td>
      <td>...</td>
      <td>32.19</td>
      <td>86.12</td>
      <td>487.7</td>
      <td>0.1768</td>
      <td>0.32510</td>
      <td>0.1395</td>
      <td>0.13080</td>
      <td>0.2803</td>
      <td>0.09970</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>363</th>
      <td>9010872</td>
      <td>B</td>
      <td>16.500</td>
      <td>18.29</td>
      <td>106.60</td>
      <td>838.1</td>
      <td>0.09686</td>
      <td>0.08468</td>
      <td>0.05862</td>
      <td>0.048350</td>
      <td>...</td>
      <td>25.45</td>
      <td>117.20</td>
      <td>1009.0</td>
      <td>0.1338</td>
      <td>0.16790</td>
      <td>0.1663</td>
      <td>0.09123</td>
      <td>0.2394</td>
      <td>0.06469</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>548</th>
      <td>923169</td>
      <td>B</td>
      <td>9.683</td>
      <td>19.34</td>
      <td>61.05</td>
      <td>285.7</td>
      <td>0.08491</td>
      <td>0.05030</td>
      <td>0.02337</td>
      <td>0.009615</td>
      <td>...</td>
      <td>25.59</td>
      <td>69.10</td>
      <td>364.2</td>
      <td>0.1199</td>
      <td>0.09546</td>
      <td>0.0935</td>
      <td>0.03846</td>
      <td>0.2552</td>
      <td>0.07920</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>40</th>
      <td>855167</td>
      <td>M</td>
      <td>13.440</td>
      <td>21.58</td>
      <td>86.18</td>
      <td>563.0</td>
      <td>0.08162</td>
      <td>0.06031</td>
      <td>0.03110</td>
      <td>0.020310</td>
      <td>...</td>
      <td>30.25</td>
      <td>102.50</td>
      <td>787.9</td>
      <td>0.1094</td>
      <td>0.20430</td>
      <td>0.2085</td>
      <td>0.11120</td>
      <td>0.2994</td>
      <td>0.07146</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>535</th>
      <td>919555</td>
      <td>M</td>
      <td>20.550</td>
      <td>20.86</td>
      <td>137.80</td>
      <td>1308.0</td>
      <td>0.10460</td>
      <td>0.17390</td>
      <td>0.20850</td>
      <td>0.132200</td>
      <td>...</td>
      <td>25.48</td>
      <td>160.20</td>
      <td>1809.0</td>
      <td>0.1268</td>
      <td>0.31350</td>
      <td>0.4433</td>
      <td>0.21480</td>
      <td>0.3077</td>
      <td>0.07569</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>148</th>
      <td>86973702</td>
      <td>B</td>
      <td>14.440</td>
      <td>15.18</td>
      <td>93.97</td>
      <td>640.1</td>
      <td>0.09970</td>
      <td>0.10210</td>
      <td>0.08487</td>
      <td>0.055320</td>
      <td>...</td>
      <td>19.85</td>
      <td>108.60</td>
      <td>766.9</td>
      <td>0.1316</td>
      <td>0.27350</td>
      <td>0.3103</td>
      <td>0.15990</td>
      <td>0.2691</td>
      <td>0.07683</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>406</th>
      <td>905189</td>
      <td>B</td>
      <td>16.140</td>
      <td>14.86</td>
      <td>104.30</td>
      <td>800.0</td>
      <td>0.09495</td>
      <td>0.08501</td>
      <td>0.05500</td>
      <td>0.045280</td>
      <td>...</td>
      <td>19.58</td>
      <td>115.90</td>
      <td>947.9</td>
      <td>0.1206</td>
      <td>0.17220</td>
      <td>0.2310</td>
      <td>0.11290</td>
      <td>0.2778</td>
      <td>0.07012</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>312</th>
      <td>89382602</td>
      <td>B</td>
      <td>12.760</td>
      <td>13.37</td>
      <td>82.29</td>
      <td>504.1</td>
      <td>0.08794</td>
      <td>0.07948</td>
      <td>0.04052</td>
      <td>0.025480</td>
      <td>...</td>
      <td>16.40</td>
      <td>92.04</td>
      <td>618.8</td>
      <td>0.1194</td>
      <td>0.22080</td>
      <td>0.1769</td>
      <td>0.08411</td>
      <td>0.2564</td>
      <td>0.08253</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>398</th>
      <td>904302</td>
      <td>B</td>
      <td>11.060</td>
      <td>14.83</td>
      <td>70.31</td>
      <td>378.2</td>
      <td>0.07741</td>
      <td>0.04768</td>
      <td>0.02712</td>
      <td>0.007246</td>
      <td>...</td>
      <td>20.35</td>
      <td>80.79</td>
      <td>496.7</td>
      <td>0.1120</td>
      <td>0.18790</td>
      <td>0.2079</td>
      <td>0.05556</td>
      <td>0.2590</td>
      <td>0.09158</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 33 columns</p>
</div>




```python
data.shape
```




    (569, 33)



33 columns ? Strange, we should only have 32 columns. Let's look more into the 'Unnamed: 32' column.


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




```python
data.columns.size
```




    33




```python
data['Unnamed: 32'].isna().sum()
```




    569



The column 'Unnamed: 32' has only NaN value in it. It contains no information. Similarly, the column 'id' does not provide any information. Therefore, we will delete these.


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




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 569 entries, 0 to 568
    Data columns (total 31 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   diagnosis                569 non-null    object
     1   radius_mean              569 non-null    float64
     2   texture_mean             569 non-null    float64
     3   perimeter_mean           569 non-null    float64
     4   area_mean                569 non-null    float64
     5   smoothness_mean          569 non-null    float64
     6   compactness_mean         569 non-null    float64
     7   concavity_mean           569 non-null    float64
     8   concave points_mean      569 non-null    float64
     9   symmetry_mean            569 non-null    float64
     10  fractal_dimension_mean   569 non-null    float64
     11  radius_se                569 non-null    float64
     12  texture_se               569 non-null    float64
     13  perimeter_se             569 non-null    float64
     14  area_se                  569 non-null    float64
     15  smoothness_se            569 non-null    float64
     16  compactness_se           569 non-null    float64
     17  concavity_se             569 non-null    float64
     18  concave points_se        569 non-null    float64
     19  symmetry_se              569 non-null    float64
     20  fractal_dimension_se     569 non-null    float64
     21  radius_worst             569 non-null    float64
     22  texture_worst            569 non-null    float64
     23  perimeter_worst          569 non-null    float64
     24  area_worst               569 non-null    float64
     25  smoothness_worst         569 non-null    float64
     26  compactness_worst        569 non-null    float64
     27  concavity_worst          569 non-null    float64
     28  concave points_worst     569 non-null    float64
     29  symmetry_worst           569 non-null    float64
     30  fractal_dimension_worst  569 non-null    float64
    dtypes: float64(30), object(1)
    memory usage: 137.9+ KB


There is no NaN value anymore and all the columns are in the appropriate type. The dataset is very 'clean' in the sense that nearly none data cleaning is needed. (it the reason why i have chosen this data)

## Exploration Data Analysis


```python
predictors = data.drop(axis = 1, labels = 'diagnosis')
target = data.loc[:,'diagnosis']
```


```python
sns.countplot(target, palette = ['brown', 'gold'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1e1e1690>




![png](BreastCancerWisconsin_files/BreastCancerWisconsin_19_1.png)



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


![png](BreastCancerWisconsin_files/BreastCancerWisconsin_20_0.png)



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


![png](BreastCancerWisconsin_files/BreastCancerWisconsin_21_0.png)



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


![png](BreastCancerWisconsin_files/BreastCancerWisconsin_22_0.png)



```python
#correlation matrix
corrmat = data.corr()
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corrmat, vmax=.9,annot = True, square=True);
```


![png](BreastCancerWisconsin_files/BreastCancerWisconsin_23_0.png)



```python
def highly_corrolated_variable(df, level_correlation):

    from collections import defaultdict

    corr = df.corr()
    d = defaultdict(list)
    list_index = corr.index.to_list()
    columns = corr.columns.to_list()

    for index in list_index:

        for col in columns:
            if (corr.loc[index, col] > level_correlation) & (col != index):
                d[index].append(col)

    return d


def feature_selection_correlation(df, level_correlation):

    from collections import defaultdict

    corr = df.corr()
    d = defaultdict(list)
    list_index = corr.index.to_list()
    columns = corr.columns.to_list()

    for index in list_index:

        for col in columns:
            if (corr.loc[index, col] > level_correlation) & (col != index):
                d[index].append(col)


    target_bin = pd.get_dummies(df.diagnosis).M
    data_with_binary = pd.concat([target_bin, df], axis = 1)
    target_corr_sorted = data_with_binary.corr().M.sort_values(ascending = False)

    list_all = target_corr_sorted.index.to_list()
    list_all.remove('M')

    for x in list_all:
        if x in d:

            for y in d[x]:
                if y in list_all:
                    list_all.remove(y)

                else:pass


        else: pass

    return list_all
```


```python
highly_corrolated_variable(data, 0.95)
```




    defaultdict(list,
                {'radius_mean': ['perimeter_mean',
                  'area_mean',
                  'radius_worst',
                  'perimeter_worst'],
                 'perimeter_mean': ['radius_mean',
                  'area_mean',
                  'radius_worst',
                  'perimeter_worst'],
                 'area_mean': ['radius_mean',
                  'perimeter_mean',
                  'radius_worst',
                  'perimeter_worst',
                  'area_worst'],
                 'radius_se': ['perimeter_se', 'area_se'],
                 'perimeter_se': ['radius_se'],
                 'area_se': ['radius_se'],
                 'radius_worst': ['radius_mean',
                  'perimeter_mean',
                  'area_mean',
                  'perimeter_worst',
                  'area_worst'],
                 'perimeter_worst': ['radius_mean',
                  'perimeter_mean',
                  'area_mean',
                  'radius_worst',
                  'area_worst'],
                 'area_worst': ['area_mean', 'radius_worst', 'perimeter_worst']})




```python
feature_selection_correlation(data, 0.95)
```




    ['concave points_worst',
     'perimeter_worst',
     'concave points_mean',
     'concavity_mean',
     'concavity_worst',
     'compactness_mean',
     'compactness_worst',
     'radius_se',
     'texture_worst',
     'smoothness_worst',
     'symmetry_worst',
     'texture_mean',
     'concave points_se',
     'smoothness_mean',
     'symmetry_mean',
     'fractal_dimension_worst',
     'compactness_se',
     'concavity_se',
     'fractal_dimension_se',
     'symmetry_se',
     'texture_se',
     'fractal_dimension_mean',
     'smoothness_se']




```python
x_all = predictors
y_all = target

num_test = 0.3
X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test)
```


```python
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

cm = confusion_matrix(y_test,predictions)
print('Confusion matrix: \n',cm)
print('Classification report: \n',classification_report(y_test,predictions))
```

    Confusion matrix:
     [[107   1]
     [ 11  52]]
    Classification report:
                   precision    recall  f1-score   support

               B       0.91      0.99      0.95       108
               M       0.98      0.83      0.90        63

        accuracy                           0.93       171
       macro avg       0.94      0.91      0.92       171
    weighted avg       0.93      0.93      0.93       171




```python
predictors_standardized = (predictors - predictors.mean()) / (predictors.std())

x_all = predictors_standardized
y_all = target

num_test = 0.3
X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test)
```


```python
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

cm = confusion_matrix(y_test,predictions)
print('Confusion matrix: \n',cm)
print('Classification report: \n',classification_report(y_test,predictions))
```

    Confusion matrix:
     [[111   1]
     [  7  52]]
    Classification report:
                   precision    recall  f1-score   support

               B       0.94      0.99      0.97       112
               M       0.98      0.88      0.93        59

        accuracy                           0.95       171
       macro avg       0.96      0.94      0.95       171
    weighted avg       0.95      0.95      0.95       171




```python
predictors_standardized = (predictors - predictors.mean()) / (predictors.std())

x_all = predictors_standardized
y_all = target

num_test = 0.3
X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test)
```


```python
clf = LogisticRegression()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

cm = confusion_matrix(y_test,predictions)
print('Confusion matrix: \n',cm)
print('Classification report: \n',classification_report(y_test,predictions))
```

    Confusion matrix:
     [[102   3]
     [  7  59]]
    Classification report:
                   precision    recall  f1-score   support

               B       0.94      0.97      0.95       105
               M       0.95      0.89      0.92        66

        accuracy                           0.94       171
       macro avg       0.94      0.93      0.94       171
    weighted avg       0.94      0.94      0.94       171



    /opt/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
