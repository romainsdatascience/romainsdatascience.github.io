---
title: "Fraud Detection using deep learning"
date: 2020-05-20
tags: [Tensorflow, Neural Network]
excerpt: "Basic NLP on Trump's speeches"
classes: wide
---


## Data description

The dataset contains transactions made by credit cards in September 2013 by european cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

<a href="https://www.kaggle.com/mlg-ulb/creditcardfraud"></a>

## Import modules


```python
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from tensorflow import backend as K
```


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    <ipython-input-102-a5fc9fdd33f9> in <module>
         13 from sklearn.model_selection import RandomizedSearchCV
         14 from sklearn.metrics import make_scorer
    ---> 15 from tensorflow import backend as K


    ImportError: cannot import name 'backend' from 'tensorflow' (/opt/anaconda3/envs/tf2/lib/python3.7/site-packages/tensorflow/__init__.py)


## Read the data


```python
data = pd.read_csv('creditcard.csv')
```

## Data Profiling

From the description, we know several things:

1. The data is highly imbalanced with 492 fraudulent transactions out of 284,807 transactions.
2. For confidentiality reasons the majority of the input variables have been transformed (PCA transformations).
3. The features 'Time' and 'Amount' have not been transformed by PCA.


```python
print('The datasets has {} rows and {} columns.'.format(data.shape[0], data.shape[1]))
```

    The datasets has 284807 rows and 31 columns.



```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 284807 entries, 0 to 284806
    Data columns (total 31 columns):
     #   Column  Non-Null Count   Dtype  
    ---  ------  --------------   -----  
     0   Time    284807 non-null  float64
     1   V1      284807 non-null  float64
     2   V2      284807 non-null  float64
     3   V3      284807 non-null  float64
     4   V4      284807 non-null  float64
     5   V5      284807 non-null  float64
     6   V6      284807 non-null  float64
     7   V7      284807 non-null  float64
     8   V8      284807 non-null  float64
     9   V9      284807 non-null  float64
     10  V10     284807 non-null  float64
     11  V11     284807 non-null  float64
     12  V12     284807 non-null  float64
     13  V13     284807 non-null  float64
     14  V14     284807 non-null  float64
     15  V15     284807 non-null  float64
     16  V16     284807 non-null  float64
     17  V17     284807 non-null  float64
     18  V18     284807 non-null  float64
     19  V19     284807 non-null  float64
     20  V20     284807 non-null  float64
     21  V21     284807 non-null  float64
     22  V22     284807 non-null  float64
     23  V23     284807 non-null  float64
     24  V24     284807 non-null  float64
     25  V25     284807 non-null  float64
     26  V26     284807 non-null  float64
     27  V27     284807 non-null  float64
     28  V28     284807 non-null  float64
     29  Amount  284807 non-null  float64
     30  Class   284807 non-null  int64  
    dtypes: float64(30), int64(1)
    memory usage: 67.4 MB


All the input variables are of the 'float64' type and there is no missing value in the dataset.


```python
data.sample(5)
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>282320</th>
      <td>170812.0</td>
      <td>2.022008</td>
      <td>0.041972</td>
      <td>-1.742753</td>
      <td>1.212770</td>
      <td>0.461721</td>
      <td>-0.690357</td>
      <td>0.379760</td>
      <td>-0.128513</td>
      <td>0.288126</td>
      <td>...</td>
      <td>0.050945</td>
      <td>0.255777</td>
      <td>-0.022608</td>
      <td>-0.448691</td>
      <td>0.402493</td>
      <td>-0.478612</td>
      <td>-0.031398</td>
      <td>-0.081430</td>
      <td>1.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37029</th>
      <td>38778.0</td>
      <td>-0.451765</td>
      <td>0.256254</td>
      <td>2.659442</td>
      <td>0.547582</td>
      <td>-0.950993</td>
      <td>0.222533</td>
      <td>-0.384636</td>
      <td>0.184891</td>
      <td>-1.485643</td>
      <td>...</td>
      <td>-0.074599</td>
      <td>0.307943</td>
      <td>0.017065</td>
      <td>0.387082</td>
      <td>-0.385559</td>
      <td>-0.217789</td>
      <td>0.211896</td>
      <td>0.145591</td>
      <td>29.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>84293</th>
      <td>60240.0</td>
      <td>-0.400545</td>
      <td>0.945926</td>
      <td>1.679598</td>
      <td>1.175528</td>
      <td>0.220309</td>
      <td>0.034641</td>
      <td>0.466287</td>
      <td>0.036702</td>
      <td>-0.624131</td>
      <td>...</td>
      <td>-0.272949</td>
      <td>-0.756520</td>
      <td>-0.051644</td>
      <td>-0.122859</td>
      <td>-0.189548</td>
      <td>0.732615</td>
      <td>-0.169642</td>
      <td>0.010626</td>
      <td>0.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5590</th>
      <td>5748.0</td>
      <td>-0.514752</td>
      <td>-0.369681</td>
      <td>0.079823</td>
      <td>-3.397274</td>
      <td>1.725499</td>
      <td>3.232077</td>
      <td>-0.833306</td>
      <td>0.665122</td>
      <td>-1.088569</td>
      <td>...</td>
      <td>-0.307747</td>
      <td>-0.487713</td>
      <td>-0.065611</td>
      <td>0.906973</td>
      <td>-0.014776</td>
      <td>-0.433156</td>
      <td>-0.151780</td>
      <td>0.103355</td>
      <td>5.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>253119</th>
      <td>156095.0</td>
      <td>-1.618037</td>
      <td>1.057979</td>
      <td>0.793057</td>
      <td>-0.862794</td>
      <td>-0.590817</td>
      <td>0.369069</td>
      <td>-1.144401</td>
      <td>-1.901101</td>
      <td>-0.249221</td>
      <td>...</td>
      <td>-1.412802</td>
      <td>-0.129842</td>
      <td>0.141859</td>
      <td>-0.383843</td>
      <td>-0.483977</td>
      <td>0.289546</td>
      <td>0.206935</td>
      <td>0.076453</td>
      <td>49.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
data['Class'].value_counts()
```




    0    284315
    1       492
    Name: Class, dtype: int64




```python
sns.countplot(data['Class'], palette = 'deep' )
```








![png](/images/AvantRandomSearch_files/AvantRandomSearch_14_1.png)


Yep ! Indeed... The dataset is really highly imbalanced ! Only 0,17% of the transactions are fraudulent.

## Data cleaning

We do not have access to the original features. All we have are their PCA transformations but what does it entail ?

The idea behind Principal Component Analysis is that the transformation that lose the least information is the one that preserves the maximum amount of variance. The first principal component (here, V1) is the normalized linear combination of the original features that has the largest variance. The second (V2) is a vector, orthogonal to V1, that "captures" the largest amount of the remaining variance. And so on...

If the assumption that by preserving the maximum amount of variance, we also preserve the maximum amount of information is true then the features conveying the most information can be ranked like: V1 > V2 > V3 ... V27 > V28.

This being said, let's take a step back and look at the different steps we take in this section:

1. The feature "Time" contains the seconds elapsed between each transaction and the first transaction in the dataset. We will get rid of this.

2. The "Amount" feature might be very useful. Nevertheless, we will have to scale it to work with it.

3. Split and standardize the data into train, valuation and test sets.


```python
# Copy the dataset
clean_data = data.copy()
```


```python
# Remove the "Time" feature
clean_data.drop("Time", axis = 1, inplace = True)
```


```python
clean_data.describe()
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
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>...</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>284807.000000</td>
      <td>284807.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.165980e-15</td>
      <td>3.416908e-16</td>
      <td>-1.373150e-15</td>
      <td>2.086869e-15</td>
      <td>9.604066e-16</td>
      <td>1.490107e-15</td>
      <td>-5.556467e-16</td>
      <td>1.177556e-16</td>
      <td>-2.406455e-15</td>
      <td>2.239751e-15</td>
      <td>...</td>
      <td>1.656562e-16</td>
      <td>-3.444850e-16</td>
      <td>2.578648e-16</td>
      <td>4.471968e-15</td>
      <td>5.340915e-16</td>
      <td>1.687098e-15</td>
      <td>-3.666453e-16</td>
      <td>-1.220404e-16</td>
      <td>88.349619</td>
      <td>0.001727</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.958696e+00</td>
      <td>1.651309e+00</td>
      <td>1.516255e+00</td>
      <td>1.415869e+00</td>
      <td>1.380247e+00</td>
      <td>1.332271e+00</td>
      <td>1.237094e+00</td>
      <td>1.194353e+00</td>
      <td>1.098632e+00</td>
      <td>1.088850e+00</td>
      <td>...</td>
      <td>7.345240e-01</td>
      <td>7.257016e-01</td>
      <td>6.244603e-01</td>
      <td>6.056471e-01</td>
      <td>5.212781e-01</td>
      <td>4.822270e-01</td>
      <td>4.036325e-01</td>
      <td>3.300833e-01</td>
      <td>250.120109</td>
      <td>0.041527</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-5.640751e+01</td>
      <td>-7.271573e+01</td>
      <td>-4.832559e+01</td>
      <td>-5.683171e+00</td>
      <td>-1.137433e+02</td>
      <td>-2.616051e+01</td>
      <td>-4.355724e+01</td>
      <td>-7.321672e+01</td>
      <td>-1.343407e+01</td>
      <td>-2.458826e+01</td>
      <td>...</td>
      <td>-3.483038e+01</td>
      <td>-1.093314e+01</td>
      <td>-4.480774e+01</td>
      <td>-2.836627e+00</td>
      <td>-1.029540e+01</td>
      <td>-2.604551e+00</td>
      <td>-2.256568e+01</td>
      <td>-1.543008e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-9.203734e-01</td>
      <td>-5.985499e-01</td>
      <td>-8.903648e-01</td>
      <td>-8.486401e-01</td>
      <td>-6.915971e-01</td>
      <td>-7.682956e-01</td>
      <td>-5.540759e-01</td>
      <td>-2.086297e-01</td>
      <td>-6.430976e-01</td>
      <td>-5.354257e-01</td>
      <td>...</td>
      <td>-2.283949e-01</td>
      <td>-5.423504e-01</td>
      <td>-1.618463e-01</td>
      <td>-3.545861e-01</td>
      <td>-3.171451e-01</td>
      <td>-3.269839e-01</td>
      <td>-7.083953e-02</td>
      <td>-5.295979e-02</td>
      <td>5.600000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.810880e-02</td>
      <td>6.548556e-02</td>
      <td>1.798463e-01</td>
      <td>-1.984653e-02</td>
      <td>-5.433583e-02</td>
      <td>-2.741871e-01</td>
      <td>4.010308e-02</td>
      <td>2.235804e-02</td>
      <td>-5.142873e-02</td>
      <td>-9.291738e-02</td>
      <td>...</td>
      <td>-2.945017e-02</td>
      <td>6.781943e-03</td>
      <td>-1.119293e-02</td>
      <td>4.097606e-02</td>
      <td>1.659350e-02</td>
      <td>-5.213911e-02</td>
      <td>1.342146e-03</td>
      <td>1.124383e-02</td>
      <td>22.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.315642e+00</td>
      <td>8.037239e-01</td>
      <td>1.027196e+00</td>
      <td>7.433413e-01</td>
      <td>6.119264e-01</td>
      <td>3.985649e-01</td>
      <td>5.704361e-01</td>
      <td>3.273459e-01</td>
      <td>5.971390e-01</td>
      <td>4.539234e-01</td>
      <td>...</td>
      <td>1.863772e-01</td>
      <td>5.285536e-01</td>
      <td>1.476421e-01</td>
      <td>4.395266e-01</td>
      <td>3.507156e-01</td>
      <td>2.409522e-01</td>
      <td>9.104512e-02</td>
      <td>7.827995e-02</td>
      <td>77.165000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.454930e+00</td>
      <td>2.205773e+01</td>
      <td>9.382558e+00</td>
      <td>1.687534e+01</td>
      <td>3.480167e+01</td>
      <td>7.330163e+01</td>
      <td>1.205895e+02</td>
      <td>2.000721e+01</td>
      <td>1.559499e+01</td>
      <td>2.374514e+01</td>
      <td>...</td>
      <td>2.720284e+01</td>
      <td>1.050309e+01</td>
      <td>2.252841e+01</td>
      <td>4.584549e+00</td>
      <td>7.519589e+00</td>
      <td>3.517346e+00</td>
      <td>3.161220e+01</td>
      <td>3.384781e+01</td>
      <td>25691.160000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 30 columns</p>
</div>




```python
# Splitting the df into train, validation and test df.

train_df, test_df = train_test_split(clean_data, test_size=0.2)
train_df, val_df = train_test_split(clean_data, test_size=0.2)

# Form np arrays of labels
train_labels = np.array(train_df.pop('Class'))
val_labels = np.array(val_df.pop('Class'))
test_labels = np.array(test_df.pop('Class'))

# Form the input variables

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

# Scale the input variables

scaler = StandardScaler()

train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features),
```

## Baseline model

First of all, we will define additional metrics other than accuracy to keep track of performance. Indeed, accuracy is less relevant when dealing with imbalanced datasets. In this case, a model predicting exclusively non-fraudulent transactions would have a 99.83% accuracy ! After that, we will create and evaluate how a basic model performs if nothing is done to mitigate the risk on performance of an imbalanced dataset.


```python
# Defining other metrics
specificity = 0.95
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.SensitivityAtSpecificity(specificity, name = 'sensitivityat0.95specificity')
]
```

Let's create the baseline model.


```python
def basic_model(metrics = METRICS):

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(train_features.shape[-1],)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation = 'relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    return model

baseline_model = basic_model()
```

As one can see here below, the basic neural network is composed of:
- one input layer.
- two dense hidden layers of 16 neurons, each of them followed by one dropout layer.
- the ouput layer.


```python
baseline_model.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_10 (Dense)             (None, 16)                480       
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 16)                0         
    _________________________________________________________________
    dense_11 (Dense)             (None, 16)                272       
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 16)                0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 1)                 17        
    =================================================================
    Total params: 769
    Trainable params: 769
    Non-trainable params: 0
    _________________________________________________________________



```python
baseline_model.fit(
    train_features,
    train_labels,
    batch_size=2048,
    epochs=20,
    validation_data=(val_features, val_labels),
    verbose=0)
```




    <tensorflow.python.keras.callbacks.History at 0x1a721bb3d0>




```python
baseline_results = baseline_model.evaluate(test_features, test_labels,
                                  batch_size=2048, verbose=0)
for name, value in zip(baseline_model.metrics_names, baseline_results):
    print(name, ': ', value)
```

    loss :  0.006685384968387253
    tp :  0.0
    fp :  0.0
    tn :  56866.0
    fn :  96.0
    accuracy :  0.9983147
    precision :  0.0
    recall :  0.0
    sensitivityat0.95specificity :  0.8229167



```python
test_predictions_baseline = baseline_model.predict(test_features, batch_size=2048)
cm = confusion_matrix(test_labels, test_predictions_baseline>0.5)

def plot_cm(cm):

    ax= plt.subplot()
    sns.heatmap(cm,
                ax = ax,
                cbar=False,
                annot=True,
                fmt='g',
                cmap = 'Blues',
               square = True)

    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
```


```python
plot_cm(cm)
```


![png](/images/AvantRandomSearch_files/AvantRandomSearch_33_0.png)


This baseline model is classifying all the instances as non-fraudulent. Not good...

## Weighted model

In this section, we run the same baseline model than in the previous section except of one small modification: we account for the imbalanced distribution of the target variable. There are several techniques to deal with imbalanced data. For example, you can try to generate copies of the under-represented class (this is called: oversampling) or to remove instances of the over-represented class (this is called undersampling). However, I have chosen to modify the class weights to account for the imbalanced classes.

When the class weights are not modified, a wrong prediction will have the same impact on the loss whether it is a false positive or a false negative (all other things being equal). In other words, there is no difference between a fraudulent transaction labeled as non-fraudulent or a non-fraudulent transaction labeled as fraudulent transaction. Nevertheless, we value the identification of a fraudulent transaction over the identification of a non-faudulent one. Therefore, through the class weights, we will modify the impact of false positive and false negative to reflect our desire to identify fraudulent transaction.


```python
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_labels),
                                                 train_labels)
```


```python
print('The weight assigned to the over-represented class is {:.3f} and the weight assigned to the under-represented class is {:.3f}.'.format(class_weights[0], class_weights[1]))
```

    The weight assigned to the over-represented class is 0.501 and the weight assigned to the under-represented class is 297.448.


You can think of it as a way to create a balanced dataset through the weights. If we multiply the two class weights by their corresponding class, we arrive at the same number.


```python
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
number0 = class_weights_dict[0] * pd.value_counts(train_labels)[0]
print("The multiplication for the over-represented class is {}".format(number0))
number1 = class_weights_dict[1] * pd.value_counts(train_labels)[1]
print("The multiplication for the over-represented class is {}".format(number1))
```

    The multiplication for the over-represented class is 113922.5
    The multiplication for the over-represented class is 113922.5



```python
weighted_model = basic_model()
```


```python
weighted_model.fit(
    train_features,
    train_labels,
    batch_size=2048,
    epochs=20,
    validation_data=(val_features, val_labels),
    verbose = 0,
    class_weight = class_weights_dict)
```




    <tensorflow.python.keras.callbacks.History at 0x1a7d5c5950>




```python
weighted_results = weighted_model.evaluate(test_features, test_labels,
                                  batch_size=2048, verbose=0)
for name, value in zip(weighted_model.metrics_names, weighted_results):
    print(name, ': ', value)
```

    loss :  0.12870246789480191
    tp :  85.0
    fp :  740.0
    tn :  56126.0
    fn :  11.0
    accuracy :  0.98681575
    precision :  0.1030303
    recall :  0.8854167
    sensitivityat0.95specificity :  0.9375



```python
test_predictions_weighted = weighted_model.predict(test_features, batch_size=2048)
cm1 = confusion_matrix(test_labels, test_predictions_weighted>0.5)

plot_cm(cm1)
```


![png](/images/AvantRandomSearch_files/AvantRandomSearch_45_0.png)


Compared to the baseline model, the performance largely improved !

## Improve hyperparameters choice

In this section, we try to increase the performance of our model through a randomized search. For the purpose, we build a new model builder function. This function has multiples parameters: number of hidden layers, number of neurons per layer, learning rate.


```python
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=(train_features.shape[-1],)):

    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))

    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
        model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(1, activation = "sigmoid"))    

    model.compile(loss = keras.losses.BinaryCrossentropy(),
              optimizer= keras.optimizers.Adam(lr=learning_rate),
              metrics= METRICS)

    return model
```


```python
top_model = build_model()
```


```python
top_model.predict(train_features[:10])
```




    array([[0.34705022],
           [0.48946247],
           [0.5311457 ],
           [0.49577686],
           [0.572325  ],
           [0.3849384 ],
           [0.3386072 ],
           [0.4653349 ],
           [0.4020078 ],
           [0.44304198]], dtype=float32)



These are the initial predictions of the model. If we take a random instance in our dataset, the probability to have a fraudulent transaction is 0.17%. Therefore, these initial predictions are far to large and will result in a large loss value for the first iterations of the model. In other words, the model will spend its first iterations to learn the distribution of the dataset.

Thus, we will modify the initial biais of the output layer. An appropriate initialisation of the biais will help our model to converge faster. This gain is particularly appreciable when running a time consuming process such as a RandomizedSearchCV.


```python
neg, pos = np.bincount(data['Class'])
initial_bias = np.log([pos/neg])
initial_bias
```




    array([-6.35935934])




```python
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=(train_features.shape[-1],)):

    output_bias = tf.keras.initializers.Constant(initial_bias)

    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))

    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
        model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(1, activation = "sigmoid", bias_initializer=output_bias))    

    model.compile(loss = keras.losses.BinaryCrossentropy(),
              optimizer= keras.optimizers.Adam(lr=learning_rate),
              metrics= METRICS)

    return model
```


```python
top_model = build_model()
top_model.predict(train_features[:10])
```




    array([[0.00487474],
           [0.00186911],
           [0.00148204],
           [0.0031276 ],
           [0.00257793],
           [0.00179559],
           [0.00226942],
           [0.00743553],
           [0.00248596],
           [0.00661289]], dtype=float32)



Yeah ! That is better !


```python
keras_model = keras.wrappers.scikit_learn.KerasRegressor(build_model)
```


```python
param_distribs = {
    "n_hidden": [1, 2, 3, 4],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2)
}

rnd_search_cv = RandomizedSearchCV(keras_model, param_distributions = param_distribs, n_iter=10, cv=3)
checkpoint_cb = keras.callbacks.ModelCheckpoint("SOTA_model_enfer.h5", save_best_only=True)

rnd_search_cv.fit(train_features, train_labels, epochs=100,
                  validation_data=(val_features, val_labels),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10), checkpoint_cb],
                 verbose = 1,
                 class_weight = class_weights_dict)
```


```python
BEST_model = keras.models.load_model("BEST_model.h5")
```

What are the parameters of the winning model ?


```python
learning_rate = tf.keras.backend.eval(BEST_model.optimizer.lr)
learning_rate
```




    0.0006451127




```python
conf = BEST_model.get_config()
print('The model is composed of {} layers having {} neurons each.'.format(len(conf['layers']), conf['layers'][0]['config']['units']))
```

    The model is composed of 9 layers having 7 neurons each.


The model saved has 9 layers in total: 4 dense layers, 4 dropout layers, 1 output layer. It has 7 neurons per dense layer and a learning rate of 0.0006451127.


```python
BEST_results = BEST_model.evaluate(test_features, test_labels,
                                  batch_size=2048, verbose=0)
for name, value in zip(weighted_model.metrics_names, BEST_results):
    print(name, ': ', value)
```

    loss :  0.1291303907226346
    tp :  76.0
    fp :  26.0
    tn :  56840.0
    fn :  20.0
    accuracy :  0.9991924
    precision :  0.74509805
    recall :  0.7916667
    sensitivityat0.95specificity :  0.9715586



```python
test_predictions_BEST = BEST_model.predict(test_features, batch_size=2048)
cm2 = confusion_matrix(test_labels, test_predictions_BEST>0.5)

plot_cm(cm2)
```


![png](/images/AvantRandomSearch_files/AvantRandomSearch_65_0.png)


# Conclusion

This is the end of our journey... It is time to remember the good times we have had together:

1. We designed a baseline model. This model had very poor performance regarding the identification of fraudulent transactions.

2. We designed a model with weighted classes to tackle to problem of fraudulent transactions identification.

3. We performed a randomized search to improve the choice of hyperparameters.

Below, we walk through the results we have for the three models.


```python
def all_plot_cm(cm, ax, title):

    sns.heatmap(cm,
                ax = ax,
                cbar=False,
                annot=True,
                fmt='g',
                cmap = 'Blues',
               square = True)

    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title(title);
```


```python
fig, ((ax1, ax2, ax3)) = plt.subplots(1,3, sharey = True, figsize=(15,15))
ax1 = all_plot_cm(cm, ax1, 'Confusion Matrix Baseline Model')
ax2 = all_plot_cm(cm1, ax2, 'Confusion Matrix Weighted Model')
ax3 = all_plot_cm(cm2, ax3, 'Confusion Matrix BEST Model')
```


![png](/images/AvantRandomSearch_files/AvantRandomSearch_69_0.png)


The Baseline model is labeling all instances as non-fraudulent. This is not very useful... The Weighted model largely improves this and does a relatively good job to identify the fraudulent transactions. Its main weakness is the large amount of false positives it is making. The BEST model decreases the number of false positives from 740 to 26 at the cost of an increase in false negatives.


```python
df_compa = pd.DataFrame([baseline_results, weighted_results, BEST_results], columns = weighted_model.metrics_names).transpose()
df_compa.columns = ['Baseline Model','Weighted Model', 'BEST model' ]
df_compa = df_compa.round(3)
```


```python
df_compa
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
      <th>Baseline Model</th>
      <th>Weighted Model</th>
      <th>BEST model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>loss</th>
      <td>0.007</td>
      <td>0.129</td>
      <td>0.129</td>
    </tr>
    <tr>
      <th>tp</th>
      <td>0.000</td>
      <td>85.000</td>
      <td>76.000</td>
    </tr>
    <tr>
      <th>fp</th>
      <td>0.000</td>
      <td>740.000</td>
      <td>26.000</td>
    </tr>
    <tr>
      <th>tn</th>
      <td>56866.000</td>
      <td>56126.000</td>
      <td>56840.000</td>
    </tr>
    <tr>
      <th>fn</th>
      <td>96.000</td>
      <td>11.000</td>
      <td>20.000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.998</td>
      <td>0.987</td>
      <td>0.999</td>
    </tr>
    <tr>
      <th>precision</th>
      <td>0.000</td>
      <td>0.103</td>
      <td>0.745</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.000</td>
      <td>0.885</td>
      <td>0.792</td>
    </tr>
    <tr>
      <th>sensitivityat0.95specificity</th>
      <td>0.823</td>
      <td>0.938</td>
      <td>0.972</td>
    </tr>
  </tbody>
</table>
</div>



The table highlights the jump that we make in precision in the BEST model. On the other hand, the recall drops from 0.885 to 0.745. The choice between the Weighted model and BEST model is not straightforward.

Imagine that a human screens through the transactions classified as fraudulent. If the Weighted model is chosen, the human will have to go through 825 transactions that contains 85 actual fraudulent transactions. With the BEST model, the number of transactions to screen goes down to 102 but only 76 of them are fraudulent. To me, the benefits of the BEST model exceed the ones of the Weighted model, nevertheless, if the human can easily spot the fraudulent transactions and that identifying the fraudulent transactions is of extreme importance one might choose the Weighted model.

## Bibliography

1. Géron, A., n.d. Hands-On Machine Learning With Scikit-Learn, Keras, And Tensorflow. 2nd ed. O'Reilly, pp.279-442.

2. Tensorflow tutorial, *Classification on imbalanced data*, https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

3. Andrej Karpathy blog, http://karpathy.github.io/2019/04/25/recipe/#2-set-up-the-end-to-end-trainingevaluation-skeleton--get-dumb-baselines

4. James, G., Witten, D., Hastie, T. and Tibshirani, R., n.d. An Introduction To Statistical Learning. Springer, pp.374-385.
