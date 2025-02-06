# **Market Mix Modelling**

**Jupyter Notebook** : https://github.com/santhosh1299/DTSC-Finance/blob/main/DTSC_Project_Report.ipynb

**Analysis Domain:** Finance

**Collaborators:**
* Niranjan Cholendiran
* Santhosh Pattamudu Manoharan
* Saaijeesh Sottalu Naresh
* Makayla Johnson



# **Introduction**

In today's highly competitive business landscape, companies face the ongoing challenge of strategically allocating their marketing budgets across various advertising channels, to drive sales and achieve a substantial return on investment. This report delves into the heart of this challenge and presents a comprehensive solution leveraging the power of "Market Mix Modeling" (MMM).

# **Data Source**

For this project, we have used a sample dataset provided by Robyn, an open-source MMM package developed by Meta Marketing Science. This dataset comprises 208 weeks' worth of sales data, offering a deep insight into the dynamics of an organization's revenue generation.

Data Source: https://github.com/facebookexperimental/Robyn

This dataset consists:

1. The allocation of financial resources across five advertising channels: TV, Print, Out-Of-Home advertising, Facebook, and Search.
2. Metrics related to media exposure, encompassing impressions and clicks, for two of the most influential media channels: Facebook and Search.
3. The role of organic media without monetary investment, such as newsletters.
4. External factors influencing sales, including events, holidays, and the performance of competitors.

# **Problem Statement**

What is the optimal allocation ratio for distributing advertising funds among diverse media channels to achieve the objective of maximizing future sales?

# **Solution**

To solve this problem, we have briefly taken the following approach to solve this problem:

1. Identified the key advertising channels that exerts the most significant influence on sales.
2. Constructed a ridge regression model using the historical media spending data in these influential channels alongside the corresponding sales outcomes.
3. Utilized the insights extracted from the model's $\beta$ coefficients to identify the most advantageous budget allocation strategy.

In the sections that follow, we provide a detailed account of our approach, methodology, findings, and recommendations.


```python
#import required libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
```

# 1. Data Pre-processing


```python
#read the csv file
df=pd.read_csv("Robyn_dt_simulated_weekly.csv")
df.sample(5)
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
      <th>Unnamed: 0</th>
      <th>DATE</th>
      <th>revenue</th>
      <th>tv_S</th>
      <th>ooh_S</th>
      <th>print_S</th>
      <th>facebook_I</th>
      <th>search_clicks_P</th>
      <th>search_S</th>
      <th>competitor_sales_B</th>
      <th>facebook_S</th>
      <th>events</th>
      <th>newsletter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2015-12-21</td>
      <td>2.215953e+06</td>
      <td>0.0</td>
      <td>520005</td>
      <td>0.000000</td>
      <td>8.802269e+06</td>
      <td>28401.744069</td>
      <td>27100</td>
      <td>7105985</td>
      <td>20687.478156</td>
      <td>na</td>
      <td>15478.000000</td>
    </tr>
    <tr>
      <th>155</th>
      <td>156</td>
      <td>2018-11-12</td>
      <td>2.666240e+06</td>
      <td>430131.9</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1.170623e+08</td>
      <td>0.000000</td>
      <td>0</td>
      <td>7958631</td>
      <td>261854.245872</td>
      <td>na</td>
      <td>19401.653846</td>
    </tr>
    <tr>
      <th>195</th>
      <td>196</td>
      <td>2019-08-19</td>
      <td>1.397435e+06</td>
      <td>0.0</td>
      <td>0</td>
      <td>24152.000000</td>
      <td>9.831605e+07</td>
      <td>0.000000</td>
      <td>0</td>
      <td>4520254</td>
      <td>292347.534332</td>
      <td>na</td>
      <td>19401.653846</td>
    </tr>
    <tr>
      <th>50</th>
      <td>51</td>
      <td>2016-11-07</td>
      <td>2.826128e+06</td>
      <td>0.0</td>
      <td>267379</td>
      <td>86578.666667</td>
      <td>6.403331e+07</td>
      <td>42223.992695</td>
      <td>31200</td>
      <td>8832206</td>
      <td>208648.339566</td>
      <td>na</td>
      <td>29426.000000</td>
    </tr>
    <tr>
      <th>141</th>
      <td>142</td>
      <td>2018-08-06</td>
      <td>2.009478e+06</td>
      <td>97811.2</td>
      <td>173577</td>
      <td>54001.000000</td>
      <td>0.000000e+00</td>
      <td>76885.476551</td>
      <td>65800</td>
      <td>6045317</td>
      <td>0.000000</td>
      <td>na</td>
      <td>59318.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# lower the column headings to maintain a consitent format
df.columns= df.columns.str.lower()
```

# 2. Feature Selection

Our objective is to only estimate media spends to achieve the revenue goal, so select only the media spend columns.


```python
df_selected= df[['date','tv_s','ooh_s','print_s','search_s','facebook_s','revenue']]
df_selected.head()
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
      <th>tv_s</th>
      <th>ooh_s</th>
      <th>print_s</th>
      <th>search_s</th>
      <th>facebook_s</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-11-23</td>
      <td>167687.6</td>
      <td>0</td>
      <td>95463.666667</td>
      <td>0</td>
      <td>228213.987444</td>
      <td>2.754372e+06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-11-30</td>
      <td>214600.9</td>
      <td>0</td>
      <td>0.000000</td>
      <td>31000</td>
      <td>34258.573511</td>
      <td>2.584277e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-12-07</td>
      <td>0.0</td>
      <td>248022</td>
      <td>3404.000000</td>
      <td>28400</td>
      <td>127691.261335</td>
      <td>2.547387e+06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-12-14</td>
      <td>625877.3</td>
      <td>0</td>
      <td>132600.000000</td>
      <td>31900</td>
      <td>84014.720306</td>
      <td>2.875220e+06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-12-21</td>
      <td>0.0</td>
      <td>520005</td>
      <td>0.000000</td>
      <td>27100</td>
      <td>20687.478156</td>
      <td>2.215953e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check for null values in the features
df_selected.isnull().any()
```




    date          False
    tv_s          False
    ooh_s         False
    print_s       False
    search_s      False
    facebook_s    False
    revenue       False
    dtype: bool



The dataset does not have any null entires.

# 3. Data Exploration


```python
#plot a Box and Whisker plot for all independent and dependent variables
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
#plot for independent varaibles
df_plot= df_selected[['tv_s', 'ooh_s', 'print_s', 'search_s', 'facebook_s']]
ax1.boxplot(df_plot.values, labels=df_plot.columns)
ax1.set_xlabel('Independent Variables')
ax1.set_ylabel('Values')
ax1.grid(True)
# plot for target varaible
df_plot= df_selected[['revenue']]
ax2.boxplot(df_plot.values, labels=df_plot.columns)
ax2.set_xlabel('Target Variable')
ax2.set_ylabel('Values')
ax2.grid(True)
fig.suptitle('Distribution Of The Data')
plt.show()
```


    
![png](output_14_0.png)
    



* The company received a minimum of 672k sales without any marketing efforts and it went upto 3.8M, marking a remarkable 470% increase, following the implementation of marketing tactics.
* The presence of numerous outliers in the independent variables may possibly  due to the influence of external factors.

## Possible Bias in the Data

**Contextual Bias:**

The context in which ads are displayed or recommendations are made can introduce bias. For example, showing certain ads in specific geographic regions can result in biased outcomes.

**Seasonal Bias:**

Some markets or products may exhibit strong seasonality. If the data is preprocessed to remove these outliers it can result in inaccurate conclusions about the effectiveness of marketing efforts.


```python
# Check the relationship (correlation) between the columns
sns.heatmap(df_selected.corr(), annot=True, cmap='Greens')
plt.show()
```


    
![png](output_16_0.png)
    


* Spends in TV and Search channels have high impact (correlation) on revenue.
* Out Of Home (OOH) advertising spends have the lowest impact on the revenue.


```python
# Drop the low correated column
df_corr= df_selected.drop(columns=['ooh_s'])
df_corr.head()
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
      <th>tv_s</th>
      <th>print_s</th>
      <th>search_s</th>
      <th>facebook_s</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-11-23</td>
      <td>167687.6</td>
      <td>95463.666667</td>
      <td>0</td>
      <td>228213.987444</td>
      <td>2.754372e+06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-11-30</td>
      <td>214600.9</td>
      <td>0.000000</td>
      <td>31000</td>
      <td>34258.573511</td>
      <td>2.584277e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-12-07</td>
      <td>0.0</td>
      <td>3404.000000</td>
      <td>28400</td>
      <td>127691.261335</td>
      <td>2.547387e+06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-12-14</td>
      <td>625877.3</td>
      <td>132600.000000</td>
      <td>31900</td>
      <td>84014.720306</td>
      <td>2.875220e+06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-12-21</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>27100</td>
      <td>20687.478156</td>
      <td>2.215953e+06</td>
    </tr>
  </tbody>
</table>
</div>



# 4. Modeling


```python
# Order the data by date
df_corr.sort_values('date',inplace= True)
```


```python
# Drop the date column as it is not considered as an influencial variable in this analysis
df_corr.drop(columns=['date'],inplace=True)
df_corr.head()
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
      <th>tv_s</th>
      <th>print_s</th>
      <th>search_s</th>
      <th>facebook_s</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>167687.6</td>
      <td>95463.666667</td>
      <td>0</td>
      <td>228213.987444</td>
      <td>2.754372e+06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>214600.9</td>
      <td>0.000000</td>
      <td>31000</td>
      <td>34258.573511</td>
      <td>2.584277e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>3404.000000</td>
      <td>28400</td>
      <td>127691.261335</td>
      <td>2.547387e+06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>625877.3</td>
      <td>132600.000000</td>
      <td>31900</td>
      <td>84014.720306</td>
      <td>2.875220e+06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>27100</td>
      <td>20687.478156</td>
      <td>2.215953e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Normalize the values to ensure all features contribute equally
scaler=preprocessing.MinMaxScaler()
scaled=scaler.fit_transform(df_corr)
df_norm=pd.DataFrame(scaled, columns= df_corr.columns)
df_norm.describe()
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
      <th>tv_s</th>
      <th>print_s</th>
      <th>search_s</th>
      <th>facebook_s</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.093920</td>
      <td>0.116803</td>
      <td>0.330845</td>
      <td>0.139325</td>
      <td>0.364436</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.180696</td>
      <td>0.203089</td>
      <td>0.263004</td>
      <td>0.205213</td>
      <td>0.226994</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.131618</td>
      <td>0.000000</td>
      <td>0.156234</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.268829</td>
      <td>0.000000</td>
      <td>0.381034</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.116464</td>
      <td>0.149357</td>
      <td>0.477442</td>
      <td>0.235255</td>
      <td>0.540733</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Split the train data as the initial 70% of the weeks and use the remaining 30% for testing
total_rows= len(df_norm)
num_train_rows= int(70*total_rows/100)
input_col_num= df_norm.shape[1]-1
df_train_x= df_norm.iloc[1:num_train_rows,:input_col_num]
df_train_y= df_norm.iloc[1:num_train_rows,input_col_num:]
df_test_x= df_norm.iloc[num_train_rows:,:input_col_num]
df_test_y= df_norm.iloc[num_train_rows:,input_col_num:]
print("Training input shape:",df_train_x.shape)
print("Training output shape:",df_train_y.shape)
print("Testing input shape:",df_test_x.shape)
print("Testing output shape:",df_test_y.shape)
```

    Training input shape: (144, 4)
    Training output shape: (144, 1)
    Testing input shape: (63, 4)
    Testing output shape: (63, 1)
    


```python
# Train the regression model and evaluate it
ridge_model = Ridge(alpha=3.0)
ridge_model.fit(df_train_x, df_train_y)
y_pred= ridge_model.predict(df_test_x)
mse_rr = mean_squared_error(df_test_y, y_pred)
r2_rr = r2_score(df_test_y, y_pred)

print("MSE:",mse_rr)
print("R2 score:", r2_rr)
```

    MSE: 0.02927144850834009
    R2 score: 0.41148076043421267
    


```python
# Compare the prdicted values with the actual sales in a plot
plt.plot(y_pred, label= 'Predicted')
plt.plot(np.array(df_test_y['revenue']), label= 'Actual')
plt.legend()
plt.show()
```


    
![png](output_25_0.png)
    


The predicted result has an $R^2$ score of 0.41.

The model has captured the overall trend, however, it has to be improved to get more accurate results.

# 5. Conclusion


```python
# Print the training results and parameters
coefficients=ridge_model.coef_[0]
labels = df_train_x.columns
sizes = coefficients # Sizes or proportions for each category
plt.figure(figsize=(6, 6))  # Optional: Set the figure size
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Fund allocation to maximize sales')
# Show the chart
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
```


    
![png](output_28_0.png)
    


The company should allocate their advertising funds in the above mentioned ratio to maximize their future sales.


# **Recommended Next Steps**

Here are a few next steps recommendation to improve the accuracy of the model:

1. **Incorporate Intermediate Metrics**: Include additional intermediate metrics such as impressions and clicks in the analysis. This will provide a more comprehensive view of the advertising campaign's effectiveness. Understanding not only sales but also the reach and response of the advertisements can lead to more informed decision-making.

2. **Account for Non-Linearity**: Explore the impact of factors like seasonality, carryover effect, and saturation effects in the model as the relationship between ad spends and sales may not always be linear.

3. **Experiment with Regression Techniques**: Diversify the approach by trying different regression techniques and incorporating hyperparameter tuning. This experimentation can help to identify the most suitable modeling method for this specific data and objectives, potentially leading to improved predictive performance.
