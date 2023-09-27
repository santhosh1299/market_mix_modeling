# DTSC-Finance

Index (Delete/re-word at last)

1. Data Pre-processing
    1. Select only monetary columns
    2. Handle nulls
2. Data Exploration
    1. Data correlation and drop the unrelated rows
    2. Range of each column
3. Modelling
    1. Order by date column and drop it
    2. Train_test split by taking initial few weeks
    3. Train the model
    4. Testing
4. Conclusion
    1. Conclusion
    2. Next steps recommendation (consider seasonality, holidays, impression & views, etc)


```python
#Import libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import r2_score
```

# 1. Data Pre-processing

Dataset Source: [Robyn](https://github.com/facebookexperimental/Robyn)

The dataset consists of 208 weeks of revenue having.

1. 5 media spend channels: tv_S, ooh_S, print_S, facebook_S, search_S
2. 2 media channels that have also the exposure information (Impression, Clicks): facebook_I, search_clicks_P
3. Organic media without spend: newsletter
4. Control variables: events, holidays, competitor sales (competitor_sales_B)


```python
#Read the csv file
#df=pd.read_csv("/content/drive/MyDrive/01. CUB Courses/01. DTSC 5301- Data Science as a Field /01. Project/00 Data/Robyn_dt_simulated_weekly.csv")
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
      <th>125</th>
      <td>126</td>
      <td>2018-04-16</td>
      <td>9.500150e+05</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>2.101131e+07</td>
      <td>50425.565260</td>
      <td>37300</td>
      <td>3167512</td>
      <td>63397.814352</td>
      <td>na</td>
      <td>21616.000000</td>
    </tr>
    <tr>
      <th>61</th>
      <td>62</td>
      <td>2017-01-23</td>
      <td>2.179918e+06</td>
      <td>0.0</td>
      <td>0</td>
      <td>80857.333333</td>
      <td>8.675806e+07</td>
      <td>61384.373539</td>
      <td>52500</td>
      <td>6998027</td>
      <td>195073.842081</td>
      <td>na</td>
      <td>13998.000000</td>
    </tr>
    <tr>
      <th>49</th>
      <td>50</td>
      <td>2016-10-31</td>
      <td>2.974772e+06</td>
      <td>704960.9</td>
      <td>0</td>
      <td>0.000000</td>
      <td>7.176559e+07</td>
      <td>34659.923981</td>
      <td>36700</td>
      <td>8937615</td>
      <td>192222.857397</td>
      <td>na</td>
      <td>13989.000000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2015-11-23</td>
      <td>2.754372e+06</td>
      <td>167687.6</td>
      <td>0</td>
      <td>95463.666667</td>
      <td>7.290385e+07</td>
      <td>0.000000</td>
      <td>0</td>
      <td>8125009</td>
      <td>228213.987444</td>
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
  </tbody>
</table>
</div>




```python
# Lowering the column headings to have a consitent format
df.columns= df.columns.str.lower()
df_selected= df[['date','tv_s','ooh_s','print_s','search_s','facebook_s','revenue']]
df_selected.head()
```

Our goal of is to only estimate media spends to achieve the revenue goal, so select only the required columns

The dataset does not have any null entires

# 02. Data Exploration (WIP)


```python
#Check the relationship (correlation) between the columns

sns.heatmap(df_selected.corr(), annot=True, cmap='Greens')
```




    <AxesSubplot:>




    
![png](output_9_1.png)
    


* Spends in TV and Search channels have the high impact (correlation) on revenue
* Out Of Home (OOH) advertising spends have the lowest impact on the revenue

# 3. Modelling


```python
# Next steps recommendation

df_selected
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
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>203</th>
      <td>2019-10-14</td>
      <td>0.0</td>
      <td>60433</td>
      <td>153723.666667</td>
      <td>112100</td>
      <td>0.000000</td>
      <td>2.456240e+06</td>
    </tr>
    <tr>
      <th>204</th>
      <td>2019-10-21</td>
      <td>154917.6</td>
      <td>0</td>
      <td>0.000000</td>
      <td>103700</td>
      <td>133624.575524</td>
      <td>2.182825e+06</td>
    </tr>
    <tr>
      <th>205</th>
      <td>2019-10-28</td>
      <td>21982.5</td>
      <td>14094</td>
      <td>17476.000000</td>
      <td>114700</td>
      <td>0.000000</td>
      <td>2.377707e+06</td>
    </tr>
    <tr>
      <th>206</th>
      <td>2019-11-04</td>
      <td>22453.0</td>
      <td>0</td>
      <td>24051.333333</td>
      <td>134100</td>
      <td>0.000000</td>
      <td>2.732825e+06</td>
    </tr>
    <tr>
      <th>207</th>
      <td>2019-11-11</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>366190.759370</td>
      <td>2.767788e+06</td>
    </tr>
  </tbody>
</table>
<p>208 rows × 7 columns</p>
</div>




```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Split the data into train and test sets
X = df_selected[['tv_s', 'ooh_s', 'print_s', 'search_s', 'facebook_s']]
y = df_selected['revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Calculate and print metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

```

    Mean Squared Error (MSE): 288426479242.19617
    R-squared (R2): 0.3123289781480262
    


```python


```

# Pending tasks

* The model's accuracy (R2) is poor, improve it
  * Try normalization?
  * Try different regression technique?
  * Do more feature engineering?
* Convert it as a report with detailed explanation of the following:
  * Problem statement trying to answer?
  * Add possible source of bias
  * Data source and description of the columns
  * Add more visualizations (either in the data exploration part or as result comparisions)
  * Complete the conclusion
* Check the data, code and report in github
  * Have a readme file explaining how to run the code
* Publish the report in github

# Personal To-do (Do if time permits, not requred for this project though)

* Leran what is multiple regression and the math behind it.
* Learn how to interpret the regression equation result (for example, which input variable influence how much the target variable)
* Learn how to evaluate a regression model (r2 score, MSE, RMSE, MAE? When to use what)

