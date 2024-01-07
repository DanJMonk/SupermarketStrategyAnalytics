## From Cart to Conversion: 
## Data-Driven Marketing Strategies for a Supermarket Chain

#### Table of Contents

1. [Project Scope](#Project-Scope)
2. [Project Beginning](#Project-Begins)
    3. [Import Data](#Import-Data)
    4. [Data Cleaning](#Data-Cleaning)
5. [Analysis](#Data-Analysis)
    6. [Univariate Analysis](#Univariate-Analysis)
    7. [Bivariate Analysis](#Bivariate-Analysis)
8. [Findings and Recomendations](#Findings-and-Recomendations)
9. [Evaluation](#Evaluation)


## Project Scope
The goal of this project is to optimize the marketing strategies of a large online supermarket chain by analyzing their comprehensive purchase data. This analysis will focus on understanding customer buying behaviors, frequency of purchases, and preferences for specific product categories or combinations. Leveraging insights from this data, the project aims to provide actionable recommendations for targeted marketing campaigns, ultimately seeking to enhance customer engagement and optimize budget utilization for this year's marketing initiatives. The scope of this project includes problem identification, goal setting, data analysis methodologies, and the evaluation of findings to inform strategic marketing decisions.

#### Problem
The challenge is refining the marketing strategies of a large online supermarket, addressing inefficiencies in current approaches and exploring unexplored customer behavior insights.

#### Impact
Effective analysis and optimization of marketing strategies can lead to increased customer engagement, better resource allocation, and ultimately, enhanced revenue for the supermarket.

#### Current Solutions & Gaps
Existing marketing efforts may lack data-driven personalization and efficiency, missing potential opportunities to connect with diverse customer preferences and purchase patterns.

#### Goals
The project aims to utilize in-depth analysis of purchase data to deliver targeted marketing recommendations, focusing on customer preferences and buying habits to improve engagement and optimize the marketing budget.

## Data
The dataset comprises 2,019,501 anonymized purchase records from one month in 2022. It includes historical data with the following attributes:
- user_id
- order_number
- order_dow
- order_hour_of_day
- days_since_prior_order
- product_id
- add_to_cart_order
- reordered
- department_id
- department
- product_name

## Analysis
The analysis methods include descriptive analysis of purchase records, data visualizations, predictive modeling for factors influencing sales, and detection of patterns in purchasing habits and revenue trends. Key metrics computed will include distributions, counts, relationships between days of the week and times of day, reordered status of records, and market basket analysis.

#### Questions Guiding Analysis:

- Sales Distribution:
    - Which products are the top sellers by volume?
    - Are there any underperforming products?
    - On which days of the week do most sales occur?
    - How do purchasing patterns vary on weekends versus weekdays?
    - During which hours of the day do most sales occur?

- Customer Preferences:
    - How often are customers ordering?
    - How does the composition of orders vary among different customer segments?
    - How many items are customers typically purchasing at a time?
    - What percentage of customers buy baby products?

- Buying Habits:
    - Does the time since the last order influence the size and composition of the current order?
    - What are the common product combinations in a single order?

## Note
Findings, reccomendations, evaluation, ethical considerations, and additional considerations can be found at the end of this analysis report.

## Project Begins

### Import Data
The dataset consists of over 2 million purchase records at a renowned online supermarket. To analyze this data it needs to be loaded in DataFrames and explored and visualized with Python. In the following steps the data is loaded and previewed to help form questions for the analysis.


```python
# Import modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```

#### Check Dataset Size
Check memory requirements to not overload machine during analysis


```python
# Loading small sample of data to check memory usage for data set
df_sample = pd.read_csv('/Users/Dan/Downloads/ECommerce_consumer behaviour.csv', nrows=1000) 

# Total memory usage of the sample DataFrame
total_memory = df_sample.memory_usage(deep=True).sum()

# Average memory usage per row
average_memory_per_row = total_memory / len(df_sample)

# Estimate memory for full dataset
total_rows_in_dataset = 2019501

# Divide by 1000000 to convert output to MB
estimated_total_memory = (average_memory_per_row * total_rows_in_dataset) / 1000000 

print("Total average memory per row = " + str(round(average_memory_per_row, )) + " bytes")
print("Total estimated memory use for entire data set = " + str(round(estimated_total_memory, )) + " MB")

```

    Total average memory per row = 216 bytes
    Total estimated memory use for entire data set = 437 MB


#### Initial Data Loading and Preview
- looks like it's a many to many relationship between order_id & user_id
- days_since_prior_order should be int64 data type with NaN converted to 0
- Consistent column names without issues
- Days_since_prior_order data type should be int insted of float
- Reordered column data type should be boolean
- Blanks values in days_since_prior_order column


```python
# Load csv to dataframe
df = pd.read_csv('/Users/Dan/Downloads/ECommerce_consumer behaviour.csv')

# Preview data
print(df.head())
```

       order_id  user_id  order_number  order_dow  order_hour_of_day  \
    0   2425083    49125             1          2                 18   
    1   2425083    49125             1          2                 18   
    2   2425083    49125             1          2                 18   
    3   2425083    49125             1          2                 18   
    4   2425083    49125             1          2                 18   
    
       days_since_prior_order  product_id  add_to_cart_order  reordered  \
    0                     NaN          17                  1          0   
    1                     NaN          91                  2          0   
    2                     NaN          36                  3          0   
    3                     NaN          83                  4          0   
    4                     NaN          83                  5          0   
    
       department_id  department        product_name  
    0             13      pantry  baking ingredients  
    1             16  dairy eggs     soy lactosefree  
    2             16  dairy eggs              butter  
    3              4     produce    fresh vegetables  
    4              4     produce    fresh vegetables  



```python
# Check columns for formatting issues

print(df.columns)
```

    Index(['order_id', 'user_id', 'order_number', 'order_dow', 'order_hour_of_day',
           'days_since_prior_order', 'product_id', 'add_to_cart_order',
           'reordered', 'department_id', 'department', 'product_name'],
          dtype='object')



```python
# Check data types

print(df.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2019501 entries, 0 to 2019500
    Data columns (total 12 columns):
     #   Column                  Dtype  
    ---  ------                  -----  
     0   order_id                int64  
     1   user_id                 int64  
     2   order_number            int64  
     3   order_dow               int64  
     4   order_hour_of_day       int64  
     5   days_since_prior_order  float64
     6   product_id              int64  
     7   add_to_cart_order       int64  
     8   reordered               int64  
     9   department_id           int64  
     10  department              object 
     11  product_name            object 
    dtypes: float64(1), int64(9), object(2)
    memory usage: 184.9+ MB
    None



```python
# Check for blanks

print(df.isnull().sum())
```

    order_id                       0
    user_id                        0
    order_number                   0
    order_dow                      0
    order_hour_of_day              0
    days_since_prior_order    124342
    product_id                     0
    add_to_cart_order              0
    reordered                      0
    department_id                  0
    department                     0
    product_name                   0
    dtype: int64


##### View name list for department and product IDs


```python
# See list of department names

# Get unique pairs of department_id and department
unique_departments = df[['department_id', 'department']].drop_duplicates()

# Sort the DataFrame based on department_id 
unique_departments.sort_values(by='department_id', inplace=True)

# Print the unique departments
print(unique_departments)
```

          department_id       department
    9                 1           frozen
    2131              2            other
    18                3           bakery
    3                 4          produce
    304               5          alcohol
    70                6    international
    22                7        beverages
    631               8             pets
    123               9  dry goods pasta
    824              10             bulk
    74               11    personal care
    8                12     meat seafood
    0                13           pantry
    35               14        breakfast
    7                15     canned goods
    1                16       dairy eggs
    71               17        household
    75               18           babies
    68               19           snacks
    89               20             deli
    177              21          missing



```python
# See list of product names

# Get unique pairs of product_id and product name
unique_product_names = df[['product_id', 'product_name']].drop_duplicates()

# Sort the DataFrame based on product_id 
unique_product_names.sort_values(by='product_id', inplace=True)

# Print the unique product names
print(unique_product_names)
```

          product_id                product_name
    105            1       prepared soups salads
    1295           2           specialty cheeses
    68             3         energy granola bars
    517            4               instant foods
    656            5  marinades meat preparation
    ...          ...                         ...
    148          130    hot cereal pancake mixes
    138          131                   dry pasta
    8841         132                      beauty
    207          133  muscles joints pain relief
    686          134  specialty wines champagnes
    
    [134 rows x 2 columns]


#### Data Cleaning
- Convert blank values in days_since_prior_order to zero and assign int64 datatype
- Convert reordered column to boolean datatype
- There are no duplicate rows in data set (duplicate if all columns matching) 

##### Convert blanks to zero in days_since_prior_order


```python
# Preview top and bottom rows days since prior order
print(df['days_since_prior_order'].head(10))
print(df['days_since_prior_order'].tail(10))
```

    0   NaN
    1   NaN
    2   NaN
    3   NaN
    4   NaN
    5   NaN
    6   NaN
    7   NaN
    8   NaN
    9   NaN
    Name: days_since_prior_order, dtype: float64
    2019491    5.0
    2019492    5.0
    2019493    5.0
    2019494    5.0
    2019495    5.0
    2019496    5.0
    2019497    3.0
    2019498    3.0
    2019499    3.0
    2019500    3.0
    Name: days_since_prior_order, dtype: float64



```python
# Convert all NaN in days_since_prior_order to 0 
df['days_since_prior_order'] = df['days_since_prior_order'].fillna(0)

# Convert the column data type integer 
df['days_since_prior_order'] = df['days_since_prior_order'].astype(int)

print(df['days_since_prior_order'].head())
print(df['days_since_prior_order'].tail())
```

    0    0
    1    0
    2    0
    3    0
    4    0
    Name: days_since_prior_order, dtype: int64
    2019496    5
    2019497    3
    2019498    3
    2019499    3
    2019500    3
    Name: days_since_prior_order, dtype: int64


##### Convert reordered to boolean data type


```python
# Convert reordered column to boolean 0 = False 1 = True
df['reordered'] = df['reordered'].astype(bool)
print(df['reordered'].head())
```

    0    False
    1    False
    2    False
    3    False
    4    False
    Name: reordered, dtype: bool


##### Check dataset for duplicate rows


```python
#Check for duplicates (duplicate if all columns matching)
duplicates = df.duplicated(keep=False)
duplicate_rows = df[duplicates]

print(duplicate_rows)

```

    Empty DataFrame
    Columns: [order_id, user_id, order_number, order_dow, order_hour_of_day, days_since_prior_order, product_id, add_to_cart_order, reordered, department_id, department, product_name]
    Index: []


### Data Analysis 

#### Basic summary statistics


```python
# Basic summary statistics 

print(df.describe())
```

               order_id       user_id  order_number     order_dow  \
    count  2.019501e+06  2.019501e+06  2.019501e+06  2.019501e+06   
    mean   1.707013e+06  1.030673e+05  1.715138e+01  2.735367e+00   
    std    9.859832e+05  5.949117e+04  1.752576e+01  2.093882e+00   
    min    1.000000e+01  2.000000e+00  1.000000e+00  0.000000e+00   
    25%    8.526490e+05  5.158400e+04  5.000000e+00  1.000000e+00   
    50%    1.705004e+06  1.026900e+05  1.100000e+01  3.000000e+00   
    75%    2.559031e+06  1.546000e+05  2.400000e+01  5.000000e+00   
    max    3.421080e+06  2.062090e+05  1.000000e+02  6.000000e+00   
    
           order_hour_of_day  days_since_prior_order    product_id  \
    count       2.019501e+06            2.019501e+06  2.019501e+06   
    mean        1.343948e+01            1.068499e+01  7.120590e+01   
    std         4.241008e+00            9.111204e+00  3.820727e+01   
    min         0.000000e+00            0.000000e+00  1.000000e+00   
    25%         1.000000e+01            4.000000e+00  3.100000e+01   
    50%         1.300000e+01            7.000000e+00  8.300000e+01   
    75%         1.600000e+01            1.500000e+01  1.070000e+02   
    max         2.300000e+01            3.000000e+01  1.340000e+02   
    
           add_to_cart_order  department_id  
    count       2.019501e+06   2.019501e+06  
    mean        8.363173e+00   9.928349e+00  
    std         7.150059e+00   6.282933e+00  
    min         1.000000e+00   1.000000e+00  
    25%         3.000000e+00   4.000000e+00  
    50%         6.000000e+00   9.000000e+00  
    75%         1.100000e+01   1.600000e+01  
    max         1.370000e+02   2.100000e+01  


#### Univariate Analysis

##### Purchases by Customer
- Which user_ids purchase the most products
    - 176478 = 460
    - 129928 = 405
    - 126305 = 384
    - 201268 = 347
    - 115495 = 283


```python
# Which customers purchase the most products
user_product_count = df.groupby('user_id')['product_id'].count()

# Sort the counts in descending order and get the top 10 users
most_user_product_count = user_product_count.sort_values(ascending=False).head(10)

print(most_user_product_count)
```

    user_id
    176478    460
    129928    405
    126305    384
    201268    347
    115495    283
    100330    271
    31903     270
    15503     258
    105213    245
    203166    240
    Name: product_id, dtype: int64



```python
most_user_product_count.plot(kind='bar', title='Top 10 Users with Most Products Purchased')
plt.xlabel('User ID')
plt.ylabel('Products Purchased')
plt.xticks(rotation = 45)
plt.show()
```


    
![png](output_30_0.png)
    


##### Purchases by product 
- Which products are the top sellers by volume?
    - Produce is the number one selling department
    - Fresh Fruits top the list as the most sold product
    - Fresh Vegetables comes in second place not far behind Fresh Fruit
    - Fresh Fruits and Vegetables account for approx 22% of purchases
- Top 5 performing products:
    - Fruit = 226,039
    - Vegetables = 212,611
    - Packaged vegetables = 109,596
    - Yogurt = 90,751
    - Packaged cheese = 61,502
- Underperforming products:
    - Frozen juice = 279
    - Beauty = 387
    - Baby accessories = 504
    - Baby bath body care = 515
    - Kitchen supplies = 561


```python
# Summary Table: Purchases by product

# Group by 'product_id' and 'product_name', then count occurrences
summary_table_product_id = df.groupby(['product_id', 'product_name']).size().reset_index(name='Total Purchases (Desc Order)')

# Calculate proportions and add as a new column
total_purchases = summary_table_product_id['Total Purchases (Desc Order)'].sum()
summary_table_product_id['Proportion (%)'] = (summary_table_product_id['Total Purchases (Desc Order)'] / total_purchases * 100).round(2)

# Sort the DataFrame based on 'Total Purchases (Desc Order)'
summary_table_product_id.sort_values(by='Total Purchases (Desc Order)', ascending=False, inplace=True)

# Create a total row
total_row = pd.DataFrame({'product_name': ['Total Purchases'], 'Total Purchases (Desc Order)': [total_purchases], 'Proportion (%)': [100]})
summary_table_product_id = pd.concat([summary_table_product_id, total_row], ignore_index=True)

# Print the DataFrame without showing the index
print(summary_table_product_id.head(5).to_string(index=False))
print()
print(summary_table_product_id.tail(6).to_string(index=False))
print()
print(total_row)


```

     product_id               product_name  Total Purchases (Desc Order)  Proportion (%)
           24.0               fresh fruits                        226039           11.19
           83.0           fresh vegetables                        212611           10.53
          123.0 packaged vegetables fruits                        109596            5.43
          120.0                     yogurt                         90751            4.49
           21.0            packaged cheese                         61502            3.05
    
     product_id        product_name  Total Purchases (Desc Order)  Proportion (%)
           10.0    kitchen supplies                           561            0.03
          102.0 baby bath body care                           515            0.03
           82.0    baby accessories                           504            0.02
          132.0              beauty                           387            0.02
          113.0        frozen juice                           279            0.01
            NaN     Total Purchases                       2019501          100.00
    
          product_name  Total Purchases (Desc Order)  Proportion (%)
    0  Total Purchases                       2019501             100



```python
# Histogram: Purchases by product 
sns.histplot(df['product_id'], kde=False)

plt.xticks(range(0, 150, 10))

plt.title('Histogram: Purchases by Product ID')
plt.xlabel('Product ID')
plt.ylabel('Frequency')
plt.show()
plt.clf()

```


    
![png](output_33_0.png)
    



    <Figure size 640x480 with 0 Axes>


##### Sales distribution by days and time
- On which days of the week do most sales occur?
    - The most sales occur on:
        - Mon = 391,873 (19%)
        - Tues = 349,236 (17%)
        - Sun = 280,751 (14%)
    - The least amount of sales occur on:
        - Fri = 234,884 (12%)    
- How do purchasing patterns vary on weekends versus weekdays?
    - Most sales occur on Monday and Tuesday (36% of total sales) 
    - Sunday, Monday and Tuesday account for (50% of total sales)
- On which hours of the day do most sales occur?
    - The majority of sales occur between 9 AM and 5 PM (63% of sales)
    - The most sales occur between 10 AM and 11 AM
    - The least amount of sales occur between 2 AM and 4 AM

##### Total purchases per day of week


```python
# Summary Table: Total purchases per day of week

# Mapping numbers to day names
day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
temp_dow = df['order_dow'].map(day_names)

# Creating the summary table
summary_table_dow = temp_dow.value_counts().reset_index()
summary_table_dow.columns = ['Day of Week', 'Total Purchases (Desc Order)']

# Calculate proportions and add as new column
total_purchases_dow = summary_table_dow['Total Purchases (Desc Order)'].sum()
summary_table_dow['Proportion (%)'] = (summary_table_dow['Total Purchases (Desc Order)'] / total_purchases_dow * 100).round(2)

# Create a total row
total_row_dow = pd.DataFrame({'Day of Week': ['Total'], 'Total Purchases (Desc Order)': [total_purchases_dow], 'Proportion (%)': [100]})
summary_table_dow = pd.concat([summary_table_dow, total_row_dow], ignore_index = True)

print(summary_table_dow.to_string(index=False))

```

    Day of Week  Total Purchases (Desc Order)  Proportion (%)
            Mon                        391831           19.40
            Tue                        349236           17.29
            Sun                        280751           13.90
            Sat                        262157           12.98
            Wed                        261912           12.97
            Thu                        238730           11.82
            Fri                        234884           11.63
          Total                       2019501          100.00



```python
# Histogram: Total purchases per day of week
sns.histplot(df['order_dow'], kde=False)

plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun'])

plt.title('Histogram: Total Purchases Day of Week')
plt.xlabel('Day of week')
plt.ylabel('Frequency')
plt.show()
plt.clf()
```


    
![png](output_37_0.png)
    



    <Figure size 640x480 with 0 Axes>


##### Total purchases per hour of day


```python
# Summary Table: Total purchases hour of day

# Creating the summary table
summary_table_hod = df['order_hour_of_day'].value_counts().reset_index()
summary_table_hod.columns = ['Hour of Day', 'Total Purchases (Desc Order)']

total_purchases_hod = summary_table_hod['Total Purchases (Desc Order)'].sum()
summary_table_hod['Proportion (%)'] = (summary_table_hod['Total Purchases (Desc Order)'] / total_purchases_hod * 100).round(2) 

# Create a total row
total_row_hod = pd.DataFrame({'Hour of Day': ['Total'], 'Total Purchases (Desc Order)': [total_purchases_hod], 'Proportion (%)': [100]})
summary_table_hod = pd.concat([summary_table_hod, total_row_hod], ignore_index = True)

print(summary_table_hod.to_string(index=False))
```

    Hour of Day  Total Purchases (Desc Order)  Proportion (%)
             10                        173306            8.58
             11                        170291            8.43
             14                        167831            8.31
             15                        167157            8.28
             13                        166376            8.24
             12                        163511            8.10
             16                        158247            7.84
              9                        150248            7.44
             17                        129383            6.41
              8                        106754            5.29
             18                        102416            5.07
             19                         78516            3.89
             20                         62110            3.08
              7                         54143            2.68
             21                         48857            2.42
             22                         40762            2.02
             23                         24331            1.20
              6                         18293            0.91
              0                         13481            0.67
              1                          7283            0.36
              5                          5732            0.28
              2                          4210            0.21
              4                          3269            0.16
              3                          2994            0.15
          Total                       2019501          100.00



```python
# Histogram: Total purchases hour of day
sns.histplot(df['order_hour_of_day'], kde=False)

plt.xticks(range(24))

plt.title('Histogram: Total Purchases Hour of Day')
plt.xlabel('Hour of day')
plt.ylabel('Frequency')
plt.show()
plt.clf()
```


    
![png](output_40_0.png)
    



    <Figure size 640x480 with 0 Axes>


##### Purchaes for days since prior order 


```python
# Summary Table: Purchases for days since prior order

# Creating the summary table
summary_table_prior_order = df['days_since_prior_order'].value_counts()

# Rename the columns
summary_table_prior_order = summary_table_prior_order.reset_index()
summary_table_prior_order.columns = ['Number of Days', 'Total Purchases (Desc Order)']

print(summary_table_prior_order.to_string(index=False))
```

     Number of Days  Total Purchases (Desc Order)
                  7                        214126
                 30                        210814
                  6                        155685
                  0                        152015
                  5                        129089
                  4                        126250
                  8                        118722
                  3                        113263
                  2                         88737
                  9                         75120
                 14                         63414
                 10                         62073
                  1                         59200
                 13                         52204
                 11                         51476
                 12                         48171
                 15                         41531
                 16                         28632
                 21                         28359
                 17                         23921
                 20                         23234
                 18                         22777
                 19                         20140
                 22                         19950
                 28                         16459
                 23                         13779
                 24                         12833
                 27                         12804
                 25                         11832
                 29                         11590
                 26                         11301



```python
# Histogram: Purchases for days since prior order
sns.histplot(df['days_since_prior_order'], kde=False)

plt.title('Histogram: Purchases for Days Since Prior Order')
plt.xlabel('Days')
plt.ylabel('Frequency')
plt.show()
plt.clf()
```


    
![png](output_43_0.png)
    



    <Figure size 640x480 with 0 Axes>


##### Purchases for cart sizes


```python
# Summary Table: Purchases for cart sizes

# Creating the summary table
summary_table_cart = df['add_to_cart_order'].value_counts()

# Rename the columns
summary_table_cart = summary_table_cart.reset_index()
summary_table_cart.columns = ['Number of Cart Items', 'Total Purchases (Desc Order)']

print(summary_table_cart.head().to_string(index=False))
```

     Number of Cart Items  Total Purchases (Desc Order)
                        1                        200000
                        2                        190134
                        3                        178480
                        4                        165743
                        5                        151983



```python
# Histogram: Purchases for cart sizes

# Filter data to remove the upper outliers
q_hi  = df['add_to_cart_order'].quantile(0.99)

filtered_df = df[(df['add_to_cart_order'] < q_hi)]

sns.histplot(filtered_df['add_to_cart_order'], kde=False)

plt.title('Histogram: Purchases for Cart Sizes')
plt.xlabel('Number of Items')
plt.ylabel('Frequency')
plt.annotate('Note: Data filtered to exclude outliers (99th percentile)', xy=(0.5, -0.15), xycoords='axes fraction', ha='center')
plt.show()
plt.clf()
```


    
![png](output_46_0.png)
    



    <Figure size 640x480 with 0 Axes>


##### Purchases by department


```python
# Summary Table: Purchases by department

# Group by 'department_id' and 'department', then count occurrences
summary_table_department = df.groupby(['department_id', 'department']).size().reset_index(name='Total Purchases (Desc Order)')

# Sort the DataFrame based on 'Total Purchases (Desc Order)'
summary_table_department.sort_values(by='Total Purchases (Desc Order)', ascending=False, inplace=True)

# Print the summary table
print(summary_table_department.to_string(index=False))

```

     department_id      department  Total Purchases (Desc Order)
                 4         produce                        588996
                16      dairy eggs                        336915
                19          snacks                        180692
                 7       beverages                        168126
                 1          frozen                        139536
                13          pantry                        116262
                 3          bakery                         72983
                15    canned goods                         66053
                20            deli                         65176
                 9 dry goods pasta                         54054
                17       household                         46446
                14       breakfast                         44605
                12    meat seafood                         44271
                11   personal care                         28134
                18          babies                         25940
                 6   international                         16738
                 5         alcohol                          9439
                 8            pets                          6013
                21         missing                          4749
                 2           other                          2240
                10            bulk                          2133



```python
# Histogram: Purchases by department
sns.histplot(df['department_id'], kde=False)

plt.xticks(range(22))

plt.title('Histogram: Purchases by Department ID')
plt.xlabel('Department')
plt.ylabel('Frequency')
plt.show()
plt.clf()

```


    
![png](output_49_0.png)
    



    <Figure size 640x480 with 0 Axes>


##### Customer reordered status


```python
# Summary Table: Cusomter reordered status

summary_table_reordered = df['reordered'].value_counts(normalize=True).reset_index()

# Rename columns
summary_table_reordered.columns = ['Reordered Status', 'Proportion (%)']

# Convert proportion to percentage and round to whole number
summary_table_reordered['Proportion (%)'] = (summary_table_reordered['Proportion (%)'] * 100).round().astype(int)

# Value counts in a separate column
summary_table_reordered['Total Users (Desc Order)'] = df['reordered'].value_counts().values

# Print the DataFrame without showing the index
print(summary_table_reordered.to_string(index=False))
```

     Reordered Status  Proportion (%)  Total Users (Desc Order)
                 True              59                   1190986
                False              41                    828515



```python
# Pie Chart: Customer reordered status
reorder_counts = df['reordered'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(reorder_counts, labels=reorder_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Customer Reordered Status Distribution')

# Show the plot
plt.show()
plt.clf()
```


    
![png](output_52_0.png)
    



    <Figure size 640x480 with 0 Axes>


#### Bivariate Analysis
Explore relationships between pairs of variables using scatter plots, correlation matrices, etc.


##### Frequency of grocery orders:
- What is the average frequency of orders per customer?
    - On average customers are ordering twice per month


```python
# What is the average frequency of orders per customer?

# Group by 'user_id' and count 'order_id'
order_count_per_user = df.groupby('user_id')['order_id'].nunique().reset_index()

# Rename columns
order_count_per_user.columns = ['User ID', 'Order Count']

# Calculate the average order count per user
total_orders = order_count_per_user['Order Count'].sum()
average_order_count = order_count_per_user['Order Count'].mean().round(0)

# Create DataFrame for the average to append to the table
total_order_df = pd.DataFrame({'User ID': ['Total Order Count'], 'Order Count': [total_orders]})
average_df = pd.DataFrame({'User ID': ['Average Order Count'], 'Order Count': [average_order_count]})

# Append the average row to the DataFrame
order_count_per_user = pd.concat([order_count_per_user, total_order_df], ignore_index=True)
order_count_per_user = pd.concat([order_count_per_user, average_df], ignore_index=True)

# Print the DataFrame
print(order_count_per_user)

```

                        User ID  Order Count
    0                         2          2.0
    1                         3          3.0
    2                         7          1.0
    3                        10          1.0
    4                        11          1.0
    ...                     ...          ...
    105270               206206          4.0
    105271               206208          2.0
    105272               206209          3.0
    105273    Total Order Count     200000.0
    105274  Average Order Count          2.0
    
    [105275 rows x 2 columns]


##### Customer segmentation & composition of orders
- How does the composition of orders vary among new and reordering customers? 
    - New customers purchase Fresh Vegetables the most
    - Returning customers purchase Fresh Fruits the most
    - Fresh Fruits and Fresh Vegetables are often purchased together
- How many customers are purchasing from babies department
    - 7.4% of customer purchase from babies department 
    


```python
# Top 10 products purchased by NEW cusotmers

# Filter for new customer purchases (reordered == 0)
new_customer_purchases = df[df['reordered'] == 0]

# Group by 'product_id' and count occurrences
product_count_new_customers = new_customer_purchases['product_id'].value_counts().reset_index()

# Rename columns
product_count_new_customers.columns = ['Product ID', 'Purchase Count']

# Create mapping of product_id to product_name
product_name_mapping = df[['product_id', 'product_name']].drop_duplicates()

# Merge the summary table with the product name mapping
product_count_new_customers = product_count_new_customers.merge(product_name_mapping, left_on='Product ID', right_on='product_id', how='left')

# Drop the extra 'product_id' column after merging
product_count_new_customers.drop('product_id', axis=1, inplace=True)

# Reorder columns 
product_count_new_customers = product_count_new_customers[['Product ID', 'product_name', 'Purchase Count']]

# Display the top products 
print("Top 10 Products New Customers Purchase")
print(product_count_new_customers.head(10).to_string(index=False))

```

    Top 10 Products New Customers Purchase
     Product ID                  product_name  Purchase Count
             83              fresh vegetables           86440
             24                  fresh fruits           63684
            123    packaged vegetables fruits           39724
            120                        yogurt           28287
             21               packaged cheese           25638
            107                chips pretzels           18691
             37                 ice cream ice           16138
            116                frozen produce           15007
            115 water seltzer sparkling water           14097
             17            baking ingredients           13992



```python
# Visualize top 10 products NEW customers purchase

top_n = 10
top_products_new_customers = product_count_new_customers.head(top_n)

# Vertical bar plot
plt.figure(figsize=(10, 6))
sns.barplot(data=top_products_new_customers, x='Product ID', y='Purchase Count', order=top_products_new_customers['Product ID'])
plt.title('Top 10 Products Purchased by NEW Customers')
plt.xlabel('Product ID')
plt.ylabel('Purchase Count')
plt.xticks(rotation=45)
plt.show()

```


    
![png](output_58_0.png)
    



```python
# Top 10 products purchased by REPEAT cusotmers

# Filter for repeat customer purchases (reordered == 1)
repeat_customer_purchases = df[df['reordered'] == 1]

# Group by 'product_id' and count occurrences
product_count_repeat_customers = repeat_customer_purchases['product_id'].value_counts().reset_index()

# Rename columns
product_count_repeat_customers.columns = ['Product ID', 'Purchase Count']

# Create mapping of product_id to product_name
product_name_mapping_repeat = df[['product_id', 'product_name']].drop_duplicates()

# Merge the summary table with the product name mapping
product_count_repeat_customers = product_count_repeat_customers.merge(product_name_mapping_repeat, left_on='Product ID', right_on='product_id', how='left')

# Drop the extra 'product_id' column after merging
product_count_repeat_customers.drop('product_id', axis=1, inplace=True)

# Reorder columns
product_count_repeat_customers = product_count_repeat_customers[['Product ID', 'product_name', 'Purchase Count']]

# Display the top products 
print("Top 10 Products Repeat Customers Purchase")
print(product_count_repeat_customers.head(10).to_string(index=False))
```

    Top 10 Products Repeat Customers Purchase
     Product ID                  product_name  Purchase Count
             24                  fresh fruits          162355
             83              fresh vegetables          126171
            123    packaged vegetables fruits           69872
            120                        yogurt           62464
             84                          milk           43162
            115 water seltzer sparkling water           38467
             21               packaged cheese           35864
             91               soy lactosefree           27251
            107                chips pretzels           26615
            112                         bread           24540



```python
# Visualize top 10 products reorder customers purchase

top_n = 10
top_products_repeat_customers = product_count_repeat_customers.head(top_n)

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(data=top_products_repeat_customers, x='Product ID', y='Purchase Count', order = top_products_repeat_customers['Product ID'])
plt.title('Top 10 Products Purchased by REORDER Customers')
plt.xlabel('Product ID')
plt.ylabel('Purchase Count')
plt.xticks(rotation=45)
plt.show()
plt.clf()


```


    
![png](output_60_0.png)
    



    <Figure size 640x480 with 0 Axes>



```python
# Customers that purchase from baby department

# Count unique user_ids who made purchases from department_id 18
unique_user_count_dept_18 = df[df['department_id'] == 18]['user_id'].nunique()

# Count total unique user_ids in the dataset
total_unique_user_count = df['user_id'].nunique()
proportion = (unique_user_count_dept_18 / total_unique_user_count) * 100
#unique_user_count = purchases_from_dept_18['user_id'].nunique()

print(f"Number of unique users who purchased from the babies department: {unique_user_count_dept_18}")
print(f'{proportion:.2} % of customers purchase from the babies department')
```

    Number of unique users who purchased from the babies department: 7792
    7.4 % of customers purchase from the babies department


##### Average cart size:
- What is the average number of items in a cart?
    - The average number of items in a cart is 10 items (adjusted for 98th percentile)


```python
# What is the average number of items in a cart?

# Group by 'order_id' and count 'add_to_cart_order'
product_count_cart = df.groupby('order_id')['add_to_cart_order'].count().reset_index()

# Rename columns 
product_count_cart.columns = ['Order ID', 'Count Items in Cart']

# Filter out high outliers, 98th percentile
threshold = product_count_cart['Count Items in Cart'].quantile(0.98)
filtered_product_count_cart = product_count_cart[product_count_cart['Count Items in Cart'] <= threshold]

# Calculate the average amount of items in cart per order, excluding outliers
average_item_count = filtered_product_count_cart['Count Items in Cart'].mean().round(0)

# Creating a DataFrame for the average to append to the table
average_item_cart_df = pd.DataFrame({'Order ID': ['Average Items in Cart (Adjusted)'], 'Count Items in Cart': [average_item_count]})

# Append the average row to the DataFrame
product_count_cart = pd.concat([product_count_cart, average_item_cart_df], ignore_index=True)

# Print the DataFrame
print(product_count_cart.head())
print(product_count_cart.tail())

```

      Order ID  Count Items in Cart
    0       10                 15.0
    1       11                  5.0
    2       28                 16.0
    3       38                  9.0
    4       56                 10.0
                                    Order ID  Count Items in Cart
    199996                           3421019                  3.0
    199997                           3421027                 12.0
    199998                           3421074                  4.0
    199999                           3421080                  9.0
    200000  Average Items in Cart (Adjusted)                 10.0


##### Impact of time since last order:
- Does the time since the last order influence the size and composition of the current order?
    - Even though statistically significant, days_since_prior_order has almost no practical predictive power over the order_size.



```python
# Does days since last order impact the current order size?

# Count the number of items in each order
order_size = df.groupby('order_id')['product_id'].count().reset_index(name='order_size')

# Merge order size with days_since_prior_order
order_data = df[['order_id', 'days_since_prior_order']].drop_duplicates()
analysis_df = order_data.merge(order_size, on='order_id')

# Check the correlation
correlation = analysis_df['days_since_prior_order'].corr(analysis_df['order_size'])
print(f"Correlation between days since last order and order size: {correlation}")



```

    Correlation between days since last order and order size: 0.05199563371806142


The practical significance of this correlation is minimal. A correlation coefficient of 0.052 suggests that the days_since_prior_order has almost no practical predictive power over the order_size.


```python
# Scatter plot to visualize the relationship 

plt.figure(figsize=(10, 6))
sns.scatterplot(data=analysis_df, x='days_since_prior_order', y='order_size')
plt.title('Relationship: Days Since Last Order and Order Size')
plt.xlabel('Days Since Last Order')
plt.ylabel('Order Size')
plt.show()

```


    
![png](output_67_0.png)
    



```python
# Is this a significance between days since prior order and order size?

# Calculate the Pearson correlation coefficient and the p-value
from scipy.stats import pearsonr

correlation, p_value = pearsonr(analysis_df['days_since_prior_order'], analysis_df['order_size'])

print(f"Correlation: {correlation}")
print(f"P-value: {p_value}")

# Interpret the significance
alpha = 0.05  # Common threshold for statistical significance
if p_value < alpha:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")

```

    Correlation: 0.051995633718061575
    P-value: 9.190251910548935e-120
    The correlation is statistically significant.


The practical significance of this correlation in a real-world context is minimal. The correlation may be statistically significant but the correlation is very weak suggesting that days_since_prior_order has almost no practical predictive power over the order_size. 

In a dataset with a large number of observations, even very small relationships can become statistically significant. Therefore, it's important to consider both the statistical significance (p-value) and the actual strength of the relationship (correlation coefficient) when interpreting results.

##### Market basket analysis 
- What are the common product combinations in a single order?
    - Fresh fruits and fresh vegetables
    - Fresh fruits and packaged vegetables
    - Fresh vegetables and packaged vegetables
    - Fresh fruits and yogurt
    - Fresh fruits, fresh vegetables and packaged vegetables 
- Can these insights guide cross-selling or promotional strategies?
    - The analysis indicates some strong purchasing patterns, as seen in both the frequent itemsets and the derived association rules. 
    - There is a strong positive relationship between fresh herbs and fresh vegetables. There is an 84.5% chance fresh vegetables will be purchased when fresh herbs are purchased.
These insights can be used for various strategic decisions such as inventory management, product placement, and promotional strategies.



```python
# FP-Growth algorithm

# Sample size fraction (adjust down for large data set)
sample_df = df.sample(frac=1)  

from mlxtend.preprocessing import TransactionEncoder

# Create a list of transactions (each transaction is a list of items)
transactions = sample_df.groupby('order_id')['product_id'].apply(list).tolist()

# Initialize TransactionEncoder and transform the data
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
transactions_df = pd.DataFrame(te_ary, columns=te.columns_)

from mlxtend.frequent_patterns import fpgrowth

# Run FP-Growth algorithm
frequent_itemsets = fpgrowth(transactions_df, min_support=0.005, use_colnames=True)

# Filter for itemsets with more than one item
multi_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 1)]

# Sort the itemsets in descending order based on support
sorted_multi_itemsets = multi_itemsets.sort_values(by='support', ascending=False)

# Display the sorted itemsets
print(sorted_multi_itemsets.head())
```

           support       itemsets
    119   0.317560       (24, 83)
    2655  0.269870      (24, 123)
    2654  0.234555      (83, 123)
    1734  0.188225      (24, 120)
    2656  0.186580  (24, 83, 123)



```python
# Iterpret FP-Growth algorithm

from mlxtend.frequent_patterns import association_rules

# Generate rules adjust threshold as needed
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)  

# Display rules
print(rules.head())  
```

      antecedents consequents  antecedent support  consequent support   support  \
    0        (24)        (83)            0.555995            0.444360  0.317560   
    1        (83)        (24)            0.444360            0.555995  0.317560   
    2        (16)        (83)            0.093005            0.444360  0.078655   
    3        (83)        (16)            0.444360            0.093005  0.078655   
    4        (24)        (16)            0.555995            0.093005  0.070135   
    
       confidence      lift  leverage  conviction  zhangs_metric  
    0    0.571156  1.285346  0.070498    1.295670       0.499993  
    1    0.714646  1.285346  0.070498    1.555978       0.399538  
    2    0.845707  1.903203  0.037327    3.601205       0.523233  
    3    0.177007  1.903203  0.037327    1.102069       0.854096  
    4    0.126143  1.356306  0.018425    1.037922       0.591667  


## Findings and Recomendations

#### Findings
The analysis revealed that the produce department leads in sales volume, with fresh fruits and fresh vegetables at the forefront. Combined, these two categories account for 22% of overall sales. Following closely are packaged vegetables, yogurt, and packaged cheese in terms of top sales. On the other end, the worst-performing products include frozen juice, beauty products, baby accessories, baby bath body care, and kitchen supplies, in that order.

Most sales (50%) occur on Mondays, Tuesdays, and Sundays, with respective percentages of 19%, 17%, and 14%. Fridays see the least amount of sales. The majority of sales (63%) take place between 9 AM and 5 PM each day, with the peak hour being between 10 AM and 11 AM. Conversely, the least popular hour for sales is between 3 AM and 4 AM.

On average, customers place orders twice per month, with each order consisting of about 10 products. The time since the last order shows almost no practical predictive power over the size of the current order. Interestingly, about 7.4% of customers purchased products from the babies department, indicating a potential proportion of customers who are parents.

A significant majority (59%) of customers are repeat buyers. Fresh vegetables top the list for initial purchases, while fresh fruit leads in reorders. Combinations of fresh fruits with fresh vegetables and fresh fruits with yogurt are common. Notably, there is an 84% likelihood of purchasing fresh vegetables when fresh herbs are bought.

Overall, the analysis highlights strong purchasing patterns, including top-performing days of the week, peak sales hours, and best-selling products. These insights can inform strategic decisions in areas like inventory management, product placement, and promotional strategies. An unexpected finding was the significant impact of fresh herbs on the purchase of fresh vegetables.

#### Recommendations
- Targeted Inventory and Supply Chain Management:
    - Prioritize stocking high-demand products, particularly fresh fruits and vegetables, given their substantial contribution to overall sales.
    - Ensure a consistent and quality supply of fresh produce, especially on high-sales days like Mondays, Tuesdays, and Sundays.

- Strategic Marketing and Promotions:
    - Develop targeted marketing campaigns for top-selling products, focusing on the produce department.
    - Implement special promotions or discounts during peak shopping hours (9 AM to 5 PM), especially between 10 AM and 11 AM.
    - Consider tailoring marketing efforts to highlight fresh fruits and vegetables, yogurt, and packaged cheese.

- Enhancing Customer Engagement:
    - Explore loyalty programs or incentives for customers who frequently purchase baby-related products, addressing the 7.4% of customers potentially identified as parents.
    - Encourage customers to try underperforming categories (like frozen juice or kitchen supplies) through bundle deals with popular items or through targeted advertising.

- Customized Online Shopping Experience:
    - Use the insights from popular product combinations (like fresh fruits and yogurt) to suggest items to customers during their online shopping experience.
    - Develop personalized shopping recommendations based on individual customer’s purchase history, especially for frequent buyers.

- Further Research and Analysis:
    - Conduct additional research into why certain products are underperforming and test different strategies to boost their sales.
    - Continuously monitor sales data to identify emerging trends or changes in customer behavior.

## Evaluation
The project's success will be measured by the effectiveness of the marketing strategies derived from our analysis. We'll evaluate the impact of these strategies on customer engagement levels and overall sales performance. Key performance indicators (KPIs) such as changes in customer purchase frequency, average basket size, and response to targeted campaigns will be closely monitored. Additionally, we'll assess the efficiency and return on investment of the marketing budget compared to previous periods. This evaluation will validate our findings and strategies and guide future marketing efforts and data analysis methodologies for the supermarket chain.

#### Ethical Considerations
The dataset consists of real, anonymized purchase data to ensure privacy.

#### Additional Considerations
While the dataset is large and robust, it only encompasses one month's worth of data and lacks specific dates. This limitation prevents further analysis into trends related to seasons, holidays, and date-specific purchasing habits.



```python

```
