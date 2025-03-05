# Real Estate Market Analysis with Python

## Investigating Property Transactions and Customer Satisfaction

### Case Description
The real estate market is a complex and dynamic domain that interests professionals, investors, policymakers, and data analysts. Understanding market trends and customer behavior is crucial for making informed decisions. In this project, we work with a leading real estate company that has gathered property and customer data, aiming to extract valuable insights through data analysis and visualization.

## Project Objective
Our goal is to preprocess, analyze, and visualize real estate property data, generating meaningful insights about property transactions and customer profiles.

## Project Requirements
To successfully complete this project, ensure you have the following:

### Prerequisites
- Python 3.x
- Jupyter Notebook

### Required Libraries
Install the necessary Python libraries before running the project:
```python
pip install pandas numpy matplotlib seaborn
```

## Part 1: Data Preprocessing

### Cleaning the Properties Dataset

#### 1. Create a Copy of the Original Dataset
To prevent accidental modifications to raw data, we create a copy:
```python
properties_data = original_properties_data.copy()
```
This allows us to work on a safe version while keeping the original dataset accessible for reference.

#### 2. Dataset Overview
Before diving into transformations, let's get an overview of the dataset:
```python
properties_data.info()
properties_data.describe(include='all')
```
This helps us understand column data types, missing values, and overall structure.

#### 3. Evaluating Missing Values
First, we check if all required values are present:
```python
properties_data.isnull().sum()
```
Although the dataset appears complete, closer inspection reveals potential issues with the `ID` column, which seems numeric but should be treated as a string.

#### 4. Verify and Convert Data Types
Ensuring the correct data type prevents unintended errors:
```python
properties_data['ID'] = properties_data['ID'].astype(str)
```
This avoids numerical operations being mistakenly performed on `ID` values.

#### 5. Address Encoding Issues
Sometimes, datasets include unnecessary metadata like a byte order mark (BOM). We rename the column to fix this:
```python
properties_data.rename(columns={'﻿ID': 'ID'}, inplace=True)
```

#### 6. Converting String-Based Identifiers
Other similar columns (`building_number` and `property_number`) need conversion to string as well:
```python
properties_data['building_number'] = properties_data['building_number'].astype(str)
properties_data['property_number'] = properties_data['property_number'].astype(str)
```

#### 7. Final Data Type Verification
Before moving forward, let's check if our transformations were successful:
```python
properties_data.dtypes
```
This ensures that all columns have the correct data types.

---

### Cleaning the Customers Dataset

#### 1. Fixing Customer ID Encoding Issues
Just like in the properties dataset, we clean the `customer_ID` column:
```python
customers_data.rename(columns={'﻿customer_ID': 'customer_ID'}, inplace=True)
```

#### 2. Checking for Missing Values
We check for missing values in the customers dataset:
```python
customers_data.isnull().sum()
```
If missing values exist, we replace `#NUM` placeholders with pandas NA values:
```python
customers_data.replace('#NUM', pd.NA, inplace=True)
```

#### 3. Converting Categorical Data
The `entity` column represents categorical data (e.g., Individual vs. Business). We convert it to numeric values:
```python
customers_data['entity'] = customers_data['entity'].map({'Individual': 0, 'Business': 1})
```
Similarly, we map `sex` values and ensure correct representation:
```python
customers_data['sex'] = customers_data['sex'].map({'Male': 0, 'Female': 1})
```

#### 4. Standardizing String Formatting
To maintain consistency, we lowercase values in specific columns:
```python
customers_data['purpose'] = customers_data['purpose'].str.lower()
customers_data['source'] = customers_data['source'].str.lower()
```

#### 5. Processing the Mortgage Column
Mapping mortgage values to binary (0 and 1):
```python
customers_data['mortgage'] = customers_data['mortgage'].map({'Yes': 1, 'No': 0})
```

#### 6. Creating a Full Name Column
We merge the `name` and `surname` columns into a new `full_name` column:
```python
customers_data['full_name'] = customers_data['name'] + ' ' + customers_data['surname']
customers_data.drop(columns=['name', 'surname'], inplace=True)
```

#### 7. Formatting Birth Date
We convert the `birth_date` column to DateTime format:
```python
customers_data['birth_date'] = pd.to_datetime(customers_data['birth_date'], errors='coerce')
```
This ensures that we can later perform age analysis easily.

---

## Part 2: Descriptive Statistics

### Breakdown by Building
I start by creating a variable named `data` that contains all relevant data for my analysis. To get a quick overview, I display the first five rows of the dataset:
```python
data.head()
```
Next, I use the `.describe()` method to obtain descriptive statistics:
```python
data.describe(include='all', datetime_is_numeric=True)
```
To analyze the dataset based on property distribution, I create a frequency distribution table and break down averages using the `building`, `area`, `price_in_dollars`, and `deal_satisfaction` columns:
```python
averages_by_building = data.groupby('building')[['area', 'price_in_dollars', 'deal_satisfaction']].mean()
```

### Breakdown by Country
To construct a frequency distribution table for `country`, I define the columns of interest and store the grouped data:
```python
totals_by_country = data.groupby('country')[['sold', 'mortgage']].sum()
```
To remove inconsistencies caused by extra spaces, I clean the `country` column:
```python
data['country'] = data['country'].str.strip()
```

### Breakdown by State
For state-level analysis, I group data by `state`, `sold`, and `mortgage`:
```python
totals_by_state = data.groupby('state')[['sold', 'mortgage']].sum()
```
I identify missing values and replace incorrect entries for non-US customers with `pd.NA`:
```python
data.loc[~data['country'].str.contains('USA', na=False), 'state'] = pd.NA
```
I refine the table by removing `mortgage`, sorting by `sold`, and renaming columns for clarity. Finally, I compute `relative_frequency` and `cumulative_frequency`:
```python
totals_by_state['relative_frequency'] = totals_by_state['sold'] / totals_by_state['sold'].sum()
totals_by_state['cumulative_frequency'] = totals_by_state['relative_frequency'].cumsum()
```

---

## Part 3: Data Analysis

### Analyzing Customer Age
We calculate `age_at_purchase` by finding the difference between `date_of_sale` and `birth_date`:
```python
customers_data['age_at_purchase'] = (customers_data['date_of_sale'] - customers_data['birth_date']).dt.days // 365
```
We compute descriptive statistics:
```python
customers_data['age_at_purchase'].describe()
```
To analyze distribution, we create age intervals:
```python
customers_data['age_group'] = pd.cut(customers_data['age_at_purchase'], bins=5)
```

### Analyzing Property Prices
We categorize properties into price intervals:
```python
real_estate_data['price_group'] = pd.cut(real_estate_data['price_in_dollars'], bins=10)
```
We analyze sold vs. unsold properties per price interval:
```python
price_analysis = real_estate_data.groupby('price_group')['sold'].sum()
```

### Relationship between Age and Price
We analyze covariance:
```python
np.cov(customers_data['age_at_purchase'], real_estate_data['price_in_dollars'])
```
We compute the correlation coefficient:
```python
np.corrcoef(customers_data['age_at_purchase'], real_estate_data['price_in_dollars'])
```
This helps determine if customer age influences property price.

---

# Data Visualization and Interpretation

## Part 4: Data Visualization

After performing data preparation and statistical analysis, we can create various visualizations to gain deeper insights into the dataset.

### Deal Satisfaction by Country (Bar Chart)
To visualize deal satisfaction by country, we utilize a bar chart. The countries are plotted on the x-axis, while the average satisfaction scores are represented on the y-axis. The visualization employs `Matplotlib` and `Seaborn` for aesthetic styling.

**Key Code Snippet:**
```python
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))
plt.bar(x=averages_by_country.index, height=averages_by_country['deal_satisfaction'], color="#108A99")
plt.xticks(rotation=45, fontsize=13)
plt.yticks(fontsize=13)
plt.title("Deal Satisfaction by Country", fontsize=18, fontweight="bold")
plt.ylabel("Deal Satisfaction", fontsize=13)
sns.despine()
plt.savefig("deal_satisfaction_by_country_bar_chart.png")
plt.show()
```

### Age Distribution (Histogram)
A histogram is used to illustrate the distribution of ages at the time of property purchase. The `bins` parameter is set to segment age intervals appropriately.

**Key Code Snippet:**
```python
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))
plt.hist(data['age_at_purchase'], bins=10, color="#108A99")
plt.title("Age Distribution", fontsize=18, weight="bold")
plt.xlabel("Age", fontsize=13)
plt.ylabel("Number of Purchases", fontsize=13)
sns.despine()
plt.savefig("age_distribution_histogram.png")
plt.show()
```

### Segmentation by State (Pareto Chart)
A Pareto chart combines a bar and line chart to display the frequency and cumulative frequency of sales by state.

**Key Code Snippet:**
```python
sns.set_style("white")
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(sold_by_state.index, sold_by_state['frequency'], color="#108A99")
ax.set_ylabel("Apartments Sold", weight='bold', fontsize=13, color="#108A99")
ax2 = ax.twinx()
ax2.set_ylim(0, 1.1)
ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax2.plot(sold_by_state.index, sold_by_state["cumulative_frequency"], color="#E85D04", marker="D")
ax2.set_ylabel("Cumulative Frequency", color="#E85D04", weight="bold", fontsize=13)
ax.set_title("Segmentation of US Clients by State", fontsize=18, weight="bold", color="#108A99")
plt.savefig("US_segmentation_by_state_pareto_diagram.png")
plt.show()
```

### Total Sales per Year (Line Chart)
To understand yearly sales trends, a line chart is plotted using `Matplotlib`.

**Key Code Snippet:**
```python
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))
plt.plot(revenue_per_year['revenue$'], color='#108A99', linewidth=3)
plt.title("Total Revenue per Year (2004-2010)", fontsize=18, fontweight="bold")
plt.ylabel("Revenue $", fontsize=13)
sns.despine()
plt.savefig("total_revenue_per_year_in_M_line_chart.png")
plt.show()
```

### Total Sales by Year and Building (Stacked Area Chart)
A stacked area chart is used to represent the total sales per year, categorized by building type.

**Key Code Snippet:**
```python
colors = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]
labels = ['Building 1', 'Building 2', 'Building 3', 'Building 4', 'Building 5']
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))
plt.stackplot(stacked_area.index, stacked_area['building1'], stacked_area['building2'], stacked_area['building3'], stacked_area['building4'], stacked_area['building5'], colors=colors, edgecolor='none')
plt.xticks(stacked_area.index, rotation=45)
plt.legend(labels=labels, loc="upper left")
plt.ylabel("Number of Sales", fontsize=13)
plt.title("Total Number of Sales per Year by Building", fontsize=18)
sns.despine()
plt.savefig("total_sales_per_year_per_building_stacked_area_chart.png")
plt.show()
```

## Part 5: Data Interpretation

After conducting an in-depth analysis and visualization, we derive key insights into customer profiles and building characteristics.

### Customer Profile Analysis
A breakdown of customer demographics reveals that the most common age range for property purchases is 31-42 years, indicating financial stability and homeownership readiness.

| Age Interval | Sold |
|-------------|------|
| 19-25       | 4    |
| 25-31       | 16   |
| 31-36       | 26   |
| 36-42       | 33   |
| 42-48       | 22   |
| 48-54       | 22   |
| 54-59       | 22   |
| 59-65       | 11   |
| 65-71       | 16   |
| 71-76       | 6    |

Real estate companies can use these insights to refine marketing strategies and target ads effectively on digital platforms like Facebook, YouTube, and Google.

### Building Characteristics Analysis
Analyzing property sales data provides insights into the types of buildings sold most frequently and their average prices.

| Building | Sold | Mortgage |
|----------|------|---------|
| 1        | 46   | 14.0    |
| 2        | 54   | 18.0    |
| 3        | 53   | 15.0    |
| 4        | 23   | 9.0     |
| 5        | 19   | 6.0     |

Although Buildings 2 and 3 are the most frequently sold, Building 4 commands the highest price and customer satisfaction, indicating a preference for luxury properties.

### Sales by Country
Approximately 90% of the recorded sales come from the United States, highlighting a predominantly US-based market.

| Country | Sold | Mortgage |
|---------|------|---------|
| Belgium | 2    | 0.0     |
| Canada  | 7    | 0.0     |
| Denmark | 1    | 0.0     |
| Germany | 1    | 0.0     |
| Mexico  | 1    | 0.0     |
| Russia  | 4    | 1.0     |
| UK      | 2    | 0.0     |
| USA     | 177  | 61.0    |

### Yearly Sales Trends
2007 recorded the highest sales, aligning with the pre-market crash housing boom. Sales dropped sharply in 2009 due to economic downturns.

### Conclusion
This analysis provides valuable insights into customer demographics, property trends, and market fluctuations. Future studies with larger datasets can refine these conclusions, aiding real estate developers in making data-driven decisions.

## Contact

For any inquiries or feedback, feel free to reach out:

- **Email:** ialokchoubey833@gmail.com
- **LinkedIn:** https://www.linkedin.com/in/alok-6b0504216/
- **GitHub:** https://github.com/alokchoubey33/

## Acknowledgments

Thank you for taking the time to go through this project. I appreciate your interest and would love to hear your thoughts!

---


This README outlines our approach, methodology, and key findings in analyzing real estate sales data using statistical methods and data visualization techniques.
actual code can be found in the code section 











