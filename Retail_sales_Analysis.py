
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load dataset
url = r"C:\Users\Verma\OneDrive\Desktop\Dataset\retail_sales_dataset.csv"

data = pd.read_csv(url)

# Display the first few rows of the dataset
data.head()


# Check for missing values
print(data.isnull().sum())

# Drop rows with missing values (if any)
data.dropna(inplace=True)

# Convert date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Remove duplicates
data.drop_duplicates(inplace=True)

# Ensure numeric columns are in the correct format
data['Quantity'] = pd.to_numeric(data['Quantity'], errors='coerce')
data['Price per Unit'] = pd.to_numeric(data['Price per Unit'], errors='coerce')
data['Total Amount'] = pd.to_numeric(data['Total Amount'], errors='coerce')

# Summary of the cleaned data
print(data.info())


# Descriptive statistics
print(data.describe())

# Sales by Date
daily_sales = data.groupby('Date')['Total Amount'].sum().reset_index()

# Sales by Product Category
category_sales = data.groupby('Product Category')['Total Amount'].sum().sort_values(ascending=False)


# Extract year and month from 'Date'
data['YearMonth'] = data['Date'].dt.to_period('M')

# Assign cohort group (based on the first purchase)
data['CohortGroup'] = data.groupby('Customer ID')['YearMonth'].transform('min')

# Calculate the number of periods since the cohort start
data['CohortPeriod'] = (data['YearMonth'] - data['CohortGroup']).apply(lambda x: x.n)

# Cohort Analysis by Retention Rate
cohort_data = data.groupby(['CohortGroup', 'CohortPeriod']).agg({'Customer ID': pd.Series.nunique})
cohort_data = cohort_data.rename(columns={'Customer ID': 'TotalCustomers'}).reset_index()

# Pivot the data to create a cohort table
cohort_pivot = cohort_data.pivot_table(index='CohortGroup', columns='CohortPeriod', values='TotalCustomers')
cohort_size = cohort_pivot.iloc[:,0]
retention_rate = cohort_pivot.divide(cohort_size, axis=0)

# Display retention rate
print(retention_rate)



# Recency: Calculate the number of days since the last purchase
current_date = data['Date'].max()
rfm = data.groupby('Customer ID').agg({
    'Date': lambda x: (current_date - x.max()).days,
    'Transaction ID': 'count',
    'Total Amount': 'sum'
})

# Rename columns
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Assign RFM scores
rfm['R'] = pd.qcut(rfm['Recency'], 4, ['1','2','3','4'])
rfm['F'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, ['4','3','2','1'])
rfm['M'] = pd.qcut(rfm['Monetary'], 4, ['4','3','2','1'])

# Calculate RFM Score
rfm['RFM_Score'] = rfm.R.astype(str) + rfm.F.astype(str) + rfm.M.astype(str)

# Display the RFM table
print(rfm.head())



import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(daily_sales['Date'], daily_sales['Total Amount'])
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()



plt.figure(figsize=(10, 6))
category_sales.plot(kind='bar')
plt.title('Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales')
plt.show()


plt.figure(figsize=(12, 8))
sns.heatmap(retention_rate, annot=True, cmap='coolwarm')
plt.title('Cohort Analysis - Retention Rate')
plt.ylabel('Cohort Group')
plt.xlabel('Cohort Period')
plt.show()



from prophet import Prophet

daily_sales = data.groupby('Date')['Total Amount'].sum().reset_index()

# Rename columns for Prophet
prophet_data = daily_sales.rename(columns={'Date': 'ds', 'Total Amount': 'y'})

# Verify the data
print(prophet_data.head())
print(prophet_data.columns)

# Initialize the Prophet model
model = Prophet()

# Fit the model
model.fit(prophet_data)

# Create future dates for prediction
future = model.make_future_dataframe(periods=90)

# Predict future sales
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.title('Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()




