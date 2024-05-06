import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df_train = pd.read_csv('./Dataset/Train.csv')

# Drop 'user_id' column from categorical_columns list
categorical_columns = df_train.select_dtypes(include=['object']).columns.tolist()
if 'user_id' in categorical_columns:
    categorical_columns.remove('user_id')

# Categorize numerical columns
numerical_columns = df_train.select_dtypes(include=['int', 'float']).columns.tolist()

# Set Streamlit configuration options
st.set_page_config(
    page_title='Dashboard',
    page_icon='',
    layout='wide'
)

# Main UI layout
st.subheader("Dashboard page of the application")

# Key Performance Indicators (KPIs)
st.write("#### Key Performance Indicators (KPIs)")

# Display mean of each numerical column with two digits
for column in numerical_columns:
    mean_value = df_train[column].mean()
    mean_value_formatted = "{:.2f}".format(mean_value)
    st.write(f"Mean of {column}: **{mean_value_formatted}**")

# Display total revenue in billions of dollars
total_revenue = df_train['REVENUE'].sum() / 1_000_000_000  # Convert to billions
st.write(f"Total of REVENUE: **{total_revenue:.2f}** Billion dollars")

# Histogram of Numeric Features
st.subheader("Numeric Features Visualization")
selected_numeric_column = st.selectbox("Select a numeric column:", numerical_columns)
fig, ax = plt.subplots()
ax.hist(df_train[selected_numeric_column], bins=30, edgecolor='black')
st.pyplot(fig)

# Bar Chart of Categorical Features
st.subheader("Categorical Features Visualization")
selected_categorical_column = st.selectbox("Select a categorical column:", categorical_columns)
value_counts = df_train[selected_categorical_column].value_counts()
fig, ax = plt.subplots()
ax.bar(value_counts.index, value_counts.values)
ax.tick_params(axis='x', rotation=45)  # Remove 'ha' parameter
st.pyplot(fig)
