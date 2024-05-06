import streamlit as st
import pandas as pd

# Load data
df_train = pd.read_csv('./Dataset/Train.csv')

# Categorize data types
numerical_columns = df_train.select_dtypes(include=['int', 'float']).columns.tolist()
categorical_columns = df_train.select_dtypes(include=['object']).columns.tolist()

# Set Streamlit configuration options
st.set_page_config(
    page_title='View Data',
    page_icon='',
    layout='wide'
)

# Main UI layout
st.subheader("Data Page")

selected_feature_type = st.selectbox("Please select group of features or Target",
                                     options=['All Features', 'Numeric Features',
                                              'Categorical Features'],
                                     key='selected_features', index=0)

if selected_feature_type == 'All Features':
    st.write(df_train.head(500000))  # Displaying the first 500,000 rows
elif selected_feature_type == 'Numeric Features':
    st.write(df_train[numerical_columns].head(500000))  # Displaying the first 500,000 rows
elif selected_feature_type == 'Categorical Features':
    st.write(df_train[categorical_columns].head(500000))  # Displaying the first 500,000 rows
