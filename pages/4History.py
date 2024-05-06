import streamlit as st
import pandas as pd

def main():
    st.subheader("Churn Prediction History")

    # Read history data
    df_history = pd.read_csv("history.csv")

    # Display history data
    st.write(df_history)

if __name__ == "__main__":
    main()
