import streamlit as st

def main():
    st.header("Welcome to Customer Churn Prediction App")

    st.write("The purpose of this application is to provide a user-friendly churn prediction platform and a user-friendly dashboard to easily understand the current data and check predictions.")

    st.subheader("Contents of the Application")

    st.markdown("- **Home Page:** The home page of the application shows you the end users of this application, the contents of the application")
    st.markdown("- **Data Page:** Provides access to view the raw data used for analysis and prediction. Users can explore the dataset to understand its structure, features, and content.")
    st.markdown("- **Dashboard Page:** Presents various analysis insights derived from the data. Includes visualizations and summaries to showcase key metrics such as churn rates, top subscription packages, customer tenure distribution, and regional churn patterns. Offers an intuitive interface for users to interact with the data and gain valuable insights.")
    st.markdown("- **Predict Page:** Allows users to input their data (such as customer features) into the machine learning model to predict churn likelihood. Users can input relevant features, such as customer tenure, transaction frequency, and subscription details. Upon submission, the model predicts whether the customer is likely to churn or not, along with a probability score.")
    st.markdown("- **History Page:** Stores the history of churn predictions made by users. Users can review past predictions, including the input features and the corresponding churn prediction results. Provides a comprehensive record of churn prediction activities for reference and analysis.")

if __name__ == "__main__":
    main()