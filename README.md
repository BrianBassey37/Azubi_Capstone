This is a Capstone Project by Team Radon comprising the following members:

- Brian Edem Bassey - Team Leader
- Caroline Muinde
- Esther Afari
- Tirusew Ayenew Cheru
- Akosua Danso

# Business Understanding (CRISP-DM) for Customer Churn Prediction Challenge for Azubian

## Business Objective

Develop a machine learning model to predict the likelihood of each customer "churning" (becoming inactive and not making any transactions for 90 days). This will enable Expresso Telecom to proactively identify at-risk customers and implement targeted retention strategies to improve customer loyalty and reduce churn rates.

## Stakeholders

1. Expresso Telecom Management Team: Responsible for strategic decision-making and resource allocation based on churn prediction insights.
2. Marketing Department: Utilizes churn predictions to design and implement targeted marketing campaigns to retain at-risk customers.
3. Customer Service Team: Leverages churn predictions to prioritize and personalize interactions with customers, addressing their concerns and enhancing satisfaction.
4. Data Analytics Team: Responsible for developing, deploying, and maintaining the churn prediction model.

## Success Criteria

1. Reduce Churn Rate: Achieve a measurable reduction in churn rate by accurately predicting and proactively addressing customer churn.
2. Model Performance: Achieve a high Area Under the Curve (AUC) score as the evaluation metric, indicating the effectiveness of the churn prediction model.
3. Business Impact: Enhance customer retention, increase revenue, and improve overall customer satisfaction and loyalty.

## Data Understanding

- Data Sources: Historical customer transaction data from Expresso Telecom's databases, including customer demographics, usage patterns, transaction history, and churn status.

## Hypotheses

| Hypothesis | Null Hypothesis (H0) | Alternative Hypothesis (H1) |
|------------|-----------------------|-----------------------------|
| 1. Customer Region and Churn Rate | A customers region will not affect churn rate | A customer's region will affect churn rate . |
| 2. Customer Tenure and Churn Rate | There is no significant relationship between customer tenure and churn rate. | Customers with longer tenure are less likely to churn compared to those with shorter tenure. |

## Analytical Questions

1. What is the overall churn rate for Expresso Telecom?
2. Which top 5 packages have more customer subscription?
3. How does customer tenure vary among churned and non-churned customers?
4. How do customers' subscription packages (top packs) relate to their likelihood of churning?
5. How does the churn rate vary across different regions or locations?

## Exploratory Data Analysis (EDA)

1. Load the Dataset: Load the dataset into python data analysis environment.
2. Review the Data: Display the first few rows of the dataset to understand its structure and contents. Check the column names and data types.
3. Handle Missing Values: Identify and handle missing values in the dataset. Decide whether to impute missing values or remove rows/columns with missing values based on the context.
4. Check Data Integrity: Look for duplicate rows and remove them if necessary. Check for data integrity issues such as outliers, incorrect data types, or inconsistent values.
5. Summary Statistics: Compute summary statistics (e.g., mean, median, standard deviation) for numerical variables. For categorical variables, calculate the frequency of each category.
6. Univariate Analysis: For numerical variables, create histograms to visualize the distribution. For categorical variables, create bar plots to visualize the frequency of each category. Compute measures of central tendency and dispersion.
7. Bivariate Analysis: Explore the relationship between pairs of variables using scatter plots for numerical variables and box plots for categorical variables. Calculate correlation coefficients to measure the strength and direction of linear relationships between numerical variables.
8. Multivariate Analysis: Use heatmaps or pair plots to visualize relationships between multiple variables. Identify potential interactions and patterns among variables.
9. Feature Engineering: Create new features that might be useful for analysis or modeling based on existing variables. Transform variables if necessary (e.g., log transformation for skewed distributions).
10. Visualization: Use various visualization techniques (e.g., scatter plots, box plots, heatmaps) to gain insights and communicate findings effectively.
11. Statistical Tests: Perform statistical tests (e.g., t-tests, chi-square tests) to investigate relationships and differences between variables.
12. Documentation: Document findings, insights, and decisions made during the EDA process. Create visualizations and summaries that can be easily understood by stakeholders.
13. Iterate: EDA is an iterative process, so revisit earlier steps as you gain more insights and refine your analysis.

## Data Preparation

- Data Cleaning: Handle missing values, remove duplicates, and correct errors in the dataset to ensure data quality and reliability.
- Feature Engineering: Create new features or transform existing ones to capture relevant information for churn prediction, such as customer tenure, average transaction value, and frequency of interactions.
- Feature Selection: Identify key features that are likely to influence churn behavior based on domain knowledge and insights gained from EDA.

## Modeling

- Model Selection: Choose appropriate machine learning algorithms for churn prediction, such as logistic regression, decision trees, or gradient boosting.
- Model Training: Train predictive models using historical customer data, and optimize model parameters to maximize predictive performance.
- Model Evaluation: Assess model performance using the Area Under the Curve (AUC) metric, validating the model's ability to distinguish between churn and non-churn instances.

## Evaluation

- Business Impact Assessment: Evaluate the potential impact of using the churn prediction model on reducing customer attrition and improving retention.
- Model Performance Review: Validate model results against business objectives and stakeholder expectations, ensuring alignment with success criteria.
- Scalability and Robustness: Assess the scalability and robustness of the model for deployment in production environments.

## Deployment

- Model Integration: Integrate the churn prediction model into Expresso Telecom's operational systems, ensuring scalability, reliability, and real
