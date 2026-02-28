# Car Price Prediction using Machine Learning
This project aims to predict the Selling Price of used cars by leveraging advanced Machine Learning techniques.
Using a dataset of car features like present price, kilometers driven, fuel type, and age, we developed a regression model to provide accurate valuations.

# Project Overview
The automotive market is dynamic, and determining the fair resale value of a vehicle is a complex task.
This project uses a Random Forest Regressor to analyze historical data and provide a data-driven pricing solution.

# Tech Stack
Language: Python
Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
Algorithm: Random Forest Regression
Environment: VS Code / Jupyter Notebook

# Key Features & Workflow
1. Data Preprocessing & Cleaning
Feature Engineering: Created a Car_Age feature from the manufacturing year to better capture the vehicle's depreciation.
Categorical Encoding: Converted non-numeric data (Fuel Type, Transmission, Selling Type) into machine-readable formats using One-Hot Encoding.
Scaling: Handled outliers and ensured features like Present_Price and Driven_kms were properly processed.

2. Exploratory Data Analysis (EDA)
Analyzed the correlation between features using a Heatmap.
Identified that Present_Price has the strongest positive correlation with the Selling_Price.

3. Model Selection & Training
We chose the Random Forest Regressor over simple Linear Regression because it:
Handles non-linear relationships effectively.
Is robust to outliers in the data.
Reduces the risk of overfitting through ensemble learning.

# Performance Results
The model achieved high accuracy on the test set:
R² Score: 0.959 (95.9% variance explained)
Mean Absolute Error (MAE): 0.63 (Lakhs)

# Real-World Applications
Automobile Dealerships: Automating the trade-in valuation process.
E-commerce Platforms: Providing "Instant Quotes" for users looking to sell their cars online.
Insurance Companies: Estimating the Insured Declared Value (IDV) of vehicles.
