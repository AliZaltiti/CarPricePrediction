# Importing necessary libraries
import pandas as pd  # for data manipulation
import numpy as np   # for numerical operations
import matplotlib.pyplot as plt  # for plotting graphs

# Importing scikit-learn tools
from sklearn.model_selection import train_test_split  # split data into train and test sets
from sklearn.preprocessing import LabelEncoder  # convert categorical data to numeric
from sklearn.linear_model import LinearRegression  # Linear Regression model
from sklearn.metrics import mean_squared_error, r2_score  # evaluate model performance

# Load the dataset
df = pd.read_csv("CarPrice_Assignment.csv")

# Display first 5 rows of the dataset
print("First 5 rows of dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in dataset:")
print(df.isnull().sum())

# Encode categorical variables (convert text to numbers)
le = LabelEncoder()
categorical_cols = df.select_dtypes(include='object').columns  # find all categorical columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Define features (X) and target (y)
X = df.drop("price", axis=1)  # all columns except 'price'
y = df["price"]  # target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Plot actual vs predicted prices
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()
