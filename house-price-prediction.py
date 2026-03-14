import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
data = pd.read_csv("train.csv")

# Display first few rows
print("Dataset Preview:")
print(data.head())

X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = data['SalePrice']


# Check for missing values
print("\nMissing values:")
print(X.isnull().sum())


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)


# Predict house prices
predictions = model.predict(X_test)


# Evaluate model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)


# Show model coefficients
print("\nModel Coefficients:")
print("Square Footage Coefficient:", model.coef_[0])
print("Bedrooms Coefficient:", model.coef_[1])
print("Bathrooms Coefficient:", model.coef_[2])
print("Intercept:", model.intercept_)

new_house = pd.DataFrame([[2000, 3, 2]], columns=['GrLivArea', 'BedroomAbvGr', 'FullBath'])
predicted_price = model.predict(new_house)

print("\nPredicted Price for 2000 sqft, 3 bed, 2 bath house:")
print(predicted_price[0])


# Visualization: Actual vs Predicted Prices
plt.figure()
plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()