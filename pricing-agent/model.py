import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample dataset
data = {
    "demand": [50, 60, 70, 80, 90],
    "competitor_price": [2000, 2100, 2200, 2300, 2400],
    "price": [1800, 1900, 2000, 2100, 2200]
}

df = pd.DataFrame(data)

X = df[["demand", "competitor_price"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "pricing_model.pkl")

print("Model trained successfully!")