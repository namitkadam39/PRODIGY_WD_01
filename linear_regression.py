import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


data = {
    'sqft': [800,900,1000,1100,1200,1400,1500,1600,1700,1800,2000,2200,2400,2600,2800],
    'bedrooms': [2,2,2,3,3,3,3,4,4,4,5,5,5,6,6],
    'bathrooms': [1,1,2,2,2,2,3,2,3,3,3,4,4,4,5],
    'price': [150000,180000,200000,230000,250000,280000,310000,330000,360000,380000,420000,460000,500000,540000,580000]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

X = df[['sqft', 'bedrooms', 'bathrooms']]
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\nPredicted Prices:", y_pred)
print("Actual Prices:", list(y_test))


mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)


new_house = pd.DataFrame([[2000, 3, 2]], 
                         columns=['sqft', 'bedrooms', 'bathrooms'])

predicted_price = model.predict(new_house)

print("\nPredicted price for new house:", predicted_price[0])

print("\nModel Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
plt.scatter(df['sqft'], df['price'])
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.title("House Price vs Square Feet")
plt.show()
