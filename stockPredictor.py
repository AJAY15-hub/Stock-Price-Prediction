import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


np.random.seed(42)
trend = np.linspace(4000, 3250, 60)
noise = np.random.normal(0, 50, 60)
stock_prices = trend + noise


df = pd.DataFrame({
    'price': stock_prices,
    'day': np.arange(len(stock_prices))
})


df['lag_1'] = df['price'].shift(1) 
df['lag_2'] = df['price'].shift(2)  
df['ma_5'] = df['price'].rolling(5).mean()  
df = df.dropna()  


X = df[['day', 'lag_1', 'lag_2', 'ma_5']]
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)


model = LinearRegression()
model.fit(X_train, y_train)


train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print(f'Train MAE: {mean_absolute_error(y_train, train_preds):.2f}')
print(f'Test MAE: {mean_absolute_error(y_test, test_preds):.2f}')
print(f'Train RMSE: {np.sqrt(mean_squared_error(y_train, train_preds)):.2f}')
print(f'Test RMSE: {np.sqrt(mean_squared_error(y_test, test_preds)):.2f}')


future_days = 30
last_data = X.iloc[-1].values.reshape(1, -1)
future_predictions = []

for _ in range(future_days):
    next_day = model.predict(last_data)[0]
    future_predictions.append(next_day)
    
    last_data = np.array([
        last_data[0, 0] + 1,  
        next_day,              
        last_data[0, 1],        
        (last_data[0, 3] * 4 + next_day) / 5 
    ]).reshape(1, -1)

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(df['day'], df['price'], 'b-', label='Actual Price')
plt.plot(X_train['day'], train_preds, 'c-', alpha=0.7, label='Train Predictions')
plt.plot(X_test['day'], test_preds, 'r-', label='Test Predictions')
plt.plot(range(len(stock_prices), len(stock_prices) + future_days), 
         future_predictions, 'g--', label='Future Predictions')

plt.title("Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
