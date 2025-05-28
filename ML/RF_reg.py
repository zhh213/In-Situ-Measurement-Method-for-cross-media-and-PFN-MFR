import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

train_data = pd.read_csv('zhh_r_train.csv')
test_data = pd.read_csv('zhh_r_test.csv')
shuffled_train_data = train_data.sample(frac=1).reset_index(drop=True)
train_data=shuffled_train_data
shuffled_test_data = test_data.sample(frac=1).reset_index(drop=True)
test_data=shuffled_test_data

X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_regressor = RandomForestRegressor(n_estimators=1, random_state=42)
rf_regressor.fit(X_train, y_train)


y_pred = rf_regressor.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0  # 避免除以零
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"MSE: {mse:.8f}")
print(f"MAE: {mae:.8f}")
print(f"R2: {r2:.8f}")
print(f"MAPE: {mape:.8f}")