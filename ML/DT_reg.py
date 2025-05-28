import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 读取数据
train_data = pd.read_csv('zhh_r_train.csv')
test_data = pd.read_csv('zhh_r_test.csv')
shuffled_train_data = train_data.sample(frac=1).reset_index(drop=True)
train_data=shuffled_train_data
shuffled_test_data = test_data.sample(frac=1).reset_index(drop=True)
test_data=shuffled_test_data
# 分离特征和目标值
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1].values
# 标准化特征值
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

# 预测
y_pred = dt_regressor.predict(X_test)

# 计算评价指标
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 计算MAPE（避免除以零错误）
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0  # 避免除以零
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)

# 输出结果
print(f"MSE: {mse:.8f}")
print(f"MAE: {mae:.8f}")
print(f"R2: {r2:.8f}")
print(f"MAPE: {mape:.8f}")