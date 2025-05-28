import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn.regressor import TabPFNRegressor
import pandas as pd
from DataAugmenter import DataAugmenter

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

augmenter_mixup = DataAugmenter(method="mixup", alpha=0.1)
augmenter_hidden_mix = DataAugmenter(method="hidden_mix", alpha=0.1)
augmenter_mask_token = DataAugmenter(method="mask_token", mask_ratio=0.1)
augmenter_vime = DataAugmenter(method="vime", mask_ratio=0.1)



X_mixup, y_mixup = augmenter_mixup.augment(X_train, y_train)
X_mask_token, _ = augmenter_mask_token.augment(X_train)
X_vime, _ = augmenter_vime.augment(X_train)
X_train_augmented = np.vstack([X_mixup, X_mask_token])
y_train_augmented = np.hstack([y_mixup, y_train])


reg = TabPFNRegressor()
reg.fit(X_train_augmented, y_train_augmented)

predictions = reg.predict(X_test)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mape = mean_absolute_percentage_error(y_test, predictions)


print(f"Mean Absolute Percentage Error (MAPE): {mape:.6f}%")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("R-squared (R^2):", r2_score(y_test, predictions))


df = pd.DataFrame({ "y_test": y_test, "predictions": predictions})
if isinstance(X_test, np.ndarray):
    X_test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
else:
    X_test_df = X_test.copy()
df = pd.concat([X_test_df, df], axis=1)
df.to_excel("predictions_vs_y_test_with_features.xlsx", index=False)

