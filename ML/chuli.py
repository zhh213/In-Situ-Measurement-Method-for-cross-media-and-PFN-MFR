import pandas as pd

# 读取Excel文件
file_path = '0.000118predictions_vs_y_test_with_features.xlsx'
df = pd.read_excel(file_path)

# 计算jihe_0、jihe_1、prediction_0、prediction_1列，取绝对值
df['jihe_0'] = df.apply(lambda row: abs(row['jihe'] - row['y_test']) if row['cat'] == 0 else None, axis=1)
df['jihe_1'] = df.apply(lambda row: abs(row['jihe'] - row['y_test']) if row['cat'] == 1 else None, axis=1)
df['prediction_0'] = df.apply(lambda row: abs(row['predictions'] - row['y_test']) if row['cat'] == 0 else None, axis=1)
df['prediction_1'] = df.apply(lambda row: abs(row['predictions'] - row['y_test']) if row['cat'] == 1 else None, axis=1)

# 去除空白，使数据连续，并对每一列单独排序
df['jihe_0'] = df['jihe_0'].dropna().sort_values().reset_index(drop=True)
df['jihe_1'] = df['jihe_1'].dropna().sort_values().reset_index(drop=True)
df['prediction_0'] = df['prediction_0'].dropna().sort_values().reset_index(drop=True)
df['prediction_1'] = df['prediction_1'].dropna().sort_values().reset_index(drop=True)

# 打印结果
print(df[['jihe_0', 'jihe_1', 'prediction_0', 'prediction_1']])

# 保存结果到新的Excel文件
output_file_path = 'updated_0.000118predictions_vs_y_test_with_features.xlsx'
df.to_excel(output_file_path, index=False)