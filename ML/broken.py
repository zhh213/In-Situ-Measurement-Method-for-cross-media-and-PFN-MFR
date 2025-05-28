import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = 'updated_0.000118predictions_vs_y_test_with_features.xlsx'
df = pd.read_excel(file_path)


# 根据列名提取数据
jihe_0 = df['jihe_0'].dropna().values  # 第二列：jihe_0，去除NaN值
prediction_0 = df['prediction_0'].dropna().values  # 第三列：prediction_0，去除NaN值
jihe_1 = df['jihe_1'].dropna().values  # 第六列：jihe_1，去除NaN值
prediction_1 = df['prediction_1'].dropna().values  # 第七列：prediction_1，去除NaN值

# 创建序号
indices_0 = range(1, len(jihe_0) + 1)
indices_1 = range(1, len(jihe_1) + 1)

# 创建一个包含两个子图的图表
plt.figure(figsize=(12, 6))

# 第一个子图：jihe_0 vs prediction_0
plt.subplot(1, 2, 1)
plt.plot(indices_0, jihe_0, marker='X', color='red', label='Error before compensation', markersize=4, linewidth=0.6, linestyle='-')
plt.plot(indices_0, prediction_0, marker='X', color='blue', label='Error after compensation', markersize=4, linewidth=0.6, linestyle='-')
plt.fill_between(indices_0, df['jihe_0'].min(), df['jihe_0'].max(), color='red', alpha=0.1)
plt.fill_between(indices_0, df['prediction_0'].min(), df['prediction_0'].max(), color='blue', alpha=0.1)
plt.title('uniform liquid film', fontsize=12)
plt.xlabel('Index', fontsize=10)
plt.ylabel('Error', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 第二个子图：jihe_1 vs prediction_1，范围根据第六列的长度
plt.subplot(1, 2, 2)
plt.plot(indices_1, jihe_1, marker='X', color='red', label='Error before compensation', markersize=4, linewidth=0.6, linestyle='-')
plt.plot(indices_1, prediction_1, marker='X', color='blue', label='Error after compensation', markersize=4, linewidth=0.6, linestyle='-')
plt.fill_between(indices_1, df['jihe_1'].min(), df['jihe_1'].max(), color='red', alpha=0.1)
plt.fill_between(indices_1, df['prediction_1'].min(), df['prediction_1'].max(), color='blue', alpha=0.1)
plt.title('elliptical liquid film', fontsize=12)
plt.xlabel('Index', fontsize=10)
plt.ylabel('Error', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 设置第二个子图的横轴范围，两边留出一定的空白
plt.xlim(min(indices_1) - 5, max(indices_1) + 5)

# 调整子图间距
plt.tight_layout()

# 显示图表
plt.show()