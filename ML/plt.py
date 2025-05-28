import matplotlib.pyplot as plt

# 定义模型和R2值
models = ["TabPFN", "Saint", "AutoXGB", "LR", "Ridge", "DT", "RF", "Tabnet"]
r2_values = [0.999862863, 0.9062, 0.995, 0.99970105, 0.99954404, 0.99813407, 0.999, 0.9439]

# 将模型和R2值按从小到大排序
sorted_data = sorted(zip(models, r2_values), key=lambda x: x[1])
sorted_models = [x[0] for x in sorted_data]
sorted_r2 = [x[1] for x in sorted_data]

# 计算RSS值（1 - R2）
rss_values = [1 - r2 for r2 in sorted_r2]

# 创建折线图
plt.figure(figsize=(10, 6))

# 绘制折线
plt.plot(sorted_models, rss_values, marker='o', linestyle='-', color='b')

# 将最低RSS值的点用五角星标记
min_rss_index = rss_values.index(min(rss_values))
plt.scatter(sorted_models[min_rss_index], min(rss_values), marker='*', color='r', s=200)

# 添加标签和标题
plt.xlabel('Models')
plt.ylabel('RSS (Residual Sum of Squares)')
plt.title('RSS Values of Regression Models')

# 显示图表
plt.tight_layout()
plt.savefig("rss_line_chart.png", dpi=300, bbox_inches='tight')
plt.show()