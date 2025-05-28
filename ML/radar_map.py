import numpy as np
import matplotlib.pyplot as plt


models = {
    "K-NN": [0.8261, 0.8066, 0.805, 0.8369, 0.7958, 0.8261],
    "SVM": [0.963, 0.9989, 0.99, 0.9987, 0.9677, 0.9783],
    "Ridge": [0.9074, 0.9647, 0.8600, 0.9593, 0.8409, 0.8043],
    "RF": [0.6852, 0.7538, 0.7050, 0.6965, 0.6943, 0.7283],
    "TabPFN": [0.9907, 0.9999, 0.9950, 0.9999, 0.9891, 0.9891],
    "tltd": [0.8187, 0.9003, 0.8294, 0.8974, 0.8294, 0.8404],
    "GBR": [0.7037, 0.7878, 0.725, 0.7604, 0.715, 0.75],
    "SAINT": [0.4050, 0.7840, 0.6705, 0.8460, 0.7520, 0.8630]
}

categories = ['Specificity', 'AUROC', 'Accuracy', 'AUPRC', 'F1', 'RECALL']
num_categories = len(categories)


angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()


for model in models:
    models[model] += models[model][:1]

angles += angles[:1]


fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))


ax.set_title('Comparison of Classification Models', y=1.15, fontsize=18, fontweight='bold')

# 隐藏最外面的环形线
ax.spines['polar'].set_visible(False)

# 设置同心圆
ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.2', '0.4', '0.6', '0.8', '1.0'], angle=0, fontsize=10)
ax.set_rlabel_position(30)

# 定义颜色列表
colors = {
    'K-NN': 'b',
    'SVM': 'gold',
    'Ridge': 'r',
    'RF': 'c',
    'TabPFN': 'hotpink',
    'tltd': 'm',
    'GBR': 'k',
    'SAINT': 'g'
}

# 定义线条样式
linestyles = {
    'SVM': '.-',
    'TabPFN': '.-',
}

# 定义线条宽度
linewidths = {
    'SVM': 3,
    'TabPFN': 3
}

# 绘制每个模型的雷达图
for model, values in models.items():
    color = colors[model]
    linestyle = linestyles.get(model, '-')
    linewidth = linewidths.get(model, 1)

    ax.plot(angles, values, linestyle, linewidth=linewidth, label=model, color=color, markersize=8)

# 将图例放置在雷达图外侧
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.25), fontsize=10)

# 设置标签位置在雷达图外侧
for i, category in enumerate(categories):
    theta = angles[i]
    radius = 1.1  # 标签显示在雷达图外侧的位置（1.1倍半径）
    ax.text(theta, radius, category, ha='center', va='center',
            fontsize=12, fontweight='bold')

# 隐藏原来的标签
ax.set_xticks([])

# 显示图表
plt.tight_layout()
plt.savefig("radar_chart.png", dpi=300, bbox_inches='tight')
plt.show()